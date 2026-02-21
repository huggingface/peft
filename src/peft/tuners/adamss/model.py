# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
    TRANSFORMERS_MODELS_TO_ADAMSS_TARGET_MODULES_MAPPING,
)

from .config import AdamssConfig
from .layer import AdamssLayer, Linear


class AdamssModel(BaseTuner):
    """
    Creates Adamss (Adaptive Multi-Subspaces) model from a pretrained model.

    The method decomposes weight matrices using SVD and clusters the decomposed space
    into multiple trainable subspaces for parameter-efficient fine-tuning.

    Args:
        model (`torch.nn.Module`): The model to be adapted.
        config (`AdamssConfig`): The configuration of the Adamss model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The Adamss model.

    Example:
        ```python
        >>> from transformers import AutoModelForImageClassification
        >>> from peft import AdamssConfig, get_peft_model

        >>> config = AdamssConfig(
        ...     r=500,
        ...     num_subspaces=5,
        ...     target_modules=["query", "value"],
        ... )

        >>> model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
        >>> adamss_model = get_peft_model(model, config)
        ```

    **Attributes**:
        - **model** ([`~torch.nn.Module`]) -- The model to be adapted.
        - **peft_config** ([`AdamssConfig`]): The configuration of the Adamss model.
    """

    prefix: str = "adamss_"
    tuner_layer_cls = (AdamssLayer,)
    target_module_mapping = TRANSFORMERS_MODELS_TO_ADAMSS_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False, state_dict: Optional[dict] = None) -> None:
        # Initialize ASA tracking before BaseTuner injects adapters so attribute exists.
        self._asa_total_subspaces = {}
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage, state_dict=state_dict)

    @staticmethod
    def _check_target_module_exists(adamss_config, key):
        """Helper to check if target module matches the pattern."""
        return check_target_module_exists(adamss_config, key)

    def _create_and_replace(
        self,
        adamss_config: AdamssConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        **optional_kwargs,
    ):
        """
        Create and replace target module with Adamss-adapted module.
        """
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Check if already an Adamss layer
        if isinstance(target, AdamssLayer):
            target.update_layer(
                adapter_name,
                adamss_config.r,
                adamss_config.num_subspaces,
                adamss_config.subspace_rank,
                adamss_config.init_weights,
                adamss_config.use_asa,
                inference_mode=adamss_config.inference_mode,
            )
            # set_adapter is called inside update_layer, but we need to handle
            # the case where adapter is not in active_adapters
            if adapter_name not in self.active_adapters:
                # Ensure the new adapter is frozen if not active
                target.set_adapter(self.active_adapters, inference_mode=adamss_config.inference_mode)
        else:
            # Create new Adamss layer
            new_module = self._create_new_module(
                adamss_config, adapter_name, target, inference_mode=adamss_config.inference_mode, **optional_kwargs
            )
            self._record_asa_total_subspaces(adapter_name, adamss_config, new_module)
            # requires_grad is handled inside _create_new_module via set_adapter
            self._replace_module(parent, target_name, new_module, target)

    def _record_asa_total_subspaces(self, adapter_name: str, adamss_config: AdamssConfig, module: AdamssLayer) -> None:
        """Track total subspaces per adapter so the ASA schedule matches adamss_pkg."""
        if not hasattr(module, "num_subspaces"):
            return
        layer_num_subspaces = module.num_subspaces.get(adapter_name, 0)
        if layer_num_subspaces is None or layer_num_subspaces <= 0:
            return
        prev_total = self._asa_total_subspaces.get(adapter_name, 0)
        self._asa_total_subspaces[adapter_name] = prev_total + layer_num_subspaces

    def _create_new_module(
        self,
        adamss_config: AdamssConfig,
        adapter_name: str,
        target: nn.Module,
        **kwargs,
    ) -> nn.Module:
        """
        Create a new Adamss module based on the target module type.
        """
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = Linear(
                target,
                adapter_name,
                r=adamss_config.r,
                num_subspaces=adamss_config.num_subspaces,
                subspace_rank=adamss_config.subspace_rank,
                init_weights=adamss_config.init_weights,
                use_asa=adamss_config.use_asa,
                use_dynamic_rank=adamss_config.use_dynamic_rank,
                svd_threshold=adamss_config.svd_threshold,
                **kwargs,
            )
        else:
            raise TypeError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear` is supported."
            )

        return new_module

    def update_and_allocate(self, global_step: int) -> None:
        """
        Update importance scores and apply ASA masking (if enabled).

        This method should be called in **every** training step after ``loss.backward()``
        and before ``optimizer.zero_grad()`` when ASA is enabled.  Internally it:

        1. Accumulates importance scores via EMA every step during the warmup period.
        2. At mask intervals, applies global top-K masking and resets importance.

        This is the single entry point for ASA – using the :class:`AdamssAsaCallback`
        with HuggingFace ``Trainer`` simply delegates to this method.  For custom
        training loops, call this directly instead of the callback.

        Args:
            global_step (`int`): The current training step.

        Example::

            for step, batch in enumerate(dataloader):
                loss = model(**batch).loss
                loss.backward()
                optimizer.step()
                model.base_model.update_and_allocate(step)
                optimizer.zero_grad()
        """
        for adapter_name in self.active_adapters:
            if adapter_name not in self.peft_config:
                continue

            config = self.peft_config[adapter_name]
            if not config.use_asa:
                continue

            within_warmup = config.init_warmup <= global_step <= config.final_warmup

            # --- collect ASA layers once ---
            asa_layers = [
                m for m in self.model.modules() if isinstance(m, AdamssLayer) and adapter_name in m.exp_avg_ipt
            ]
            if not asa_layers:
                continue

            # Step 1: accumulate importance EVERY step during warmup
            if within_warmup:
                for module in asa_layers:
                    module.update_importance(adapter_name, config.asa_importance_beta, config.asa_uncertainty_beta)

            # Step 2: at mask intervals → schedule, global mask, then reset
            is_mask_interval = global_step % config.mask_interval == 0
            if within_warmup and is_mask_interval:
                current_target = self._schedule_threshold(global_step, config)
                if current_target is not None:
                    self._global_mask_to_target(current_target, adapter_name, asa_layers)

                # Reset importance AFTER masking for fresh accumulation
                for module in asa_layers:
                    module.reset_importance(adapter_name)

    # ------------------------------------------------------------------
    # ASA helpers
    # ------------------------------------------------------------------

    def _schedule_threshold(self, step: int, config) -> Optional[int]:
        """Calculate current target subspaces based on warmup schedule."""
        total = self._asa_total_subspaces.get(next(iter(self.active_adapters), "default"), 0)
        if total == 0:
            total = self._get_asa_total_subspaces()
        if total == 0:
            return None

        if step < config.init_warmup:
            return None
        elif step <= config.final_warmup:
            mul_coeff = 1.0 - (step - config.init_warmup) / (config.final_warmup - config.init_warmup)
            mul_coeff = max(0.0, min(1.0, mul_coeff))
            exponent = getattr(config, "asa_schedule_exponent", 3.0)
            return int(config.asa_target_subspaces + (total - config.asa_target_subspaces) * (mul_coeff**exponent))
        else:
            return config.asa_target_subspaces

    def _get_asa_total_subspaces(self) -> int:
        """Get total number of subspaces from model (sum across all layers)."""
        total = 0
        for module in self.model.modules():
            if isinstance(module, AdamssLayer) and module.num_subspaces:
                total += next(iter(module.num_subspaces.values()))
        return total

    def _global_mask_to_target(self, target_subspaces: int, adapter_name: str, asa_layers: list) -> None:
        """
        Apply **global** top-K masking across all layers.

        Collects importance scores from every subspace in every layer, ranks them
        globally, and keeps only the top ``target_subspaces`` active.
        """
        # 1. Collect (module, subspace_idx, score) for every subspace
        subspace_scores: list[tuple] = []
        for module in asa_layers:
            n_subspaces = module.num_subspaces.get(adapter_name, 0)
            exp_ipt = module.exp_avg_ipt[adapter_name]
            exp_unc = module.exp_avg_unc[adapter_name]

            for i in range(n_subspaces):
                key_A, key_B = f"A_{i}", f"B_{i}"
                if key_A not in exp_ipt or key_B not in exp_ipt:
                    continue
                score = (exp_ipt[key_A] * exp_unc[key_A]).mean() + (exp_ipt[key_B] * exp_unc[key_B]).mean()
                subspace_scores.append((module, i, score))

        if not subspace_scores:
            return

        # 2. Compute global threshold via kth-value
        all_scores = torch.stack([s[2] for s in subspace_scores])
        if target_subspaces >= len(subspace_scores):
            threshold = float("-inf")
        else:
            threshold = -torch.kthvalue(-all_scores, target_subspaces)[0].item()

        # 3. Apply masking
        for module, idx, score in subspace_scores:
            is_active = float(score) > threshold
            param_A = module.adamss_A[adapter_name]
            param_B = module.adamss_B[adapter_name]
            if idx < len(param_A):
                param_A[idx].requires_grad = is_active
                if not is_active:
                    param_A[idx].grad = None
            if idx < len(param_B):
                param_B[idx].requires_grad = is_active
                if not is_active:
                    param_B[idx].grad = None
