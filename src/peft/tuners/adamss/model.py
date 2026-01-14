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

import re
import warnings
from dataclasses import asdict
from enum import Enum
from typing import Optional

import torch
from torch import nn
from tqdm import tqdm

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _get_submodules,
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

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False, state_dict: dict = None) -> None:
        # Initialize ASA tracking before BaseTuner injects adapters so attribute exists.
        self._asa_total_kk = {}
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage, state_dict=state_dict)
        # Track the trainable adapter name (following AdaLora pattern)
        # Handle both dict config (from PeftModel) and single config (from inject_adapter_in_model)
        adapter_config = config[adapter_name] if isinstance(config, dict) else config
        if not adapter_config.inference_mode:
            self.trainable_adapter_name = adapter_name
        # Cache total subspace counts per adapter to keep ASA schedule deterministic

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
                adamss_config, adapter_name, target,
                inference_mode=adamss_config.inference_mode,
                **optional_kwargs
            )
            self._record_total_kk(adapter_name, adamss_config, new_module)
            # requires_grad is handled inside _create_new_module via set_adapter
            self._replace_module(parent, target_name, new_module, target)

    def _record_total_kk(self, adapter_name: str, adamss_config: AdamssConfig, module: nn.Module) -> None:
        """Track total subspaces per adapter so the ASA schedule matches adamss_pkg."""
        if not hasattr(module, "KK"):
            return
        total_kk = module.KK.get(adapter_name, 0)
        if total_kk is None or total_kk <= 0:
            return
        prev_total = self._asa_total_kk.get(adapter_name, 0)
        self._asa_total_kk[adapter_name] = prev_total + total_kk
        adamss_config.total_kk = self._asa_total_kk[adapter_name]

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
            raise ValueError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear` is supported."
            )

        return new_module

    def _replace_module(self, parent, child_name, new_module, child):
        """Replace a module with a new module."""
        setattr(parent, child_name, new_module)
        # Ensure base layer weight dtype matches
        # Skip moving if parameters are on meta device (will be loaded later)
        if hasattr(child, "weight"):
            # Check if any parameters are on meta device
            meta_params = any(p.device.type == "meta" for p in new_module.parameters())
            if not meta_params:
                new_module.to(child.weight.device)

        # Copy state for modules to save
        if hasattr(new_module, "base_layer"):
            new_module.base_layer.load_state_dict(child.state_dict(), strict=False)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        """
        Mark only Adamss parameters as trainable.
        """
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias_params = [
                "adamss_A",
                "adamss_B",
            ]
            for n, p in model.named_parameters():
                if any(f"{active_adapter}.{bp}" in n for bp in bias_params):
                    p.requires_grad = True

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        """Prepare adapter config."""
        if peft_config.target_modules is None:
            # Default target modules for different model types
            if model_config.get("model_type") == "vit":
                peft_config.target_modules = ["query", "value"]
            else:
                # Try to infer from common patterns
                peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING.get(
                    model_config.get("model_type"), ["q_proj", "v_proj"]
                )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        """
        Unload and optionally merge Adamss adapters.
        
        This properly replaces adapter layers with their base layers after merging.
        """
        # Call parent implementation which handles the layer replacement
        return super()._unload_and_optionally_merge(
            merge=merge,
            progressbar=progressbar,
            safe_merge=safe_merge,
            adapter_names=adapter_names,
        )

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ):
        """
        Merge Adamss weights into base model and unload adapters.
        """
        return self._unload_and_optionally_merge(
            merge=True, progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self):
        """
        Unload Adamss adapters and return base model.
        """
        return self._unload_and_optionally_merge(merge=False)
    
    def update_and_allocate(self, global_step: int) -> None:
        """
        Update importance scores and apply ASA masking (if enabled).
        
        This method should be called in every training step after `loss.backward()` 
        and before `optimizer.zero_grad()` when ASA is enabled.
        
        Args:
            global_step (`int`): The current training step.
            
        Example:
            ```python
            >>> loss = model(**batch).loss
            >>> loss.backward()
            >>> optimizer.step()
            >>> model.base_model.update_and_allocate(global_step)  # Update ASA
            >>> optimizer.zero_grad()
            ```
        """
        # Get the trainable adapter name
        if not hasattr(self, 'trainable_adapter_name'):
            # Fallback: use first adapter in peft_config
            adapter_names = list(self.peft_config.keys())
            if not adapter_names:
                return
            self.trainable_adapter_name = adapter_names[0]
        
        adamss_config = self.peft_config[self.trainable_adapter_name]
        
        # Only proceed if ASA is enabled
        if not adamss_config.use_asa:
            return
        
        from .layer import AdamssLayer
        
        within_warmup = adamss_config.init_warmup <= global_step <= adamss_config.final_warmup
        should_mask = within_warmup and (global_step % adamss_config.mask_interval == 0)

        if not should_mask:
            return

        asa_layers = []
        for module in self.model.modules():
            if isinstance(module, AdamssLayer) and module.exp_avg_ipt:
                asa_layers.append(module)

        if not asa_layers:
            return

        for module in asa_layers:
            for adapter_name in module.exp_avg_ipt.keys():
                module.reset_importance(adapter_name)
                module.update_importance(adapter_name, adamss_config.beta1, adamss_config.beta2)

        curr_kk = self._schedule_threshold(global_step, adamss_config)
        if curr_kk is not None:
            self._mask_to_target(curr_kk)
    
    def _schedule_threshold(self, step: int, config) -> Optional[int]:
        """Calculate current target KK based on warmup schedule (aligned with adamss_pkg)."""
        total_kk = getattr(config, "total_kk", None)
        if not total_kk:
            total_kk = self._get_total_kk()

        if not total_kk:
            return None

        if step < config.init_warmup:
            # Initial warmup: use all subspaces; no masking
            return None
        elif step <= config.final_warmup:
            # Gradual decrease following adamss_pkg schedule
            mul_coeff = 1.0 - (step - config.init_warmup) / (config.final_warmup - config.init_warmup)
            # Clamp for numerical stability
            mul_coeff = max(0.0, min(1.0, mul_coeff))
            curr_kk = int(config.target_kk + (total_kk - config.target_kk) * (mul_coeff ** getattr(config, 'tt', 3.0)))
            return curr_kk
        else:
            # After final warmup: fix target_kk
            return config.target_kk
    
    def _get_total_kk(self) -> int:
        """Get total number of subspaces from model."""
        from .layer import AdamssLayer
        for module in self.model.modules():
            if isinstance(module, AdamssLayer) and module.KK:
                return list(module.KK.values())[0]
        return 0
    
    def _mask_to_target(self, target_kk: int) -> None:
        """Apply masking to all Adamss layers."""
        from .layer import AdamssLayer
        for module in self.model.modules():
            if isinstance(module, AdamssLayer):
                for adapter_name in module.exp_avg_ipt.keys():
                    module.mask_to_target(adapter_name, target_kk)

