# Copyright 2025-present the HuggingFace Inc. team.
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

import warnings

import numpy as np
import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_UNILORA_TARGET_MODULES_MAPPING

from .config import UniLoraConfig
from .layer import Linear, UniLoraLayer


class UniLoraModel(BaseTuner):
    """Creates a UniLora adapter around a pretrained model."""

    prefix: str = "unilora_"
    tuner_layer_cls = UniLoraLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_UNILORA_TARGET_MODULES_MAPPING

    def __init__(
        self,
        model,
        config,
        adapter_name,
        low_cpu_mem_usage: bool = False,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__(
            model,
            config,
            adapter_name,
            low_cpu_mem_usage=low_cpu_mem_usage,
            state_dict=state_dict,
        )

    def _ensure_unilora_theta_d(self) -> None:
        if not hasattr(self, "unilora_theta_d"):
            self.unilora_theta_d = nn.ParameterDict({})

    def inject_adapter(
        self,
        model: nn.Module,
        adapter_name: str,
        autocast_adapter_dtype: bool = True,
        low_cpu_mem_usage: bool = False,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self._ensure_unilora_theta_d()
        self._init_unilora_theta_d(self.peft_config[adapter_name], adapter_name)
        super().inject_adapter(
            model,
            adapter_name,
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            state_dict=state_dict,
        )
        self._assign_unilora_indices_and_scales(adapter_name)

    def _iter_unilora_layers(self):
        for module in self.model.modules():
            if isinstance(module, UniLoraLayer):
                yield module

    def _assign_unilora_indices_and_scales(self, adapter_name: str) -> None:
        layers = [
            module
            for module in self._iter_unilora_layers()
            if adapter_name in module.unilora_indices_A.keys() and adapter_name in module.unilora_indices_B.keys()
        ]
        lora_param_count = sum(
            module.unilora_indices_A[adapter_name].numel() + module.unilora_indices_B[adapter_name].numel()
            for module in layers
        )
        if lora_param_count == 0:
            return

        config = self.peft_config[adapter_name]
        indices = self.generate_index(lora_param_count, config.theta_d_length, config.proj_seed)
        chunk_idx = 0

        for module in layers:
            param_numel = module.unilora_indices_A[adapter_name].numel()
            chunk = indices[chunk_idx : chunk_idx + param_numel]
            module.unilora_indices_A[adapter_name] = chunk.view_as(module.unilora_indices_A[adapter_name]).clone()
            chunk_idx += param_numel

            param_numel = module.unilora_indices_B[adapter_name].numel()
            chunk = indices[chunk_idx : chunk_idx + param_numel]
            module.unilora_indices_B[adapter_name] = chunk.view_as(module.unilora_indices_B[adapter_name]).clone()
            chunk_idx += param_numel

        if chunk_idx != indices.numel():
            raise ValueError(
                "UniLora index assignment consumed a different number of indices than were generated. "
                f"Expected {indices.numel()} values but consumed {chunk_idx}. Please open an issue with your "
                "model architecture and UniLora configuration."
            )

        counts = torch.bincount(indices, minlength=config.theta_d_length)
        inv_sqrt_counts = torch.zeros(config.theta_d_length, dtype=torch.float32)
        used_indices = counts > 0
        inv_sqrt_counts[used_indices] = counts[used_indices].float().rsqrt()

        for module in layers:
            scales_A = inv_sqrt_counts[module.unilora_indices_A[adapter_name].long()]
            scales_B = inv_sqrt_counts[module.unilora_indices_B[adapter_name].long()]
            module.update_scaling(adapter_name, scales_A, scales_B)

    @staticmethod
    def generate_index(lora_param_count: int, theta_d_length: int, proj_seed: int) -> torch.Tensor:
        """Assign deterministic `theta_d` indices to the flattened UniLora parameter space.

        A plain `np.random.choice(np.arange(theta_d_length), size=lora_param_count)` samples each position
        independently, which can leave some `theta_d` entries unused for smaller adapters. UniLora instead uses a
        balanced deterministic assignment: each index appears either `floor(D / d)` or `ceil(D / d)` times, where `D`
        is the flattened LoRA parameter count and `d` is `theta_d_length`. This keeps per-index normalization stable
        while still shuffling the assignment with `proj_seed`.
        """
        if lora_param_count < 0:
            raise ValueError(f"`lora_param_count` must be non-negative, got {lora_param_count}.")
        if theta_d_length <= 0:
            raise ValueError(f"`theta_d_length` must be positive, got {theta_d_length}.")

        base_count = lora_param_count // theta_d_length
        remaining = lora_param_count % theta_d_length
        rng = np.random.default_rng(proj_seed)
        data = np.repeat(np.arange(theta_d_length), base_count)
        if remaining > 0:
            extras = rng.choice(theta_d_length, size=remaining, replace=False)
            data = np.concatenate([data, extras])
        rng.shuffle(data)
        return torch.as_tensor(data, dtype=torch.long)

    def _init_unilora_theta_d(self, config: UniLoraConfig, adapter_name: str) -> None:
        if adapter_name in self.unilora_theta_d:
            return

        if config.init_weights:
            unilora_theta_d = torch.empty(config.theta_d_length)
            torch.nn.init.uniform_(unilora_theta_d, -config.init_theta_d_bound, config.init_theta_d_bound)
        else:
            unilora_theta_d = torch.randn(config.theta_d_length)
        self.unilora_theta_d[adapter_name] = nn.Parameter(unilora_theta_d)

    def _pre_injection_hook(self, model: nn.Module, config: UniLoraConfig, adapter_name: str) -> None:
        self._ensure_unilora_theta_d()
        self._init_unilora_theta_d(config, adapter_name)

    def _create_and_replace(
        self,
        unilora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current key should not be `None` when creating a UniLora layer.")

        self._init_unilora_theta_d(unilora_config, adapter_name)

        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "fan_in_fan_out": unilora_config.fan_in_fan_out,
            "bias": bias,
        }

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                unilora_theta_d=self.unilora_theta_d,
                r=unilora_config.r,
                theta_d_length=unilora_config.theta_d_length,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_theta_d=self.unilora_theta_d,
                adapter_name=adapter_name,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(unilora_config, unilora_theta_d, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = unilora_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = unilora_config.fan_in_fan_out = True
        else:
            raise TypeError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `transformers.pytorch_utils.Conv1D`."
            )

        return Linear(
            base_layer=target,
            unilora_theta_d=unilora_theta_d,
            adapter_name=adapter_name,
            r=unilora_config.r,
            theta_d_length=unilora_config.theta_d_length,
            unilora_dropout=unilora_config.unilora_dropout,
            **kwargs,
        )
