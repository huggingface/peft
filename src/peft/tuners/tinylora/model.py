# Copyright 2023-present the HuggingFace Inc. team.
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

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_TINYLORA_TARGET_MODULES_MAPPING,
)

from ..tuners_utils import _maybe_include_all_linear_layers
from .config import TinyLoraConfig
from .layer import Embedding, Linear, TinyLoraLayer


class TinyLoraModel(BaseTuner):
    """
    Creates TinyLoRA model from a pretrained transformers model.

    TinyLoRA is an extremely parameter-efficient fine-tuning method that uses SVD decomposition of frozen weights and
    projects a tiny trainable vector through fixed random tensors. Based on the paper "Learning to Reason in 13
    Parameters" (arXiv:2602.04118).

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`TinyLoraConfig`]): The configuration of the TinyLoRA model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, *optional*, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The TinyLoRA model.

    Example:
        ```python
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import TinyLoraConfig, get_peft_model

        >>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        >>> config = TinyLoraConfig(r=2, u=64, target_modules=["q_proj", "v_proj"])
        >>> model = get_peft_model(base_model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`TinyLoraConfig`]): The configuration of the TinyLoRA model.
    """

    prefix: str = "tinylora_"
    tuner_layer_cls = TinyLoraLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_TINYLORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage=False, **kwargs):
        super().__init__(model, config, adapter_name, low_cpu_mem_usage, **kwargs)

    def _init_tinylora_v(self, config: TinyLoraConfig, adapter_name: str) -> None:
        """Re-initialize the tinylora_v vectors with uniform random values."""
        for key, v in self.tinylora_v.items():
            if key.startswith(f"{adapter_name}_group_"):
                nn.init.uniform_(v, -config.init_v_bound, config.init_v_bound)

    def _pre_injection_hook(self, model: nn.Module, config: TinyLoraConfig, adapter_name: str) -> None:
        """Initialize shared trainable vectors based on ntie before layer injection."""
        # Initialize the shared parameter dict for v vectors
        self.tinylora_v = nn.ParameterDict({})

        # Count target layers to determine number of groups
        self._target_layer_count = self._count_target_layers(config)

        # Track layer index during injection
        self._current_layer_idx = 0

    def _count_target_layers(self, config: TinyLoraConfig) -> int:
        """Count the number of layers that will be targeted."""
        model_config = self.get_model_config(self.model)
        peft_config = self._prepare_adapter_config(config, model_config)
        peft_config = _maybe_include_all_linear_layers(peft_config, self.model)

        count = 0
        for key, module in self.model.named_modules():
            if not self._check_target_module_exists(peft_config, key):
                continue
            if isinstance(module, (nn.Linear, Conv1D, nn.Embedding)):
                count += 1

        return count

    def _check_new_adapter_config(self, config: TinyLoraConfig) -> None:
        """Check the config when a new adapter is being added."""
        super()._check_new_adapter_config(config)

        for existing_config in self.peft_config.values():
            if existing_config is config:
                continue

            if existing_config.projection_seed != config.projection_seed:
                raise ValueError(
                    f"TinyLoRA projection seed must be the same for all adapters. Got {config.projection_seed=} but "
                    f"previous config had {existing_config.projection_seed}."
                )

        save_projection_unique_values = sorted({c.save_projection for c in self.peft_config.values()})
        if len(save_projection_unique_values) > 1:
            raise ValueError(
                "TinyLoRA projection tensors must be saved for all adapters or none, but got multiple different values: "
                f"{save_projection_unique_values}"
            )

    def _create_and_replace(
        self,
        tinylora_config: TinyLoraConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Get layer index for this module
        layer_idx = self._current_layer_idx
        self._current_layer_idx += 1

        # Determine the group for this layer based on ntie
        num_groups = max(1, self._target_layer_count // tinylora_config.ntie)
        group_idx = min(layer_idx // tinylora_config.ntie, num_groups - 1)
        # Use format "{adapter_name}_group_{idx}" - handled specially in save_and_load.py
        v_key = f"{adapter_name}_group_{group_idx}"

        # Initialize v for this group if not already done
        if v_key not in self.tinylora_v:
            # Get dtype from target layer's weight
            if hasattr(target, "weight"):
                dtype = target.weight.dtype
            else:
                dtype = None  # Will default to float32
            v = nn.Parameter(torch.empty(tinylora_config.u, dtype=dtype))
            if tinylora_config.init_weights:
                nn.init.uniform_(v, -tinylora_config.init_v_bound, tinylora_config.init_v_bound)
            else:
                # Initialize to zeros for identity operation
                nn.init.zeros_(v)
            self.tinylora_v[v_key] = v

        kwargs = {
            "r": tinylora_config.r,
            "u": tinylora_config.u,
            "tinylora_dropout": tinylora_config.tinylora_dropout,
            "fan_in_fan_out": tinylora_config.fan_in_fan_out,
            "init_weights": tinylora_config.init_weights,
            "projection_seed": tinylora_config.projection_seed,
        }

        if isinstance(target, TinyLoraLayer):
            target.set_layer_idx(layer_idx)
            target.update_layer(
                adapter_name,
                self.tinylora_v,
                v_key,
                tinylora_config.r,
                tinylora_config.u,
                tinylora_config.tinylora_dropout,
                tinylora_config.init_weights,
                tinylora_config.projection_seed,
                fan_in_fan_out=tinylora_config.fan_in_fan_out,
            )
        else:
            new_module = self._create_new_module(
                tinylora_config, self.tinylora_v, v_key, adapter_name, target, **kwargs
            )
            new_module.set_layer_idx(layer_idx)

            if adapter_name not in self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

        # Ensure the shared v parameters remain trainable for the active adapter,
        # but only if the adapter is not in inference mode
        if adapter_name in self.active_adapter:
            inference_mode = getattr(tinylora_config, "inference_mode", False)
            if not inference_mode:
                for key, param in self.tinylora_v.items():
                    if adapter_name in key:
                        param.requires_grad = True

    @staticmethod
    def _create_new_module(
        tinylora_config: TinyLoraConfig,
        tinylora_v: nn.ParameterDict,
        v_key: str,
        adapter_name: str,
        target: nn.Module,
        **kwargs,
    ):
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
                kwargs["fan_in_fan_out"] = tinylora_config.fan_in_fan_out = False
            new_module = Linear(
                target,
                tinylora_v,
                v_key,
                adapter_name,
                **kwargs,
            )
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = tinylora_config.fan_in_fan_out = True
            new_module = Linear(
                target,
                tinylora_v,
                v_key,
                adapter_name,
                **kwargs,
            )
        elif isinstance(target_base_layer, torch.nn.Embedding):
            # Remove Linear-specific kwargs not applicable to Embedding
            kwargs.pop("fan_in_fan_out", None)
            new_module = Embedding(
                target,
                tinylora_v,
                v_key,
                adapter_name,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `transformers.pytorch_utils.Conv1D`."
            )

        return new_module

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        """
        Returns the number of trainable TinyLoRA parameters and total parameters.
        """
        trainable_params = 0
        all_param = 0

        for name, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def print_trainable_parameters(self) -> None:
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || "
            f"trainable%: {100 * trainable_params / all_param:.4f}"
        )

    def _cast_adapter_dtype(self, adapter_name: str, autocast_adapter_dtype: bool = True) -> None:
        """
        Cast the adapter weights to the correct dtype.

        Override to also handle the model-level tinylora_v parameters which use keys like '{adapter_name}_group_{idx}'
        instead of just the adapter name.
        """
        # Call parent implementation for layer-level parameters
        super()._cast_adapter_dtype(adapter_name, autocast_adapter_dtype)

        if not autocast_adapter_dtype:
            return

        # Handle model-level tinylora_v parameters
        dtypes_to_convert_to_fp32 = {torch.float16, torch.bfloat16}
        for key, param in self.tinylora_v.items():
            # Check if this key belongs to the adapter (key format: "{adapter_name}_group_{idx}")
            if key.startswith(f"{adapter_name}_group_"):
                if param.dtype in dtypes_to_convert_to_fp32:
                    param.data = param.data.to(torch.float32)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        """
        Mark only the adapter layers as trainable.

        Override the base class method to ensure that the shared tinylora_v parameters remain trainable, since they are
        stored at the model level. Only do this for adapters that are not in inference mode.
        """
        # First, call the parent implementation
        super()._mark_only_adapters_as_trainable(model)

        # Then, explicitly ensure the shared tinylora_v parameters are trainable
        # for active adapters that are not in inference mode
        for active_adapter in self.active_adapters:
            # Check if this adapter is in inference mode
            if active_adapter in self.peft_config:
                inference_mode = getattr(self.peft_config[active_adapter], "inference_mode", False)
                if inference_mode:
                    continue
            for key, param in self.tinylora_v.items():
                if active_adapter in key:
                    param.requires_grad = True
