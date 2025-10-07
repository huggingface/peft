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

import torch
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
    TRANSFORMERS_MODELS_TO_WAVEFT_TARGET_MODULES_MAPPING,
)
from peft.utils.other import get_pattern_key

from .layer import WaveFTLayer, WaveFTLinear


class WaveFTModel(BaseTuner):
    prefix: str = "waveft_"
    tuner_layer_cls: type[BaseTunerLayer] = WaveFTLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_WAVEFT_TARGET_MODULES_MAPPING

    def _calculate_proportional_parameters(self, model: torch.nn.Module, waveft_config):
        """Calculate proportional parameter allocation for all target modules."""
        target_modules_info = []
        for name, module in model.named_modules():
            if check_target_module_exists(waveft_config, name):
                # Handle case where module is already wrapped with WaveFT
                if isinstance(module, WaveFTLayer):
                    # Use the base layer for dimension calculations
                    base_module = module.base_layer
                    if isinstance(base_module, torch.nn.Linear):
                        input_dim, output_dim = base_module.in_features, base_module.out_features
                    elif isinstance(base_module, Conv1D):
                        input_dim, output_dim = base_module.weight.shape[1], base_module.weight.shape[0]
                    else:
                        continue
                elif isinstance(module, torch.nn.Linear):
                    input_dim, output_dim = module.in_features, module.out_features
                elif isinstance(module, Conv1D):
                    input_dim, output_dim = module.weight.shape[1], module.weight.shape[0]
                else:
                    continue
                target_modules_info.append((name, input_dim, output_dim))

        if not target_modules_info:
            raise ValueError("No target modules found for proportional parameter allocation.")

        total_sum = sum(input_dim * output_dim for (_, input_dim, output_dim) in target_modules_info)
        num_layers = len(target_modules_info)
        total_budget = waveft_config.n_frequency * num_layers

        n_frequency_dict = {}
        for name, input_dim, output_dim in target_modules_info:
            layer_ratio = (input_dim * output_dim) / total_sum
            n_freq = round(layer_ratio * total_budget)
            n_frequency_dict[name] = n_freq

        return n_frequency_dict

    def _create_and_replace(
        self,
        waveft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Calculate proportional parameters if needed (only once per adapter)
        if waveft_config.proportional_parameters:
            if not hasattr(self, "_proportional_params_cache"):
                self._proportional_params_cache = {}
            if adapter_name not in self._proportional_params_cache:
                n_frequency_dict = self._calculate_proportional_parameters(self.model, waveft_config)
                self._proportional_params_cache[adapter_name] = n_frequency_dict

        # Determine n_frequency: Priority order:
        # 1. From proportional parameter cache (if proportional_parameters=True)
        # 2. From optional_kwargs (if passed directly)
        # 3. From n_frequency_pattern in config
        # 4. From default n_frequency in config
        n_frequency = None
        if (
            waveft_config.proportional_parameters
            and hasattr(self, "_proportional_params_cache")
            and adapter_name in self._proportional_params_cache
        ):
            n_frequency = self._proportional_params_cache[adapter_name].get(current_key)

        if n_frequency is None and "n_frequency" in optional_kwargs:
            n_frequency = optional_kwargs["n_frequency"]

        if n_frequency is None:
            pattern_keys = list(waveft_config.n_frequency_pattern.keys())
            target_name_key = get_pattern_key(pattern_keys, current_key)
            n_frequency = waveft_config.n_frequency_pattern.get(target_name_key, waveft_config.n_frequency)

        # Determine wavelet_family
        wavelet_family = None
        if "wavelet_family" in optional_kwargs:
            wavelet_family = optional_kwargs["wavelet_family"]
        if wavelet_family is None:
            wavelet_family = waveft_config.wavelet_family

        scaling = waveft_config.scaling
        random_loc_seed = waveft_config.random_loc_seed
        bias = hasattr(target, "bias") and target.bias is not None
        # Prepare kwargs for module creation/update
        kwargs = {
            "n_frequency": n_frequency,
            "scaling": scaling,
            "fan_in_fan_out": waveft_config.fan_in_fan_out,
            "init_weights": waveft_config.init_weights,
            "random_loc_seed": waveft_config.random_loc_seed,
            "wavelet_family": wavelet_family,  # Use determined wavelet family
        }
        kwargs["bias"] = bias

        if isinstance(target, WaveFTLayer):
            target.update_layer(
                adapter_name,
                n_frequency,
                scaling,
                waveft_config.init_weights,
                random_loc_seed,
                wavelet_family=wavelet_family,  # Pass determined wavelet family
                use_idwt=waveft_config.use_idwt,
            )
        else:
            new_module = self._create_new_module(waveft_config, adapter_name, target, **kwargs)
            if adapter_name != self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(waveft_config, adapter_name, target, **kwargs):
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
                kwargs["fan_in_fan_out"] = waveft_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = waveft_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`."
            )

        kwargs["wavelet_family"] = waveft_config.wavelet_family
        kwargs["use_idwt"] = waveft_config.use_idwt
        new_module = WaveFTLinear(target, adapter_name, **kwargs)

        return new_module

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        super().delete_adapter(adapter_name)
        # Clean up proportional parameters cache
        if hasattr(self, "_proportional_params_cache") and adapter_name in self._proportional_params_cache:
            del self._proportional_params_cache[adapter_name]
