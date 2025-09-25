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
from dataclasses import asdict
from enum import Enum
from typing import Optional

import torch
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
    TRANSFORMERS_MODELS_TO_WAVEFT_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _get_submodules,
)
from peft.utils.other import get_pattern_key

from .config import WaveFTConfig
from .layer import WaveFTLayer, WaveFTLinear


class WaveFTModel(BaseTuner):
    prefix: str = "waveft_"

    def _check_new_adapter_config(self, config: WaveFTConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not fully correspond to the error message.
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_module_exists(waveft_config, key):
        return check_target_module_exists(waveft_config, key)

    def _calculate_proportional_parameters(self, model: torch.nn.Module, waveft_config):
        """Calculate proportional parameter allocation for all target modules."""
        target_modules_info = []
        for name, module in model.named_modules():
            if self._check_target_module_exists(waveft_config, name):
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
            if not hasattr(self, '_proportional_params_cache'):
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
        if (waveft_config.proportional_parameters and
            hasattr(self, '_proportional_params_cache') and
            adapter_name in self._proportional_params_cache):
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
            "wavelet_family": wavelet_family, # Use determined wavelet family
        }
        kwargs["bias"] = bias

        if isinstance(target, WaveFTLayer):
            target.update_layer(
                adapter_name,
                n_frequency,
                scaling,
                waveft_config.init_weights,
                random_loc_seed,
                wavelet_family=wavelet_family, # Pass determined wavelet family
                use_idwt=waveft_config.use_idwt
            )
        else:
            new_module = self._create_new_module(waveft_config, adapter_name, target, **kwargs)
            if adapter_name != self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        meta = torch.device("meta")
        # dispatch to correct device
        for name, module in new_module.named_modules():
            if self.prefix in name:
                if not any(p.device == meta for p in module.parameters()):
                    module.to(child.weight.device)

    def _mark_only_adapters_as_trainable(self, model: torch.nn.Module) -> None:
        # Freeze non-adapter parameters based on prefix
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        # Handle bias trainability separately
        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue
            elif bias == "all":
                # Unfreeze all bias parameters in the model
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "wave_only":
                # Unfreeze only biases within WaveFTLayers that have the active adapter
                for m in model.modules():
                    if isinstance(m, WaveFTLayer) and hasattr(m, "bias") and m.bias is not None:
                        # Ensure the layer has the active adapter configured before unfreezing bias
                        if active_adapter in m.waveft_n_frequency: # Check if adapter exists for this layer
                            m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

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
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
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

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":
                raise
            return getattr(self.model, name)



    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
            config_dict[key] = config
        return config_dict

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name: str | list[str], inference_mode: bool = False) -> None:
        """Set the active adapter(s).

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
            inference_mode (`bool`): Whether to set the adapter in inference mode.
        """
        for module in self.model.modules():
            if isinstance(module, WaveFTLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name, inference_mode=inference_mode)
        self.active_adapter = adapter_name

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_WAVEFT_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_WAVEFT_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue

            if hasattr(target, "base_layer"):
                if merge:
                    target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                self._replace_module(parent, target_name, target.get_base_layer(), target)
            elif isinstance(target, ModulesToSaveWrapper):
                # save any additional trainable modules part of `modules_to_save`
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def delete_adapter(self, adapter_name: str):
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        # Clean up proportional parameters cache
        if hasattr(self, '_proportional_params_cache') and adapter_name in self._proportional_params_cache:
            del self._proportional_params_cache[adapter_name]

        # we cannot use self.prefix as we want to include non-trainable waveft parameters
        key_list = [key for key, _ in self.model.named_modules() if "waveft" not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, WaveFTLayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapter[:]

        self.active_adapter = new_adapter or []

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        r"""
        This method merges the Wave layers into the base model. This is needed if someone wants to use the base
        model as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self) -> torch.nn.Module:
        """
        Gets back the base model by removing all the Wave modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)

