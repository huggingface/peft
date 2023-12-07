# coding=utf-8
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

import logging
import re
import warnings
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import torch
from torch import nn

from peft.utils import COMMON_LAYERS_PATTERN

from ..config import PeftConfig
from ..utils import ModulesToSaveWrapper, _get_submodules


logger = logging.getLogger(__name__)


class BaseTuner(nn.Module, ABC):
    r"""
    A base tuner model that provides the common methods and attributes for all tuners that are injectable into a
    torch.nn.Module

    For adding a new Tuner class, one needs to overwrite the following methods:

    - **_prepare_adapter_config**:
        A private method to eventually prepare the adapter config, for example in case the field `target_modules` is
        missing.
    - **_check_target_module_exists**:
        A helper private method to check if the passed module's key name matches any of the target modules in the
        adatper_config.
    - **_create_and_replace**:
        A private method to create and replace the target module with the adapter module.
    - **_check_target_module_exists**:
        A private helper method to check if the passed module's key name matches any of the target modules in the
        adatper_config.

    The easiest is to check what is done in the `peft.tuners.lora.LoraModel` class.

    Attributes:
        model (`torch.nn.Module`):
            The model to which the adapter tuner layers will be attached.
        forward (`Callable`):
            The forward method of the model.
        peft_config (`Union[`PeftConfig`, dict[str, PeftConfig]]`):
            The adapter configuration object, it should be a dictionary of `str` to `PeftConfig` objects. One can also
            pass a PeftConfig object and a new adapter will be created with the default name `adapter` or create a new
            dictionary with a key `adapter_name` and a value of that peft config.
        config (`dict[str, Any]`):
            The model configuration object, it should be a dictionary of `str` to `Any` objects.
    """

    def __init__(self, model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str) -> None:
        super().__init__()

        self.model = model

        # For advanced developpers, if you want to attach multiple adapters to your
        # model, just add a `peft_config` dict attribute to your model.
        if not hasattr(self, "peft_config"):
            self.peft_config = {adapter_name: peft_config} if isinstance(peft_config, PeftConfig) else peft_config
        else:
            logger.info(
                "Already found a `peft_config` attribute in the model. This will lead to having multiple adapters"
                " in the model. Make sure to know what you are doing!"
            )
            if isinstance(peft_config, PeftConfig):
                self.peft_config[adapter_name] = peft_config
            else:
                # user is adding a dict of PeftConfigs
                self.peft_config.update(peft_config)

        self.active_adapter = adapter_name
        self.inject_adapter(self.model, adapter_name)

        # Copy the peft_config in the injected model.
        self.model.peft_config = self.peft_config

    @property
    def active_adapters(self) -> list[str]:
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        # is already a list of str
        return self.active_adapter

    def forward(self, *args: Any, **kwargs: Any):
        return self.model.forward(*args, **kwargs)

    @abstractmethod
    def _prepare_adapter_config(self, peft_config: PeftConfig, model_config: dict) -> PeftConfig:
        r"""
        A private method to eventually prepare the adapter config. For transformers based models, if
        `peft_config.target_modules` is None, we can automatically infer the target modules from the
        `TRANSFORMERS_MODELS_TO_XXX_TARGET_MODULES_MAPPING`. This method can be further refactored in the future to
        automatically infer it for all tuner models.

        Check out `peft.tuner.lora.LoraModel._prepare_adapter_config` for an example.

        Args:
            peft_config (`str`):
                The adapter config.
            model_config (`str`):
                The transformers model config, that config should contain the `model_type` key.
        """
        ...

    @abstractmethod
    def _check_target_module_exists(peft_config: PeftConfig, key: str) -> bool:
        r"""
        A helper private method to check if the passed module's key name matches any of the target modules in the
        `peft_config.target_modules` list. If it does, return `True`, else return `False`.

        Args:
            peft_config (`PeftConfig`):
                The adapter config.
            key (`str`):
                The module's key name.
        """
        ...

    @abstractmethod
    def _create_and_replace(
        self,
        peft_config: PeftConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        **optional_kwargs: Any,
    ) -> None:
        r"""
        Inplace replacement of the target module with the adapter layer. This method needs to be overriden by all the
        tuner classes.

        Check `peft.tuners.lora.LoraModel._create_and_replace` for an example.

        Args:
            peft_config (`PeftConfig`):
                The adapter config.
            adapter_name (`str`):
                The adapter name.
            target (`nn.Module`):
                The target module.
            target_name (`str`):
                The target module's name.
            parent (`nn.Module`):
                The parent module.
            **optional_kwargs (`dict`):
                The optional keyword arguments to pass to deal with particular cases (e.g. 8bit, 4bit quantization)
        """
        ...

    @abstractmethod
    def _mark_only_adapters_as_trainable(self):
        r"""
        A helper method to mark only the adapter layers as trainable (i.e. module.requires_grad = False) This needs to
        be overriden for all tuner classes to match the correct key names.

        Check `peft.tuners.lora.LoraModel._mark_only_adapters_as_trainable` for an example.
        """
        ...

    def _check_new_adapter_config(self, config: PeftConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        pass

    def inject_adapter(self, model: nn.Module, adapter_name: str):
        r"""
        Creates adapter layers and replaces the target modules with the adapter layers. This method is called under the
        hood by `peft.mapping.get_peft_model` if a non-prompt tuning adapter class is passed.

        The corresponding PEFT config is directly retrieved from the `peft_config` attribute of the BaseTuner class.

        Args:
            model (`nn.Module`):
                The model to be tuned.
            adapter_name (`str`):
                The adapter name.
        """
        peft_config = self.peft_config[adapter_name]
        # Note: If possible, all checks should be performed *at the start of this method*.
        # This way, we can raise early if something goes wrong, without leaving the model
        # in a bad (half-initialized) state.
        self._check_new_adapter_config(peft_config)

        is_target_modules_in_base_model = False
        key_list = [key for key, _ in model.named_modules()]

        _check_for_modules_to_save = getattr(peft_config, "modules_to_save", None) is not None
        _has_modules_to_save = False

        model_config = getattr(model, "config", {"model_type": "custom"})
        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()

        peft_config = self._prepare_adapter_config(peft_config, model_config)

        for key in key_list:
            # Check for modules_to_save in case
            if _check_for_modules_to_save and any(
                key.endswith(f"{module_to_save}") for module_to_save in peft_config.modules_to_save
            ):
                # Optionally set the modules to save
                parent, target, target_name = _get_submodules(model, key)

                if not isinstance(target, ModulesToSaveWrapper):
                    new_module = ModulesToSaveWrapper(target, adapter_name)
                    setattr(parent, target_name, new_module)
                else:
                    target.update(adapter_name)

                _has_modules_to_save = True
                continue

            if not self._check_target_module_exists(peft_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(model, key)

            optional_kwargs = {
                "loaded_in_8bit": getattr(model, "is_loaded_in_8bit", False),
                "loaded_in_4bit": getattr(model, "is_loaded_in_4bit", False),
                "current_key": key,
            }
            self._create_and_replace(peft_config, adapter_name, target, target_name, parent, **optional_kwargs)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

        self._mark_only_adapters_as_trainable()

        if self.peft_config[adapter_name].inference_mode:
            for n, p in self.model.named_parameters():
                if adapter_name in n:
                    p.requires_grad = False

        if _has_modules_to_save:
            if not hasattr(model, "modules_to_save"):
                model.modules_to_save = set(peft_config.modules_to_save)
            else:
                model.modules_to_save.update(set(peft_config.modules_to_save))

    def merge_adapter(self, adapter_names: Optional[list[str]] = None) -> None:
        """
        This method merges the adapter layers into the base model.

        Merging adapters can lead to a speed up of the forward pass. A copy of the adapter weights is still kept in
        memory, which is required to unmerge the adapters. In order to merge the adapter weights without keeping them
        in memory, please call `merge_and_unload`.

        Args:
            safe_merge (`bool`, *optional*):
                If `True`, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If `None`, all active adapters will be merged.
                Defaults to `None`.
        """
        for module in self.model.modules():
            if isinstance(module, BaseTunerLayer):
                module.merge(adapter_names=adapter_names)

    def unmerge_adapter(self):
        """
        This method unmerges all merged adapter layers from the base model.
        """
        for module in self.model.modules():
            if isinstance(module, BaseTunerLayer):
                module.unmerge()

    def _unloading_checks(self, adapter_names: Optional[List[str]]):
        adapters_to_consider = adapter_names or self.active_adapters
        is_modules_to_save_available = any(
            self.peft_config[adapter].modules_to_save for adapter in adapters_to_consider
        )
        if is_modules_to_save_available and len(adapters_to_consider) > 1:
            raise ValueError("Cannot unload multiple adapters that specify `modules_to_save`.")


class BaseTunerLayer(ABC):
    r"""
    A tuner layer mixin that provides the common methods and attributes for all tuners.

    Args:
        is_plugable (`bool`, *optional*):
            Whether the adapter layer can be plugged to any pytorch module
        active_adapters (Union[List[`str`], `str`], *optional*):
            The name of the active adapter.
    """

    active_adapter = None

    # All names of layers that may contain adapter (trainable) weights
    adapter_layer_names: tuple[str] = ()
    # All names of other parameters that may contain adapter-related parameters
    other_param_names: tuple[str] = ()

    # indicates whether all adapters should be disabled
    _disable_adapters: bool = False

    # the currently active adapter(s)
    _active_adapter: str | list[str] = "default"

    # List all merged adapters
    merged_adapters: list[str] = []

    def get_base_layer(self) -> nn.Module:
        """
        (Recursively) get the base_layer.

        This is necessary for the case that the tuner layer wraps another tuner layer.

        """
        base_layer = self
        while hasattr(base_layer, "base_layer"):
            base_layer = base_layer.base_layer
        return base_layer

    @property
    def weight(self) -> torch.Tensor:
        # This is required for some transformers code, e.g. for T5, weight is accessed as:
        #     self.wo.weight
        # where "wo" is the adapter layer.
        # https://github.com/huggingface/transformers/blob/78f6ed6c70b29c1560780e3869a7ad4c6b3d2710/src/transformers
        # /models/t5/modeling_t5.py#L292
        base_layer = self.get_base_layer()
        if hasattr(base_layer, "qweight"):
            # QuantLinear
            weight = base_layer.qweight
        else:
            # Other layers
            weight = base_layer.weight
        return weight

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError

    def unmerge(self) -> None:
        raise NotImplementedError

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    @property
    def disable_adapters(self) -> bool:
        # use a property to ensure that disable_adapters is not set directly, instead use the enable_adapters method
        return self._disable_adapters

    @property
    def active_adapter(self) -> str:
        # use a property to ensure that active_adapter is not set directly, instead use the set_adapter method
        return self._active_adapter

    @property
    def active_adapters(self):
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        # is already a list of str
        return self.active_adapter

    def enable_adapters(self, enabled: bool) -> None:
        """Toggle the enabling and disabling of adapters

        Takes care of setting the requires_grad flag for the adapter weights.

        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
        if enabled:
            self.set_adapter(self.active_adapters)
            self._disable_adapters = False
        else:
            # disable grads on all adapter layers
            for layer_name in self.adapter_layer_names:
                layer = getattr(self, layer_name)
                layer.requires_grad_(False)
            self._disable_adapters = True

    def set_adapter(self, adapter_names: str | list[str]) -> None:
        """Set the active adapter(s).

        Args:
            adapter_name (`str` or `List[str]`): Name of the adapter(s) to be activated.
        """
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Deactivate grads on the inactive adapter and activate grads on the active adapter
        for layer_name in self.adapter_layer_names:
            module_dict = getattr(self, layer_name)
            for key, layer in module_dict.items():
                if key in adapter_names:
                    # Note: It is possible that not a single layer is called with requires_grad_(True) here. This may
                    # happen if a completely different adapter layer is being activated.
                    layer.requires_grad_(True)
                else:
                    layer.requires_grad_(False)

        self._active_adapter = adapter_names

    def _all_available_adapter_names(self) -> list[str]:
        """Return a sorted list of all available adapter names"""
        adapter_names = set()
        for name in self.adapter_layer_names + self.other_param_names:
            # we check each possible attribute and if it's a dict or ModuleDict, we assume that the keys are the adapter
            # names
            attr = getattr(self, name)
            if hasattr(attr, "keys"):
                adapter_names.update(attr.keys())
        return sorted(adapter_names)

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Delete an adapter from the layer

        This should be called on all adapter layers, or else we will get an inconsistent state.

        This method will also set a new active adapter if the deleted adapter was an active adapter. It is important
        that the new adapter is chosen in a deterministic way, so that the same adapter is chosen on all layers.

        Args:
            adapter_name (`str`): The name of the adapter to delete

        """
        for attr in self.adapter_layer_names + self.other_param_names:
            if adapter_name in getattr(self, attr):
                del getattr(self, attr)[adapter_name]

        if adapter_name in self.active_adapters:
            # choose a new active adapter
            active_adapters = self.active_adapters[:]
            active_adapters.remove(adapter_name)
            if active_adapters:
                self.set_adapter(active_adapters)
            else:
                # no active adapters left, set a new default adapter
                # here we get the list of all adapters existing adapter names and choose the first one
                remaining_adapters = self._all_available_adapter_names()
                if not remaining_adapters:
                    self.set_adapter([])
                else:
                    new_active_adapter = remaining_adapters[0]
                    warnings.warn(
                        f"Adapter {adapter_name} was active which is now deleted. Setting active adapter to "
                        f"{new_active_adapter}."
                    )
                    self.set_adapter(remaining_adapters[0])


def check_target_module_exists(config, key: str) -> bool | re.Match[str] | None:
    """A helper method to check if the passed module's key name matches any of the target modules in the adapter_config.

    Args:
        config (`LoraConfig` | `LycorisConfig`): A config to match target modules from
        key (`str`): A key to search any matches in config

    Returns:
        `bool` | `re.Match[str]` | `None`: True of match object if key matches any target modules from config, False or
        None if no match found
    """
    if isinstance(config.target_modules, str):
        target_module_found = re.fullmatch(config.target_modules, key)
    else:
        target_module_found = key in config.target_modules or any(
            key.endswith(f".{target_key}") for target_key in config.target_modules
        )
        is_using_layer_indexes = getattr(config, "layers_to_transform", None) is not None
        layer_indexing_pattern = getattr(config, "layers_pattern", None)

        if is_using_layer_indexes and target_module_found:
            layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
            layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern

            for pattern in layers_pattern:
                layer_index = re.match(f".*.{pattern}\.(\d+)\.*", key)
                if layer_index is not None:
                    layer_index = int(layer_index.group(1))
                    if isinstance(config.layers_to_transform, int):
                        target_module_found = layer_index == config.layers_to_transform
                    else:
                        target_module_found = layer_index in config.layers_to_transform

                    break
                else:
                    target_module_found = False
    return target_module_found


def inspect_matched_modules(tuner: BaseTuner, adapter_name: str = "default") -> dict:
    """
    A helper function to inspect the set of matched and unmatched modules for a PEFT model and the given adapter.
    """
    config = tuner.peft_config[adapter_name]
    key_list = [key for key, _ in tuner.model.named_modules()]
    module_dict = {"matched": [], "unmatched": []}
    for key in key_list:
        if tuner._check_target_module_exists(config, key):
            module_dict["matched"].append(key)
        else:
            module_dict["unmatched"].append(key)
    return module_dict
