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
from abc import ABC, abstractmethod

from torch import nn

from ..config import PeftConfig
from ..utils import _get_submodules


class BaseTuner(nn.Module, ABC):
    r"""
    A base tuner model that provides the common methods and attributes for all tuners. Each `BaseTuner` class needs to
    be implemented in case the adapter is a plugable adapter.
    """

    def __init__(self, model, peft_config, adapter_name):
        super().__init__()

        self.model = model
        self.forward = self.model.forward

        # Some adapters might be already attached
        if not hasattr(self, "peft_config"):
            self.peft_config = peft_config

        # transformers models have a .config attribute, whose presence is assumed later on
        if not hasattr(self, "config"):
            self.config = {"model_type": "custom"}

        self.create_and_replace(self.model, adapter_name)

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
        **optionnal_kwargs: dict,
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
            **optionnal_kwargs (`dict`):
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

    def create_and_replace(self, model: nn.Module, adapter_name: str):
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

        is_target_modules_in_base_model = False
        key_list = [key for key, _ in model.named_modules()]

        model_config = getattr(model, "config", {"model_type": "custom"})
        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()

        peft_config = self._prepare_adapter_config(peft_config, model_config)

        for key in key_list:
            if not self._check_target_module_exists(peft_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(model, key)

            optionnal_kwargs = {
                "loaded_in_8bit": getattr(model, "is_loaded_in_8bit", False),
                "loaded_in_4bit": getattr(model, "is_loaded_in_4bit", False),
                "current_key": key,
            }
            self._create_and_replace(peft_config, adapter_name, target, target_name, parent, **optionnal_kwargs)

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

    def merge_adapter(self):
        """
        This method merges the LoRa layers into the base model.
        """
        for module in self.model.modules():
            if isinstance(module, BaseTunerLayer):
                module.merge()

    def unmerge_adapter(self):
        """
        This method unmerges the LoRa layers from the base model.
        """
        for module in self.model.modules():
            if isinstance(module, BaseTunerLayer):
                module.unmerge()


class BaseTunerLayer(ABC):
    r"""
    A tuner layer mixin that provides the common methods and attributes for all tuners.

    Args:
        is_plugable (`bool`, *optional*):
            Whether the adapter layer can be plugged to any pytorch module
        active_adapter (`str`, *optional*):
            The name of the active adapter.
    """
    active_adapter = None

    def merge(self):
        raise NotImplementedError

    def unmerge(self):
        raise NotImplementedError
