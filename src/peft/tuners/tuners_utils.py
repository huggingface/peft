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

    def __init__(self, model, peft_config, adapter_name, adapter_layer_class=None):
        super().__init__()

        self.model = model
        self.forward = self.model.forward

        # Some adapters might be already attached
        if not hasattr(self, "peft_config"):
            self.peft_config = peft_config

        # transformers models have a .config attribute, whose presence is assumed later on
        if not hasattr(self, "config"):
            self.config = {"model_type": "custom"}

        self.adapter_layer_class = adapter_layer_class
        self.create_and_replace(self.model, adapter_name)

    @abstractmethod
    def _prepare_adapter_config(self, peft_config, model_config):
        ...

    @abstractmethod
    def _check_target_module_exists(peft_config, key):
        ...

    @abstractmethod
    def _create_and_replace(self, peft_config, adapter_name, target, target_name, parent, **optionnal_kwargs):
        ...

    @abstractmethod
    def _mark_only_adapters_as_trainable(self):
        ...

    def create_and_replace(self, model, adapter_name):
        peft_config = self.peft_config[adapter_name]

        if not isinstance(peft_config, PeftConfig):
            raise ValueError(f"peft_config must be an instance of PeftConfig got {type(peft_config)} instead.")

        # TODO: test that
        for module in model.modules():
            if isinstance(module, BaseTunerLayerMixin):
                module.requires_grad_(False)

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
        if not self.adapter_layer_class.supports_merging:
            raise ValueError(f"{self.__class__.__name__} does not support merging adapter layers")
        for module in self.model.modules():
            if isinstance(module, self.adapter_layer_class):
                module.merge()

    def unmerge_adapter(self):
        """
        This method unmerges the LoRa layers from the base model.
        """
        if not self.adapter_layer_class.supports_merging:
            raise ValueError(f"{self.__class__.__name__} does not support merging adapter layers")

        for module in self.model.modules():
            if isinstance(module, self.adapter_layer_class):
                module.unmerge()


class BaseTunerLayerMixin:
    r"""
    A tuner layer mixin that provides the common methods and attributes for all tuners.

    Args:
        is_plugable (`bool`, *optional*):
            Whether the adapter layer can be plugged to any pytorch module
        active_adapter (`str`, *optional*):
            The name of the active adapter.
    """
    active_adapter = None
    supports_merging = False
    _is_peft_tuner_layer = True

    def merge(self):
        raise NotImplementedError

    def unmerge(self):
        raise NotImplementedError
