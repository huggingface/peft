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


class BaseTunerMixin(ABC):
    r"""
    A base tuner model mixin that provides the common methods and attributes for all tuners.

    Each `BaseTunerMixin` class needs to be implemented in case the adapter is a plugable adapter.
    """

    @staticmethod
    @abstractmethod
    def create_and_replace(
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        **optionnal_kwargs,
    ):
        ...


class BaseTunerLayerMixin:
    r"""
    A tuner layer mixin that provides the common methods and attributes for all tuners.

    Args:
        is_plugable (`bool`, *optional*):
            Whether the adapter layer can be plugged to any pytorch module
        active_adapter (`str`, *optional*):
            The name of the active adapter.
    """
    _is_plugable = False
    active_adapter = None

    @property
    def peft_is_plugable(self):
        return self._is_plugable

    def __post_init__(self):
        if self.active_adapter is None:
            raise ValueError("active_adapter must be set in the subclass")
