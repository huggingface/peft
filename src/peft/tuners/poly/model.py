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

from contextlib import contextmanager
from typing import Any

import torch
from torch import nn

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_POLY_TARGET_MODULES_MAPPING

from .config import PolyConfig
from .layer import Linear, PolyLayer


class PolyModel(BaseTuner):
    prefix: str = "poly_"
    tuner_layer_cls = PolyLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_POLY_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        poly_config: PolyConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        **optional_kwargs: Any,
    ):
        if isinstance(target, PolyLayer):
            target.update_layer(adapter_name, poly_config)
        else:
            new_module = self._create_new_module(
                poly_config,
                adapter_name,
                target,
            )
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(poly_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            return Linear(target, adapter_name, poly_config, **kwargs)
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`."
            )

    def _register_pre_hooks(self, task_ids):
        """Helper method to register pre hooks."""
        if task_ids is None:
            return []

        def pre_hook(_, args, kwargs):
            kwargs["task_ids"] = task_ids
            return args, kwargs

        handles = []

        for module in self.model.modules():
            if isinstance(module, Linear):
                handle = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
                handles.append(handle)

        return handles

    @contextmanager
    def _manage_pre_hooks(self, task_ids):
        """Context manager to handle the lifecycle of pre hooks."""
        handles = self._register_pre_hooks(task_ids)
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def forward(self, *args, task_ids=None, **kwargs):
        with self._manage_pre_hooks(task_ids):
            return self.model(*args, **kwargs)

    def generate(self, *args, task_ids=None, **kwargs):
        with self._manage_pre_hooks(task_ids):
            return self.model.generate(*args, **kwargs)
