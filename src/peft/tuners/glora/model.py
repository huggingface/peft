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

from contextlib import contextmanager
from functools import partial
from typing import Any

from torch import nn

from peft.tuners.glora.layer import GloraLayer
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_GLORA_TARGET_MODULES_MAPPING

from .config import GloraConfig
from .layer import GloraLinear


def _adapter_names_pre_forward_hook(target, args, kwargs, adapter_names):
    kwargs["adapter_names"] = adapter_names
    return args, kwargs


class GloraModel(BaseTuner):
    """
    Creates Generalized Low Rank Adapter (GLoRA) model from a pretrained transformers model.
    """

    prefix: str = "glora_"
    tuner_layer_cls = GloraLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_GLORA_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        peft_config: GloraConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        **optional_kwargs: Any,
    ) -> None:
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        if isinstance(target, GloraLinear):
            target.update_layer(adapter_name, peft_config.r, config=peft_config)
            if adapter_name not in self.active_adapters:
                target.requires_grad_(False)
        else:
            new_module = self._create_new_module(peft_config, adapter_name, target)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(
        peft_config: GloraConfig,
        adapter_name: str,
        target: nn.Module,
        **optional_kwargs: Any,
    ) -> GloraLinear:
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if type(target_base_layer) is not nn.Linear:
            raise ValueError(
                f"Target module {target} is not a plain torch.nn.Linear (after unwrapping); "
                "GLoRA does not support this layer type."
            )

        return GloraLinear(target, adapter_name, config=peft_config)

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        adapter_names = kwargs.pop("adapter_names", None)
        if adapter_names is None:
            yield
            return

        if self.training:
            raise ValueError("Cannot pass `adapter_names` when the model is in training mode.")

        expected_adapters = set()
        for layer in self.modules():
            if isinstance(layer, GloraLayer):
                expected_adapters |= layer.glora_A.keys()
        unique_adapters = {name for name in adapter_names if name != "__base__"}
        unexpected_adapters = unique_adapters - expected_adapters
        if unexpected_adapters:
            raise ValueError(f"Trying to infer with non-existing adapter(s): {', '.join(sorted(unexpected_adapters))}")

        hook_handles = []
        for module in self.modules():
            if isinstance(module, GloraLayer):
                pre_forward = partial(_adapter_names_pre_forward_hook, adapter_names=adapter_names)
                handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                hook_handles.append(handle)

        yield

        for handle in hook_handles:
            handle.remove()
