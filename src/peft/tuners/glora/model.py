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

from typing import Any

from torch import nn

from peft.tuners.glora.layer import GloraLayer
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_GLORA_TARGET_MODULES_MAPPING

from .config import GloraConfig
from .layer import GloraLinear


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
