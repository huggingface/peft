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

from __future__ import annotations

import re

import torch
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_EWORA_TARGET_MODULES_MAPPING

from .layer import EworaLayer, Linear


class EworaModel(BaseTuner):
    """
    Creates an Expert-Weighted LoRA (EWoRA) model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`EworaConfig`]): The configuration of the EWoRA model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The EWoRA model.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import EworaConfig, get_peft_model

        >>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        >>> config = EworaConfig(r=8)
        >>> model = get_peft_model(base_model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`EworaConfig`]): The configuration of the EWoRA model.
    """

    prefix: str = "ewora_"
    tuner_layer_cls = EworaLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_EWORA_TARGET_MODULES_MAPPING

    def _create_and_replace(self, ewora_config, adapter_name, target, target_name, parent, current_key):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # a per-layer rank can be set through `rank_pattern`
        pattern_keys = list(ewora_config.rank_pattern.keys())
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = ewora_config.rank_pattern.get(target_name_key, ewora_config.r)

        if isinstance(target, EworaLayer):
            target.update_layer(adapter_name, r, config=ewora_config)
        else:
            new_module = self._create_new_module(ewora_config, adapter_name, target, r=r)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(ewora_config, adapter_name, target, r):
        target_base_layer = target.get_base_layer() if isinstance(target, BaseTunerLayer) else target

        if isinstance(target_base_layer, (torch.nn.Linear, Conv1D)):
            return Linear(target, adapter_name, config=ewora_config, r=r)

        raise TypeError(
            f"Target module {target} is not supported. Currently, only the following modules are supported: "
            "`torch.nn.Linear`, `transformers.pytorch_utils.Conv1D`."
        )
