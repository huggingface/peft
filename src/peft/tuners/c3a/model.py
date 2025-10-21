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

import re
from itertools import chain

import torch

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_C3A_TARGET_MODULES_MAPPING,
)

from .layer import C3ALayer, C3ALinear


class C3AModel(BaseTuner):
    """
    Creates C3A model from a pretrained transformers model.

    The method is described in detail in https://huggingface.co/papers/2407.19342.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`C3AConfig`]): The configuration of the C3A model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The C3A model.

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`C3AConfig`]): The configuration of the C3A model.
    """

    prefix: str = "c3a_"
    tuner_layer_cls = C3ALayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_C3A_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        c3a_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(c3a_config.block_size_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)

        block_size = c3a_config.block_size_pattern.get(target_name_key, c3a_config.block_size)
        kwargs = {
            "block_size": block_size,
            "init_weights": c3a_config.init_weights,
        }

        if isinstance(target, C3ALinear):
            target.update_layer(
                adapter_name,
                block_size,
                c3a_config.init_weights,
            )
        else:
            new_module = self._create_new_module(c3a_config, adapter_name, target, **kwargs)
            if adapter_name != self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(c3a_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = C3ALinear(target, adapter_name, **kwargs)
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear` is supported."
            )

        return new_module
