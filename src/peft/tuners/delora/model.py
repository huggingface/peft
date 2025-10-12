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

import torch

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_DELORA_TARGET_MODULES_MAPPING,
)
from peft.utils.other import get_pattern_key

from .config import DeloraConfig
from .layer import DeloraLayer, DeloraLinear


class DeloraModel(BaseTuner):
    """
    Creates DeLoRA model from a pretrained transformers model.

    The method is described in detail in [TODO].

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`DeloraConfig`]): The configuration of the DeLoRA model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The DeLoRA model.

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`DeloraConfig`]): The configuration of the DeLoRA model.
    """

    prefix: str = "delora_"
    tuner_layer_cls = DeloraLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_DELORA_TARGET_MODULES_MAPPING

    def _check_new_adapter_config(self, config: DeloraConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        super()._check_new_adapter_config(config)

    def _create_and_replace(
        self,
        delora_config,
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
        r_key = get_pattern_key(delora_config.rank_pattern.keys(), current_key)
        lambda_key = get_pattern_key(delora_config.lambda_pattern.keys(), current_key)
        r = delora_config.rank_pattern.get(r_key, delora_config.r)
        delora_lambda = delora_config.lambda_pattern.get(lambda_key, delora_config.delora_lambda)

        kwargs = {
            "r": r,
            "delora_lambda": delora_lambda,
            "module_dropout": delora_config.module_dropout,
            "init_weights": delora_config.init_weights,
        }

        if isinstance(target, DeloraLinear):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = self._create_new_module(delora_config, adapter_name, target, **kwargs)
            if adapter_name != self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(delora_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = DeloraLinear(target, adapter_name, **kwargs)

        return new_module
