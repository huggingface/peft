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

import torch

from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
)
from peft.utils import TRANSFORMERS_MODELS_TO_PEANUT_TARGET_MODULES_MAPPING

from .layer import Linear, PeanutLayer


class PeanutModel(BaseTuner):
    """
    Creates a PEANuT model from a pretrained transformers model.

    The method is described in detail in https://arxiv.org/abs/2410.01870.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`PeanutConfig`]): The configuration of the PEANuT model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The PEANuT PEFT model.

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`PeanutConfig`]): The configuration of the PEANuT model.
    """

    prefix: str = "peanut_"
    tuner_layer_cls = PeanutLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_PEANUT_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        peanut_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        if isinstance(target, PeanutLayer):
            target.update_layer(
                adapter_name,
                peanut_config.r,
                config=peanut_config,
            )
        else:
            new_module = self._create_new_module(peanut_config, adapter_name, target)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(peanut_config, adapter_name, target):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            return Linear(
                target,
                adapter_name,
                r=peanut_config.r,
                config=peanut_config,
            )

        raise NotImplementedError(f"PEANuT does not support target modules of type {type(target_base_layer)} yet.")
