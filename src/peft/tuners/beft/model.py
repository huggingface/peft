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

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_BEFT_TARGET_MODULES_MAPPING,
)

from .layer import BeftLayer, Linear


class BeftModel(BaseTuner):
    """
    Creates a Infused Adapter by only fine-tuning the added bias terms of value projections from a pretrained
    transformers model in low-training-data regimes (BEFT). The method is described in detail in
    https://arxiv.org/abs/2509.15974

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`BeftConfig`]): The configuration of the (BEFT) model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The (BEFT) model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import BeftModel, BeftConfig

        >>> config = BeftConfig(
        ...     peft_type="Beft",
        ...     task_type="SEQ_2_SEQ_LM",
        ...     target_modules=["v"],
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> beft_model = BeftModel(model, config, adapter_name="default")
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`BeftConfig`]): The configuration of the (BEFT) model.
    """

    prefix: str = "beft_"
    tuner_layer_cls = BeftLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_BEFT_TARGET_MODULES_MAPPING

    @staticmethod
    def _create_new_module(beft_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = Linear(target, adapter_name, config=beft_config, **kwargs)
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear` is supported."
            )
        return new_module

    def _create_and_replace(
        self,
        beft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        kwargs = {}

        if isinstance(target, BeftLayer):
            target.update_layer(
                adapter_name,
                config=beft_config,
            )
        else:
            new_module = self._create_new_module(beft_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
