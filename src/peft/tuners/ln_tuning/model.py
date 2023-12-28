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
from __future__ import annotations

from typing import Any

from torch.nn.modules import Module

from peft.config import PeftConfig
from peft.tuners.tuners_utils import BaseTuner, check_target_module_exists
from peft.utils import TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING


class LNTuningModel(BaseTuner):
    """
    Creates LayerNorm tuning from a pretrained transformer model.

    The method is described in detail in https://arxiv.org/abs/2312.11420.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`LNTuningConfig`]): The configuration of the Lora model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        'torch.nn.Module': The adapted model with LayerNorm tuned on.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import LNTuningModel, LNTuningConfig

        >>> config = LNTuningConfig(
        ...     task_type="CAUSAL_LM",
        ... )

        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> ln_model = LNTuningModel(model, config, "default")
        ```

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import get_peft_model, TaskType, LNTuningConfig

        >>> peft_config = LNTuningConfig(
        ...     task_type=TaskType.CAUSAL_LM,
        ... )

        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> model = get_peft_model(model, peft_config)
        >>> model.print_trainable_parameters()
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, model, config, adapter_name) -> None:
        self.adapter_name = adapter_name
        super().__init__(model, config, adapter_name)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @staticmethod
    def _prepare_adapter_config(peft_config: PeftConfig, model_config: dict) -> PeftConfig:
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _create_and_replace(
        self,
        peft_config: PeftConfig,
        adapter_name: str,
        target: Module,
        target_name: str,
        parent: Module,
        **optional_kwargs: Any,
    ) -> None:
        # Don't need to do anything here: just mark the layernorms as trainable is enough
        pass

    def _mark_only_adapters_as_trainable(self, model: Module):
        # Need to mark all layernorms as trainable
        for n, p in model.named_parameters():
            # check to see if the parameter is a layernorm
            flag = False
            for module_name in self.peft_config[self.adapter_name].target_modules:
                if module_name in n:
                    flag = True
                    break
            p.requires_grad = flag

    def _check_target_module_exists(self, peft_config: PeftConfig, key: str) -> bool:
        return check_target_module_exists(peft_config, key)
