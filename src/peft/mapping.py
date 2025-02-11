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

from typing import TYPE_CHECKING, Any

import torch

from .utils import PeftType


if TYPE_CHECKING:
    from .config import PeftConfig
    from .tuners.tuners_utils import BaseTuner


# these will be filled by the register_peft_method function
PEFT_TYPE_TO_CONFIG_MAPPING: dict[PeftType, type[PeftConfig]] = {}
PEFT_TYPE_TO_TUNER_MAPPING: dict[PeftType, type[BaseTuner]] = {}
PEFT_TYPE_TO_MIXED_MODEL_MAPPING: dict[PeftType, type[BaseTuner]] = {}
PEFT_TYPE_TO_PREFIX_MAPPING: dict[PeftType, str] = {}


def get_peft_config(config_dict: dict[str, Any]) -> PeftConfig:
    """
    Returns a Peft config object from a dictionary.

    Args:
        config_dict (`Dict[str, Any]`): Dictionary containing the configuration parameters.
    """

    return PEFT_TYPE_TO_CONFIG_MAPPING[config_dict["peft_type"]](**config_dict)


def inject_adapter_in_model(
    peft_config: PeftConfig, model: torch.nn.Module, adapter_name: str = "default", low_cpu_mem_usage: bool = False
) -> torch.nn.Module:
    r"""
    A simple API to create and inject adapter in-place into a model. Currently the API does not support prompt learning
    methods and adaption prompt. Make sure to have the correct `target_names` set in the `peft_config` object. The API
    calls `get_peft_model` under the hood but would be restricted only to non-prompt learning methods.

    Args:
        peft_config (`PeftConfig`):
            Configuration object containing the parameters of the Peft model.
        model (`torch.nn.Module`):
            The input model where the adapter will be injected.
        adapter_name (`str`, `optional`, defaults to `"default"`):
            The name of the adapter to be injected, if not provided, the default adapter name is used ("default").
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.
    """
    if peft_config.is_prompt_learning or peft_config.is_adaption_prompt:
        raise ValueError("`create_and_replace` does not support prompt learning and adaption prompt yet.")

    if peft_config.peft_type not in PEFT_TYPE_TO_TUNER_MAPPING.keys():
        raise ValueError(
            f"`inject_adapter_in_model` does not support {peft_config.peft_type} yet. Please use `get_peft_model`."
        )

    tuner_cls = PEFT_TYPE_TO_TUNER_MAPPING[peft_config.peft_type]

    # By instantiating a peft model we are injecting randomly initialized LoRA layers into the model's modules.
    peft_model = tuner_cls(model, peft_config, adapter_name=adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    return peft_model.model
