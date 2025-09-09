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

from typing import TYPE_CHECKING, Any, Optional

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
    peft_config: PeftConfig,
    model: torch.nn.Module,
    adapter_name: str = "default",
    low_cpu_mem_usage: bool = False,
    state_dict: Optional[dict[str, torch.Tensor]] = None,
) -> torch.nn.Module:
    r"""
    Create PEFT layers and inject them into the model in-place.

    Currently the API does not support prompt learning methods and adaption prompt.

    This function is similar to [`get_peft_model`] but it does not return a [`PeftModel`] instance. Instead, it returns
    the original, mutated instance of the passed model.

    Args:
        peft_config (`PeftConfig`):
            Configuration object containing the parameters of the PEFT model.
        model (`torch.nn.Module`):
            The input model where the adapter will be injected.
        adapter_name (`str`, `optional`, defaults to `"default"`):
            The name of the adapter to be injected, if not provided, the default adapter name is used ("default").
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.
        state_dict (`dict`, *optional*, defaults to `None`)
            If a `state_dict` is passed here, the adapters will be injected based on the entries of the state_dict.
            This can be useful when the exact `target_modules` of the PEFT method is unknown, for instance because the
            checkpoint was created without meta data. Note that the values from the `state_dict` are not used, only the
            keys are used to determine the correct layers that should be adapted.
    """
    if peft_config.is_prompt_learning or peft_config.is_adaption_prompt:
        raise ValueError("`create_and_replace` does not support prompt learning and adaption prompt yet.")

    if peft_config.peft_type not in PEFT_TYPE_TO_TUNER_MAPPING.keys():
        raise ValueError(
            f"`inject_adapter_in_model` does not support {peft_config.peft_type} yet. Please use `get_peft_model`."
        )

    tuner_cls = PEFT_TYPE_TO_TUNER_MAPPING[peft_config.peft_type]

    # By instantiating a peft model we are injecting randomly initialized LoRA layers into the model's modules.
    peft_model = tuner_cls(
        model, peft_config, adapter_name=adapter_name, low_cpu_mem_usage=low_cpu_mem_usage, state_dict=state_dict
    )

    return peft_model.model
