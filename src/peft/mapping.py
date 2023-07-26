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

from typing import TYPE_CHECKING, Any, Dict

from .peft_model import (
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForFeatureExtraction,
    PeftModelForQuestionAnswering,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
)
from .tuners import (
    TUNERS_MAPPING,
    AdaLoraConfig,
    AdaptionPromptConfig,
    IA3Config,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
)
from .utils import _get_submodules, _prepare_prompt_learning_config


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from .config import PeftConfig


MODEL_TYPE_TO_PEFT_MODEL_MAPPING = {
    "SEQ_CLS": PeftModelForSequenceClassification,
    "SEQ_2_SEQ_LM": PeftModelForSeq2SeqLM,
    "CAUSAL_LM": PeftModelForCausalLM,
    "TOKEN_CLS": PeftModelForTokenClassification,
    "QUESTION_ANS": PeftModelForQuestionAnswering,
    "FEATURE_EXTRACTION": PeftModelForFeatureExtraction,
}

PEFT_TYPE_TO_CONFIG_MAPPING = {
    "ADAPTION_PROMPT": AdaptionPromptConfig,
    "PROMPT_TUNING": PromptTuningConfig,
    "PREFIX_TUNING": PrefixTuningConfig,
    "P_TUNING": PromptEncoderConfig,
    "LORA": LoraConfig,
    "ADALORA": AdaLoraConfig,
    "IA3": IA3Config,
}


def get_peft_config(config_dict: Dict[str, Any]):
    """
    Returns a Peft config object from a dictionary.

    Args:
        config_dict (`Dict[str, Any]`): Dictionary containing the configuration parameters.
    """

    return PEFT_TYPE_TO_CONFIG_MAPPING[config_dict["peft_type"]](**config_dict)


def get_peft_model(model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default") -> PeftModel:
    """
    Returns a Peft model object from a model and a config.

    Args:
        model ([`transformers.PreTrainedModel`]): Model to be wrapped.
        peft_config ([`PeftConfig`]): Configuration object containing the parameters of the Peft model.
    """
    model_config = getattr(model, "config", {"model_type": "custom"})
    if hasattr(model_config, "to_dict"):
        model_config = model_config.to_dict()

    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)

    if peft_config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys() and not peft_config.is_prompt_learning:
        return PeftModel(model, peft_config, adapter_name=adapter_name)
    if peft_config.is_prompt_learning:
        peft_config = _prepare_prompt_learning_config(peft_config, model_config)
    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](model, peft_config, adapter_name=adapter_name)


# TODO: docstring and typehints
def create_and_replace(peft_type, peft_config, model, adapter_name):
    if peft_type not in TUNERS_MAPPING:
        raise ValueError(
            f"Task type {peft_type} is not supported. Supported task types are {list(TUNERS_MAPPING.keys())}"
        )
    tuner_cls = TUNERS_MAPPING[peft_type]

    is_target_modules_in_base_model = False
    key_list = [key for key, _ in model.named_modules()]

    for key in key_list:
        if not tuner_cls._check_target_module_exists(peft_config, key):
            continue

        is_target_modules_in_base_model = True
        parent, target, target_name = _get_submodules(model, key)

        optionnal_kwargs = {
            "loaded_in_8bit": getattr(model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(model, "is_loaded_in_4bit", False),
            "current_key": key,
        }

        tuner_cls.create_and_replace(peft_config, adapter_name, target, target_name, parent, **optionnal_kwargs)

    if not is_target_modules_in_base_model:
        raise ValueError(
            f"Target modules {peft_config.target_modules} not found in the base model. "
            f"Please check the target modules and try again."
        )
