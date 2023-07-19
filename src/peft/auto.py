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

import importlib
from typing import Optional

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

from .mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING
from .peft_model import (
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForFeatureExtraction,
    PeftModelForQuestionAnswering,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
)
from .utils import PeftConfig


class _BaseAutoPeftModel:
    _target_class = None
    _target_peft_class = None

    def __init__(self, *args, **kwargs):
        # For consistency with transformers: https://github.com/huggingface/transformers/blob/91d7df58b6537d385e90578dac40204cb550f706/src/transformers/models/auto/auto_factory.py#L400
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        **kwargs,
    ):
        r"""
        A wrapper around all the preprocessing steps a user needs to perform in order to load a PEFT model. The kwargs
        are passed along to `PeftConfig` that automatically takes care of filtering the kwargs of the Hub methods and
        the config object init.
        """
        peft_config = PeftConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        base_model_path = peft_config.base_model_name_or_path

        task_type = getattr(peft_config, "task_type", None)

        if cls._target_class is not None:
            target_class = cls._target_class
        elif cls._target_class is None and task_type is not None:
            # this is only in the case where we use `AutoPeftModel`
            raise ValueError(
                "Cannot use `AutoPeftModel` with a task type, please use a specific class for your task type. (e.g. `AutoPeftModelForCausalLM` for `task_type='CAUSAL_LM'`)"
            )

        if task_type is not None:
            expected_target_class = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[task_type]
            if cls._target_peft_class.__name__ != expected_target_class.__name__:
                raise ValueError(
                    f"Expected target PEFT class: {expected_target_class.__name__}, but you have asked for: {cls._target_peft_class.__name__ }"
                    " make sure that you are loading the correct model for your task type."
                )
        elif task_type is None and getattr(peft_config, "auto_mapping", None) is not None:
            auto_mapping = getattr(peft_config, "auto_mapping", None)
            base_model_class = auto_mapping["base_model_class"]
            parent_library_name = auto_mapping["parent_library"]

            parent_library = importlib.import_module(parent_library_name)
            target_class = getattr(parent_library, base_model_class)
        else:
            raise ValueError(
                "Cannot infer the auto class from the config, please make sure that you are loading the correct model for your task type."
            )

        base_model = target_class.from_pretrained(base_model_path, **kwargs)

        return cls._target_peft_class.from_pretrained(
            base_model,
            pretrained_model_name_or_path,
            adapter_name=adapter_name,
            is_trainable=is_trainable,
            config=config,
            **kwargs,
        )


class AutoPeftModel(_BaseAutoPeftModel):
    _target_class = None
    _target_peft_class = PeftModel


class AutoPeftModelForCausalLM(_BaseAutoPeftModel):
    _target_class = AutoModelForCausalLM
    _target_peft_class = PeftModelForCausalLM


class AutoPeftModelForSeq2SeqLM(_BaseAutoPeftModel):
    _target_class = AutoModelForSeq2SeqLM
    _target_peft_class = PeftModelForSeq2SeqLM


class AutoPeftModelForSequenceClassification(_BaseAutoPeftModel):
    _target_class = AutoModelForSequenceClassification
    _target_peft_class = PeftModelForSequenceClassification


class AutoPeftModelForTokenClassification(_BaseAutoPeftModel):
    _target_class = AutoModelForTokenClassification
    _target_peft_class = PeftModelForTokenClassification


class AutoPeftModelForQuestionAnswering(_BaseAutoPeftModel):
    _target_class = AutoModelForQuestionAnswering
    _target_peft_class = PeftModelForQuestionAnswering


class AutoPeftModelForFeatureExtraction(_BaseAutoPeftModel):
    _target_class = AutoModel
    _target_peft_class = PeftModelForFeatureExtraction
