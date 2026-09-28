#  Copyright 2026-present the HuggingFace Inc. team.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License governing permissions and limitations under the License.

import pytest
import torch
from transformers import AutoModelForQuestionAnswering, AutoModelForTokenClassification

from peft import BOFTConfig, IA3Config, LoraConfig, VeraConfig

from .testing_common import PeftCommonTester


# Note: models from peft-internal-testing are just the safetensors versions of hf-internal-testing. The auto classes
# add the token classification / question answering head to the same backbone.
PEFT_TOKEN_CLS_MODELS_TO_TEST = [
    "peft-internal-testing/tiny-random-BertForSequenceClassification",
    "peft-internal-testing/tiny-random-RobertaForSequenceClassification",
]

PEFT_QA_MODELS_TO_TEST = PEFT_TOKEN_CLS_MODELS_TO_TEST


def _all_configs(task_type):
    return [
        (LoraConfig, {"task_type": task_type, "target_modules": None}),
        (IA3Config, {"task_type": task_type, "target_modules": None, "feedforward_modules": None}),
        (BOFTConfig, {"task_type": task_type, "target_modules": None}),
        (VeraConfig, {"task_type": task_type, "target_modules": None, "r": 8}),
    ]


TOKEN_CLS_CONFIGS = _all_configs("TOKEN_CLS")
QA_CONFIGS = _all_configs("QUESTION_ANS")


class TestTokenClassificationModels(PeftCommonTester):
    r"""
    Tests for `PeftModelForTokenClassification`, which overrides `add_adapter` to add its head to `modules_to_save`.
    Most of the functionality is already covered by the other model tests.
    """

    transformers_class = AutoModelForTokenClassification

    def prepare_inputs_for_testing(self):
        input_ids = torch.tensor([[1, 1, 1], [1, 2, 1]]).to(self.torch_device)
        attention_mask = torch.tensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @pytest.mark.parametrize("model_id", PEFT_TOKEN_CLS_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", TOKEN_CLS_CONFIGS)
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_add_adapter_no_autocast_adapter_dtype(self, model_id, config_cls, config_kwargs, dtype):
        self._test_add_adapter_no_autocast_adapter_dtype(model_id, config_cls, config_kwargs.copy(), dtype=dtype)


class TestQuestionAnsweringModels(PeftCommonTester):
    r"""
    Tests for `PeftModelForQuestionAnswering`, which overrides `add_adapter` to add its head to `modules_to_save`. Most
    of the functionality is already covered by the other model tests.
    """

    transformers_class = AutoModelForQuestionAnswering

    def prepare_inputs_for_testing(self):
        input_ids = torch.tensor([[1, 1, 1], [1, 2, 1]]).to(self.torch_device)
        attention_mask = torch.tensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @pytest.mark.parametrize("model_id", PEFT_QA_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", QA_CONFIGS)
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_add_adapter_no_autocast_adapter_dtype(self, model_id, config_cls, config_kwargs, dtype):
        self._test_add_adapter_no_autocast_adapter_dtype(model_id, config_cls, config_kwargs.copy(), dtype=dtype)
