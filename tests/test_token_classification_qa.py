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
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    BertConfig,
    BertForQuestionAnswering,
)

from peft import BOFTConfig, IA3Config, LoraConfig, PromptTuningConfig, VeraConfig, get_peft_model

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

    def get_bert_qa_model(self):
        # BERT uses token_type_ids (segment embeddings) and position_ids, which are relevant for extractive QA
        config = BertConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=37,
            type_vocab_size=2,
            max_position_embeddings=64,
        )
        torch.manual_seed(0)
        return BertForQuestionAnswering(config).eval().to(self.torch_device)

    def get_bert_qa_inputs(self):
        torch.manual_seed(0)
        input_ids = torch.randint(5, 100, (2, 10))
        return {
            "input_ids": input_ids.to(self.torch_device),
            "attention_mask": torch.ones_like(input_ids).to(self.torch_device),
            "token_type_ids": torch.tensor([[0] * 4 + [1] * 6, [0] * 5 + [1] * 5]).to(self.torch_device),
            "position_ids": (torch.arange(10).unsqueeze(0).expand(2, -1) + 3).to(self.torch_device),
        }

    @pytest.mark.parametrize("input_name", ["token_type_ids", "position_ids"])
    def test_token_type_ids_and_position_ids_are_passed_to_base_model(self, input_name):
        # PeftModelForQuestionAnswering.forward takes token_type_ids and position_ids as named arguments, check that
        # they reach the base model, see #3857. LoRA is enough here, as all non-prompt-learning methods share this path.
        model = self.get_bert_qa_model()
        inputs = self.get_bert_qa_inputs()
        kwargs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        with torch.inference_mode():
            output_base = model(**kwargs, **{input_name: inputs[input_name]}).start_logits
            output_base_without = model(**kwargs).start_logits
        # sanity check: the input changes the output of the base model
        assert not torch.allclose(output_base, output_base_without)

        config = LoraConfig(task_type="QUESTION_ANS", target_modules=["query", "value"])
        model = get_peft_model(model, config).eval()  # LoRA is initialized as identity transform
        with torch.inference_mode():
            output_peft = model(**kwargs, **{input_name: inputs[input_name]}).start_logits
        assert torch.allclose(output_peft, output_base, atol=1e-4, rtol=1e-4)

    def test_prompt_learning_uses_token_type_ids(self):
        # For prompt learning, token_type_ids are padded for the virtual tokens and passed to the base model
        model = self.get_bert_qa_model()
        inputs = self.get_bert_qa_inputs()
        config = PromptTuningConfig(task_type="QUESTION_ANS", num_virtual_tokens=4)
        model = get_peft_model(model, config).eval()
        kwargs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        with torch.inference_mode():
            output_with = model(**kwargs, token_type_ids=inputs["token_type_ids"]).start_logits
            output_without = model(**kwargs).start_logits
        assert not torch.allclose(output_with, output_without)
