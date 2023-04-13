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
import unittest

import torch
from parameterized import parameterized
from transformers import AutoModelForCausalLM

from .testing_common import PeftCommonTester, PeftTestConfigManager


PEFT_DECODER_MODELS_TO_TEST = [
    "hf-internal-testing/tiny-random-OPTForCausalLM",
    "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "hf-internal-testing/tiny-random-GPT2LMHeadModel",
    "hf-internal-testing/tiny-random-BloomForCausalLM",
    "hf-internal-testing/tiny-random-gpt_neo",
    "hf-internal-testing/tiny-random-GPTJForCausalLM",
]

FULL_GRID = {
    "model_ids": PEFT_DECODER_MODELS_TO_TEST,
    "task_type": "CAUSAL_LM",
}


class PeftDecoderModelTester(unittest.TestCase, PeftCommonTester):
    r"""
    Test if the PeftModel behaves as expected. This includes:
    - test if the model has the expected methods

    We use parametrized.expand for debugging purposes to test each model individually.
    """
    transformers_class = AutoModelForCausalLM

    def prepare_inputs_for_testing(self):
        input_ids = torch.tensor([[1, 1, 1], [1, 2, 1]]).to(self.torch_device)
        attention_mask = torch.tensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)

        input_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        return input_dict

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_attributes_parametrized(self, test_name, model_id, config_cls, config_kwargs):
        self._test_model_attr(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_prepare_for_training_parametrized(self, test_name, model_id, config_cls, config_kwargs):
        self._test_prepare_for_training(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_save_pretrained(self, test_name, model_id, config_cls, config_kwargs):
        self._test_save_pretrained(model_id, config_cls, config_kwargs)

    @parameterized.expand(
        PeftTestConfigManager.get_grid_parameters(
            {
                "model_ids": PEFT_DECODER_MODELS_TO_TEST,
                "lora_kwargs": {"init_lora_weights": [False]},
                "task_type": "CAUSAL_LM",
            },
        )
    )
    def test_merge_layers(self, test_name, model_id, config_cls, config_kwargs):
        self._test_merge_layers(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_generate(self, test_name, model_id, config_cls, config_kwargs):
        self._test_generate(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_generate_half_prec(self, test_name, model_id, config_cls, config_kwargs):
        self._test_generate_half_prec(model_id, config_cls, config_kwargs)
