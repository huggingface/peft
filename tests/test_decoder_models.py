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

from peft import AdaLoraConfig

from .testing_common import PeftCommonTester, PeftTestConfigManager


PEFT_DECODER_MODELS_TO_TEST = [
    "hf-internal-testing/tiny-random-OPTForCausalLM",
    "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "hf-internal-testing/tiny-random-GPT2LMHeadModel",
    "hf-internal-testing/tiny-random-BloomForCausalLM",
    "hf-internal-testing/tiny-random-gpt_neo",
    "hf-internal-testing/tiny-random-GPTJForCausalLM",
    "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM",
    "HuggingFaceM4/tiny-random-LlamaForCausalLM",
]

FULL_GRID = {
    "model_ids": PEFT_DECODER_MODELS_TO_TEST,
    "task_type": "CAUSAL_LM",
}


def skip_non_pt_mqa(test_list):
    r"""
    Skip tests that are prefix tuning for MQA models (not supported yet)
    """
    return [test for test in test_list if not ("prefix_tuning" in test[0] and "GPTBigCodeForCausalLM" in test[0])]


def skip_adalora_and_gpt2(test_list):
    return [test for test in test_list if not (("GPT2LMHeadModel" in test[1]) and (test[2] == AdaLoraConfig))]


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
    def test_adapter_name(self, test_name, model_id, config_cls, config_kwargs):
        self._test_adapter_name(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_prepare_for_training_parametrized(self, test_name, model_id, config_cls, config_kwargs):
        self._test_prepare_for_training(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_save_pretrained(self, test_name, model_id, config_cls, config_kwargs):
        self._test_save_pretrained(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_save_pretrained_selected_adapters(self, test_name, model_id, config_cls, config_kwargs):
        self._test_save_pretrained_selected_adapters(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_from_pretrained_config_construction(self, test_name, model_id, config_cls, config_kwargs):
        self._test_from_pretrained_config_construction(model_id, config_cls, config_kwargs)

    @parameterized.expand(
        PeftTestConfigManager.get_grid_parameters(
            {
                "model_ids": PEFT_DECODER_MODELS_TO_TEST,
                "lora_kwargs": {"init_lora_weights": [False]},
                "ia3_kwargs": {"init_ia3_weights": [False]},
                "task_type": "CAUSAL_LM",
            },
        )
    )
    def test_merge_layers(self, test_name, model_id, config_cls, config_kwargs):
        self._test_merge_layers(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID, filter_params_func=skip_non_pt_mqa))
    def test_generate(self, test_name, model_id, config_cls, config_kwargs):
        self._test_generate(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID, filter_params_func=skip_non_pt_mqa))
    def test_generate_half_prec(self, test_name, model_id, config_cls, config_kwargs):
        self._test_generate_half_prec(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID, filter_params_func=skip_non_pt_mqa))
    def test_prefix_tuning_half_prec_conversion(self, test_name, model_id, config_cls, config_kwargs):
        self._test_prefix_tuning_half_prec_conversion(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_training_decoders(self, test_name, model_id, config_cls, config_kwargs):
        self._test_training(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_training_decoders_layer_indexing(self, test_name, model_id, config_cls, config_kwargs):
        self._test_training_layer_indexing(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_training_decoders_gradient_checkpointing(self, test_name, model_id, config_cls, config_kwargs):
        self._test_training_gradient_checkpointing(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_inference_safetensors(self, test_name, model_id, config_cls, config_kwargs):
        self._test_inference_safetensors(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_peft_model_device_map(self, test_name, model_id, config_cls, config_kwargs):
        self._test_peft_model_device_map(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_delete_adapter(self, test_name, model_id, config_cls, config_kwargs):
        self._test_delete_adapter(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_adding_multiple_adapters_with_bias_raises(self, test_name, model_id, config_cls, config_kwargs):
        self._test_adding_multiple_adapters_with_bias_raises(model_id, config_cls, config_kwargs)

    @parameterized.expand(
        PeftTestConfigManager.get_grid_parameters(
            {
                "model_ids": PEFT_DECODER_MODELS_TO_TEST,
                "lora_kwargs": {"init_lora_weights": [False]},
                "adalora_kwargs": {"init_lora_weights": [False]},
                "task_type": "CAUSAL_LM",
            },
            filter_params_func=skip_adalora_and_gpt2,
        )
    )
    def test_unload_adapter(self, test_name, model_id, config_cls, config_kwargs):
        self._test_unload_adapter(model_id, config_cls, config_kwargs)

    @parameterized.expand(
        PeftTestConfigManager.get_grid_parameters(
            {
                "model_ids": PEFT_DECODER_MODELS_TO_TEST,
                "lora_kwargs": {"init_lora_weights": [False]},
                "task_type": "CAUSAL_LM",
            },
        )
    )
    def test_weighted_combination_of_adapters(self, test_name, model_id, config_cls, config_kwargs):
        self._test_weighted_combination_of_adapters(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID, filter_params_func=skip_non_pt_mqa))
    def test_training_prompt_learning_tasks(self, test_name, model_id, config_cls, config_kwargs):
        self._test_training_prompt_learning_tasks(model_id, config_cls, config_kwargs)

    @parameterized.expand(
        PeftTestConfigManager.get_grid_parameters(
            {
                "model_ids": PEFT_DECODER_MODELS_TO_TEST,
                "lora_kwargs": {"init_lora_weights": [False]},
                "ia3_kwargs": {"init_ia3_weights": [False]},
                "adalora_kwargs": {"init_lora_weights": [False]},
                "task_type": "CAUSAL_LM",
            },
            filter_params_func=skip_non_pt_mqa,
        )
    )
    def test_disable_adapter(self, test_name, model_id, config_cls, config_kwargs):
        self._test_disable_adapter(model_id, config_cls, config_kwargs)

    def test_generate_adalora_no_dropout(self):
        # test for issue #730
        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        config_kwargs = {
            "target_modules": None,
            "task_type": "CAUSAL_LM",
            "lora_dropout": 0.0,
        }
        self._test_generate(model_id, AdaLoraConfig, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID, filter_params_func=skip_non_pt_mqa))
    def test_passing_input_embeds_works(self, test_name, model_id, config_cls, config_kwargs):
        self._test_passing_input_embeds_works(test_name, model_id, config_cls, config_kwargs)
