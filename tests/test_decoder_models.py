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
from unittest.mock import Mock, call, patch

import pytest
import torch
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import AdaLoraConfig, PromptTuningConfig, PromptTuningInit, get_peft_model

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
    def test_prompt_tuning_text_prepare_for_training(self, test_name, model_id, config_cls, config_kwargs):
        # Test that prompt tuning works with text init
        if config_cls != PromptTuningConfig:
            return pytest.skip(f"This test does not apply to {config_cls}")

        config_kwargs = config_kwargs.copy()
        config_kwargs["prompt_tuning_init"] = PromptTuningInit.TEXT
        config_kwargs["prompt_tuning_init_text"] = "This is a test prompt."
        config_kwargs["tokenizer_name_or_path"] = model_id
        self._test_prepare_for_training(model_id, config_cls, config_kwargs)

    def test_prompt_tuning_text_tokenizer_kwargs(self):
        # Allow users to pass additional arguments to Tokenizer.from_pretrained
        # Fix for #1032
        mock = Mock()
        orig_from_pretrained = AutoTokenizer.from_pretrained

        def mock_autotokenizer_from_pretrained(*args, **kwargs):
            mock(*args, **kwargs)
            return orig_from_pretrained(config.tokenizer_name_or_path)

        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        config = PromptTuningConfig(
            base_model_name_or_path=model_id,
            tokenizer_name_or_path=model_id,
            num_virtual_tokens=10,
            prompt_tuning_init=PromptTuningInit.TEXT,
            task_type="CAUSAL_LM",
            prompt_tuning_init_text="This is a test prompt.",
            tokenizer_kwargs={"trust_remote_code": True, "foo": "bar"},
        )
        model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
        with patch("transformers.AutoTokenizer.from_pretrained", mock_autotokenizer_from_pretrained):
            model = get_peft_model(model, config)

        expected_call = call(model_id, trust_remote_code=True, foo="bar")
        assert mock.call_args == expected_call

    def test_prompt_tuning_config_invalid_args(self):
        # Raise an error when tokenizer_kwargs is used with prompt_tuning_init!='TEXT', because this argument has no
        # function in that case
        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        with pytest.raises(ValueError, match="tokenizer_kwargs only valid when using prompt_tuning_init='TEXT'."):
            PromptTuningConfig(
                base_model_name_or_path=model_id,
                tokenizer_name_or_path=model_id,
                num_virtual_tokens=10,
                task_type="CAUSAL_LM",
                prompt_tuning_init_text="This is a test prompt.",
                prompt_tuning_init=PromptTuningInit.RANDOM,  # <= should not be used together with tokenizer_kwargs
                tokenizer_kwargs={"trust_remote_code": True, "foo": "bar"},
            )

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_save_pretrained(self, test_name, model_id, config_cls, config_kwargs):
        self._test_save_pretrained(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_save_pretrained_pickle(self, test_name, model_id, config_cls, config_kwargs):
        self._test_save_pretrained(model_id, config_cls, config_kwargs, safe_serialization=False)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_save_pretrained_selected_adapters(self, test_name, model_id, config_cls, config_kwargs):
        self._test_save_pretrained_selected_adapters(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_save_pretrained_selected_adapters_pickle(self, test_name, model_id, config_cls, config_kwargs):
        self._test_save_pretrained_selected_adapters(model_id, config_cls, config_kwargs, safe_serialization=False)

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
    def test_merge_layers_multi(self, test_name, model_id, config_cls, config_kwargs):
        self._test_merge_layers_multi(model_id, config_cls, config_kwargs)

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
    def test_merge_layers_nan(self, test_name, model_id, config_cls, config_kwargs):
        self._test_merge_layers_nan(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_generate(self, test_name, model_id, config_cls, config_kwargs):
        self._test_generate(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_generate_pos_args(self, test_name, model_id, config_cls, config_kwargs):
        # positional args are supported for PeftModelForCausalLM
        self._test_generate_pos_args(model_id, config_cls, config_kwargs, raises_err=False)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_merge_layers_fp16(self, test_name, model_id, config_cls, config_kwargs):
        self._test_merge_layers_fp16(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_generate_half_prec(self, test_name, model_id, config_cls, config_kwargs):
        self._test_generate_half_prec(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
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
    def test_delete_inactive_adapter(self, test_name, model_id, config_cls, config_kwargs):
        self._test_delete_inactive_adapter(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_adding_multiple_adapters_with_bias_raises(self, test_name, model_id, config_cls, config_kwargs):
        self._test_adding_multiple_adapters_with_bias_raises(model_id, config_cls, config_kwargs)

    @parameterized.expand(
        PeftTestConfigManager.get_grid_parameters(
            {
                "model_ids": PEFT_DECODER_MODELS_TO_TEST,
                "lora_kwargs": {"init_lora_weights": [False]},
                "adalora_kwargs": {"init_lora_weights": [False]},
                "ia3_kwargs": {"init_ia3_weights": [False]},
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

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
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

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_passing_input_embeds_works(self, test_name, model_id, config_cls, config_kwargs):
        self._test_passing_input_embeds_works(test_name, model_id, config_cls, config_kwargs)
