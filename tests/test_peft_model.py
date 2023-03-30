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
import os
import tempfile
import unittest

import torch
from parameterized import parameterized
from transformers import AutoModelForCausalLM

from peft import (
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)

from .testing_common import PeftTestConfigManager


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
}


class PeftTestMixin:
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"


class PeftModelTester(unittest.TestCase, PeftTestMixin):
    r"""
    Test if the PeftModel behaves as expected. This includes:
    - test if the model has the expected methods

    We use parametrized.expand for debugging purposes to test each model individually.
    """

    def _test_model_attr(self, model_id, config_cls, config_kwargs):
        model = AutoModelForCausalLM.from_pretrained(model_id)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)

        self.assertTrue(hasattr(model, "save_pretrained"))
        self.assertTrue(hasattr(model, "from_pretrained"))
        self.assertTrue(hasattr(model, "push_to_hub"))

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_attributes_parametrized(self, test_name, model_id, config_cls, config_kwargs):
        self._test_model_attr(model_id, config_cls, config_kwargs)

    def _test_prepare_for_training(self, model_id, config_cls, config_kwargs):
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.torch_device)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)

        dummy_input = torch.LongTensor([[1, 1, 1]]).to(self.torch_device)
        dummy_output = model.get_input_embeddings()(dummy_input)

        self.assertTrue(not dummy_output.requires_grad)

        # load with `prepare_model_for_int8_training`
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.torch_device)
        model = prepare_model_for_int8_training(model)

        for param in model.parameters():
            self.assertTrue(not param.requires_grad)

        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)

        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        dummy_input = torch.LongTensor([[1, 1, 1]]).to(self.torch_device)
        dummy_output = model.get_input_embeddings()(dummy_input)

        self.assertTrue(dummy_output.requires_grad)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_prepare_for_training_parametrized(self, test_name, model_id, config_cls, config_kwargs):
        self._test_prepare_for_training(model_id, config_cls, config_kwargs)

    def _test_save_pretrained(self, model_id, config_cls, config_kwargs):
        model = AutoModelForCausalLM.from_pretrained(model_id)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)

            model_from_pretrained = AutoModelForCausalLM.from_pretrained(model_id)
            model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname)

            # check if the state dicts are equal
            state_dict = get_peft_model_state_dict(model)
            state_dict_from_pretrained = get_peft_model_state_dict(model_from_pretrained)

            # check if same keys
            self.assertEqual(state_dict.keys(), state_dict_from_pretrained.keys())

            # check if tensors equal
            for key in state_dict.keys():
                self.assertTrue(
                    torch.allclose(
                        state_dict[key].to(self.torch_device), state_dict_from_pretrained[key].to(self.torch_device)
                    )
                )

            # check if `adapter_model.bin` is present
            self.assertTrue(os.path.exists(os.path.join(tmp_dirname, "adapter_model.bin")))

            # check if `adapter_config.json` is present
            self.assertTrue(os.path.exists(os.path.join(tmp_dirname, "adapter_config.json")))

            # check if `pytorch_model.bin` is not present
            self.assertFalse(os.path.exists(os.path.join(tmp_dirname, "pytorch_model.bin")))

            # check if `config.json` is not present
            self.assertFalse(os.path.exists(os.path.join(tmp_dirname, "config.json")))

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_save_pretrained(self, test_name, model_id, config_cls, config_kwargs):
        self._test_save_pretrained(model_id, config_cls, config_kwargs)

    def _test_merge_layers(self, model_id, config_cls, config_kwargs):
        model = AutoModelForCausalLM.from_pretrained(model_id)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        if config.peft_type != "LORA":
            with self.assertRaises(AttributeError):
                model = model.merge_and_unload()
        elif model.config.model_type == "gpt2":
            with self.assertRaises(ValueError):
                model = model.merge_and_unload()
        else:
            dummy_input = torch.LongTensor([[1, 2, 3, 2, 1]]).to(self.torch_device)
            model.eval()
            logits_lora = model(dummy_input)[0]

            model = model.merge_and_unload()

            logits_merged = model(dummy_input)[0]

            transformers_model = AutoModelForCausalLM.from_pretrained(model_id).to(self.torch_device)

            logits_transformers = transformers_model(dummy_input)[0]

            self.assertTrue(torch.allclose(logits_lora, logits_merged, atol=1e-3, rtol=1e-3))
            self.assertFalse(torch.allclose(logits_merged, logits_transformers, atol=1e-3, rtol=1e-3))

            with tempfile.TemporaryDirectory() as tmp_dirname:
                model.save_pretrained(tmp_dirname)

                model_from_pretrained = AutoModelForCausalLM.from_pretrained(tmp_dirname).to(self.torch_device)

                logits_merged_from_pretrained = model_from_pretrained(dummy_input)[0]

                self.assertTrue(torch.allclose(logits_merged, logits_merged_from_pretrained, atol=1e-3, rtol=1e-3))

    @parameterized.expand(
        PeftTestConfigManager.get_grid_parameters(
            {
                "model_ids": PEFT_DECODER_MODELS_TO_TEST,
                "lora_kwargs": {"init_lora_weights": [False], "merge_weights": [False, True]},
            },
        )
    )
    def test_merge_layers(self, test_name, model_id, config_cls, config_kwargs):
        self._test_merge_layers(model_id, config_cls, config_kwargs)

    def _test_generate(self, model_id, config_cls, config_kwargs):
        model = AutoModelForCausalLM.from_pretrained(model_id)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        input_ids = torch.LongTensor([[1, 1, 1], [2, 1, 2]]).to(self.torch_device)
        attention_mask = torch.LongTensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)

        # check if `generate` works
        _ = model.generate(input_ids=input_ids, attention_mask=attention_mask)

        with self.assertRaises(TypeError):
            # check if `generate` raises an error if no positional arguments are passed
            _ = model.generate(input_ids, attention_mask=attention_mask)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_generate(self, test_name, model_id, config_cls, config_kwargs):
        self._test_generate(model_id, config_cls, config_kwargs)
