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
from unittest import TestCase

import torch

from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
from peft.tuners.adaption_prompt import AdaptionPromptConfig, is_llama_available
from peft.utils.other import prepare_model_for_int8_training
from peft.utils.save_and_load import get_peft_model_state_dict
from tests.testing_common import PeftCommonTester


if is_llama_available():
    # We guard the import statement so that our unit tests will pass in CI environments
    # that don't have a transformers package with Llama.
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel


class AdaptionPromptTester(TestCase, PeftCommonTester):
    """
    Tests for the AdaptionPrompt model.

    These tests were mostly adapted from `test_peft_model.py`, but since we haven't checked in the test checkpoints for
    Llama into `hf-internal-testing`, we separate them for now.
    """

    def setUp(self):
        """Check that llama is available in transformers package before running each test."""
        if not is_llama_available():
            self.skipTest("Llama not available in transformers. Skipping test.")

    @staticmethod
    def _create_test_llama_config():
        """Create a test config for a small Llama model for testing."""
        return LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=4,
            use_cache=False,
        )

    def test_attributes(self) -> None:
        model = LlamaModel(self._create_test_llama_config())
        config = AdaptionPromptConfig(adapter_layers=1, adapter_len=4)
        model = get_peft_model(model, config)

        self.assertTrue(hasattr(model, "save_pretrained"))
        self.assertTrue(hasattr(model, "from_pretrained"))
        self.assertTrue(hasattr(model, "push_to_hub"))

    def test_prepare_for_training(self) -> None:
        model = LlamaForCausalLM(self._create_test_llama_config())
        config = AdaptionPromptConfig(adapter_layers=1, adapter_len=4, task_type="CAUSAL_LM")
        model = get_peft_model(model, config)

        dummy_input = torch.LongTensor([[1, 1, 1]]).to(self.torch_device)
        dummy_output = model.get_input_embeddings()(dummy_input)

        self.assertTrue(not dummy_output.requires_grad)

    def test_prepare_for_int8_training(self) -> None:
        model = LlamaForCausalLM(self._create_test_llama_config())
        model = prepare_model_for_int8_training(model)

        for param in model.parameters():
            self.assertTrue(not param.requires_grad)

        config = AdaptionPromptConfig(adapter_layers=1, adapter_len=4, task_type="CAUSAL_LM")
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

    def test_save_pretrained(self) -> None:
        seed = 420
        torch.manual_seed(seed)
        model = LlamaForCausalLM(self._create_test_llama_config())
        config = AdaptionPromptConfig(adapter_layers=2, adapter_len=4, task_type="CAUSAL_LM")
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)

            torch.manual_seed(seed)
            model_from_pretrained = LlamaForCausalLM(self._create_test_llama_config())
            model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname)

            # check if the state dicts are equal
            state_dict = get_peft_model_state_dict(model)
            state_dict_from_pretrained = get_peft_model_state_dict(model_from_pretrained)

            # check if same keys
            self.assertEqual(state_dict.keys(), state_dict_from_pretrained.keys())

            # Check that the number of saved parameters is 4 -- 2 layers of (tokens and gate).
            self.assertEqual(len(list(state_dict.keys())), 4)

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

    def test_generate(self) -> None:
        model = LlamaForCausalLM(self._create_test_llama_config())
        config = AdaptionPromptConfig(adapter_layers=2, adapter_len=4, task_type="CAUSAL_LM")
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        input_ids = torch.LongTensor([[1, 1, 1], [2, 1, 2]]).to(self.torch_device)
        attention_mask = torch.LongTensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)

        # check if `generate` works
        _ = model.generate(input_ids=input_ids, attention_mask=attention_mask)

        with self.assertRaises(TypeError):
            # check if `generate` raises an error if no positional arguments are passed
            _ = model.generate(input_ids, attention_mask=attention_mask)
