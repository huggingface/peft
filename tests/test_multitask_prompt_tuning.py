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

import importlib
import os
import tempfile
from unittest import TestCase

import pytest
import torch
from parameterized import parameterized
from torch.testing import assert_close

from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
from peft.tuners.multitask_prompt_tuning import MultitaskPromptTuningConfig, MultitaskPromptTuningInit
from peft.utils.other import WEIGHTS_NAME, prepare_model_for_int8_training
from peft.utils.save_and_load import get_peft_model_state_dict
from tests.testing_common import PeftCommonTester


def is_llama_available() -> bool:
    """Check if Llama is available in the transformers library (it's not in earlier versions)."""
    try:
        return importlib.util.find_spec("transformers.models.llama.modeling_llama") is not None
    except ModuleNotFoundError:
        return False


if is_llama_available():
    # We guard the import statement so that our unit tests will pass in CI environments
    # that don't have a transformers package with Llama.
    from transformers import LlamaConfig, LlamaForCausalLM


class MultiTaskPromptTuningTester(TestCase, PeftCommonTester):
    """
    Tests for the AdaptionPrompt model.

    Some of these tests were adapted from `test_peft_model.py` (which has been refactored since), but since we haven't
    checked in the test checkpoints for Llama into `hf-internal-testing`, we separate them for now.
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
            num_hidden_layers=8,
            num_attention_heads=4,
            use_cache=False,
        )

    @classmethod
    def _create_multitask_prompt_tuning_config(cls) -> MultitaskPromptTuningConfig:
        return MultitaskPromptTuningConfig(
            task_type="CAUSAL_LM",
            num_virtual_tokens=50,
            num_tasks=3,
            prompt_tuning_init_text=(
                "classify the following into either positive or negative, or entailment, neutral or contradiction:"
            ),
        )

    def test_prepare_for_training(self) -> None:
        model = LlamaForCausalLM(self._create_test_llama_config())
        model = get_peft_model(model, self._create_multitask_prompt_tuning_config())
        model = model.to(self.torch_device)

        dummy_input = torch.LongTensor([[1, 1, 1]]).to(self.torch_device)
        dummy_output = model.get_input_embeddings()(dummy_input)

        assert not dummy_output.requires_grad

    def test_prepare_for_int8_training(self) -> None:
        model = LlamaForCausalLM(self._create_test_llama_config())
        model = prepare_model_for_int8_training(model)
        model = model.to(self.torch_device)

        for param in model.parameters():
            assert not param.requires_grad

        model = get_peft_model(model, self._create_multitask_prompt_tuning_config())

        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        dummy_input = torch.LongTensor([[1, 1, 1]]).to(self.torch_device)
        dummy_output = model.get_input_embeddings()(dummy_input)

        assert dummy_output.requires_grad

    def test_save_pretrained(self) -> None:
        seed = 420
        torch.manual_seed(seed)
        model = LlamaForCausalLM(self._create_test_llama_config())
        model = get_peft_model(model, self._create_multitask_prompt_tuning_config())
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
            assert state_dict.keys() == state_dict_from_pretrained.keys()

            # Check that the number of saved parameters is 4 -- 2 layers of (tokens and gate).
            assert len(state_dict) == 3

            # check if tensors equal
            for key in state_dict.keys():
                assert torch.allclose(
                    state_dict[key].to(self.torch_device), state_dict_from_pretrained[key].to(self.torch_device)
                )

            # check if `adapter_model.safetensors` is present
            assert os.path.exists(os.path.join(tmp_dirname, "adapter_model.safetensors"))

            # check if `adapter_config.json` is present
            assert os.path.exists(os.path.join(tmp_dirname, "adapter_config.json"))

            # check if `pytorch_model.bin` is not present
            assert not os.path.exists(os.path.join(tmp_dirname, "pytorch_model.bin"))

            # check if `config.json` is not present
            assert not os.path.exists(os.path.join(tmp_dirname, "config.json"))

    def test_save_pretrained_regression(self) -> None:
        seed = 420
        torch.manual_seed(seed)
        model = LlamaForCausalLM(self._create_test_llama_config())
        model = get_peft_model(model, self._create_multitask_prompt_tuning_config())
        model = model.to(self.torch_device)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname, safe_serialization=False)

            torch.manual_seed(seed)
            model_from_pretrained = LlamaForCausalLM(self._create_test_llama_config())
            model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname)

            # check if the state dicts are equal
            state_dict = get_peft_model_state_dict(model)

            state_dict_from_pretrained = get_peft_model_state_dict(model_from_pretrained)

            # check if same keys
            assert state_dict.keys() == state_dict_from_pretrained.keys()

            # Check that the number of saved parameters is 4 -- 2 layers of (tokens and gate).
            assert len(state_dict) == 3

            # check if tensors equal
            for key in state_dict.keys():
                assert torch.allclose(
                    state_dict[key].to(self.torch_device), state_dict_from_pretrained[key].to(self.torch_device)
                )

            # check if `adapter_model.bin` is present for regression
            assert os.path.exists(os.path.join(tmp_dirname, "adapter_model.bin"))

            # check if `adapter_config.json` is present
            assert os.path.exists(os.path.join(tmp_dirname, "adapter_config.json"))

            # check if `pytorch_model.bin` is not present
            assert not os.path.exists(os.path.join(tmp_dirname, "pytorch_model.bin"))

            # check if `config.json` is not present
            assert not os.path.exists(os.path.join(tmp_dirname, "config.json"))

    def test_generate(self) -> None:
        model = LlamaForCausalLM(self._create_test_llama_config())
        model = get_peft_model(model, self._create_multitask_prompt_tuning_config())
        model = model.to(self.torch_device)

        input_ids = torch.LongTensor([[1, 1, 1], [2, 1, 2]]).to(self.torch_device)
        attention_mask = torch.LongTensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)
        task_ids = torch.LongTensor([1, 2]).to(self.torch_device)

        # check if `generate` works
        _ = model.generate(input_ids=input_ids, attention_mask=attention_mask, task_ids=task_ids)

        # check if `generate` works if positional arguments are passed
        _ = model.generate(input_ids, attention_mask=attention_mask, task_ids=task_ids)

    def test_use_cache(self) -> None:
        """Test that MultiTaskPromptTuning works when Llama config use_cache=True."""
        torch.manual_seed(0)
        input_ids = torch.LongTensor([[1, 1, 1], [2, 1, 2]]).to(self.torch_device)
        task_ids = torch.LongTensor([1, 2]).to(self.torch_device)

        original = LlamaForCausalLM(self._create_test_llama_config()).eval()
        mpt = get_peft_model(original, self._create_multitask_prompt_tuning_config())
        mpt = mpt.to(self.torch_device)

        expected = mpt.generate(input_ids=input_ids, max_length=8, task_ids=task_ids)

        # Set use_cache = True and generate output again.
        mpt.base_model.config.use_cache = True
        actual = mpt.generate(input_ids=input_ids, max_length=8, task_ids=task_ids)
        assert_close(expected, actual, rtol=0, atol=0)

    def test_bf16_inference(self) -> None:
        """Test that MultiTaskPromptTuning works when Llama using a half-precision model."""
        input_ids = torch.LongTensor([[1, 1, 1], [2, 1, 2]]).to(self.torch_device)
        task_ids = torch.tensor([1, 2]).to(self.torch_device)

        original = LlamaForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-random-LlamaForCausalLM", torch_dtype=torch.bfloat16
        )
        mpt = get_peft_model(original, self._create_multitask_prompt_tuning_config())
        mpt = mpt.to(self.torch_device)
        _ = mpt.generate(input_ids=input_ids, task_ids=task_ids)

    def test_generate_text_with_random_init(self) -> None:
        model = LlamaForCausalLM(self._create_test_llama_config())

        config = self._create_multitask_prompt_tuning_config()
        config.prompt_tuning_init = MultitaskPromptTuningInit.RANDOM

        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        input_ids = torch.LongTensor([[1, 1, 1], [2, 1, 2]]).to(self.torch_device)
        attention_mask = torch.LongTensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)
        task_ids = torch.LongTensor([0]).to(self.torch_device)

        # check if `generate` works
        _ = model.generate(input_ids=input_ids, attention_mask=attention_mask, task_ids=task_ids)

        with pytest.raises(ValueError):
            # check if `generate` raises an error if task_ids are not passed
            _ = model.generate(input_ids, attention_mask=attention_mask)

    @parameterized.expand(
        [
            MultitaskPromptTuningInit.AVERAGE_SOURCE_TASKS,
            MultitaskPromptTuningInit.EXACT_SOURCE_TASK,
            MultitaskPromptTuningInit.ONLY_SOURCE_SHARED,
        ],
    )
    def test_generate_text_with_other_init(self, prompt_tuning_init) -> None:
        with tempfile.TemporaryDirectory() as tmp_dirname:
            model = LlamaForCausalLM(self._create_test_llama_config())
            model = get_peft_model(model, self._create_multitask_prompt_tuning_config())
            model.save_pretrained(tmp_dirname, safe_serialization=False)  # bc torch.load is used

            config = MultitaskPromptTuningConfig(
                task_type="CAUSAL_LM",
                num_virtual_tokens=50,
                num_tasks=1,
                prompt_tuning_init_text=(
                    "classify the following into either positive or negative, or entailment, neutral or contradiction:"
                ),
                prompt_tuning_init=prompt_tuning_init,
                prompt_tuning_init_state_dict_path=os.path.join(tmp_dirname, WEIGHTS_NAME),
            )
            model = LlamaForCausalLM(self._create_test_llama_config())
            model = get_peft_model(model, config)
            model = model.to(self.torch_device)

            input_ids = torch.LongTensor([[1, 1, 1], [2, 1, 2]]).to(self.torch_device)
            attention_mask = torch.LongTensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)
            task_ids = torch.LongTensor([0]).to(self.torch_device)

            # check if `generate` works
            _ = model.generate(input_ids=input_ids, attention_mask=attention_mask, task_ids=task_ids)

            with pytest.raises(ValueError):
                # check if `generate` raises an error if task_ids are not passed
                _ = model.generate(input_ids, attention_mask=attention_mask)
