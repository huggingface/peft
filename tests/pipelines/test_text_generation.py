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
import tempfile
import unittest

import torch
from parameterized import parameterized
from transformers import AutoModelForCausalLM

from peft import LoraConfig, PeftModel, get_peft_model, peft_pipeline

from ..testing_common import PeftTestConfigManager


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


class PeftTextGenerationPipelineTester:
    r"""
    A large testing suite for testing common functionality of the PEFT models.

    Attributes:
        torch_device (`torch.device`):
            The device on which the tests will be run.
    """
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    transformers_class = AutoModelForCausalLM

    def _create_pipeline(self, model_id, config_cls, config_kwargs, save_dir, kwargs={}):
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        model.save_pretrained(save_dir)

        pipe = peft_pipeline("text-generation", save_dir, **kwargs)
        return pipe

    def _test_load_pipeline(self, model_id, config_cls, config_kwargs):
        with tempfile.TemporaryDirectory() as tmp_dirname:
            pipe = self._create_pipeline(model_id, config_cls, config_kwargs, tmp_dirname)
            self.assertTrue(isinstance(pipe.model, PeftModel))

            output = pipe("Hello world")
            self.assertTrue(isinstance(output, list))
            self.assertTrue("generated_text" in output[0].keys())

            output = pipe(["Hello world", "Bonjour tout le monde"])
            self.assertTrue(isinstance(output, list))
            self.assertTrue("generated_text" in output[0].keys())

    def _test_load_pipeline_bf16(self, model_id, config_cls, config_kwargs):
        with tempfile.TemporaryDirectory() as tmp_dirname:
            pipe = self._create_pipeline(
                model_id,
                config_cls,
                config_kwargs,
                tmp_dirname,
                kwargs={"base_model_kwargs": {"torch_dtype": torch.bfloat16}},
            )
            self.assertTrue(isinstance(pipe.model, PeftModel))
            self.assertTrue(pipe.model.base_model.dtype == torch.bfloat16)

            output = pipe("Hello world")
            self.assertTrue(isinstance(output, list))
            self.assertTrue("generated_text" in output[0].keys())

    def _test_load_pipeline_merged_lora(self, model_id, config_cls, config_kwargs):
        if not isinstance(config_cls, LoraConfig):
            return

        with tempfile.TemporaryDirectory() as tmp_dirname:
            pipe = self._create_pipeline(
                model_id, config_cls, config_kwargs, tmp_dirname, kwargs={"merge_model": True}
            )
            self.assertTrue(isinstance(pipe.model, AutoModelForCausalLM))

            output = pipe("Hello world")
            self.assertTrue(isinstance(output, list))
            self.assertTrue("generated_text" in output[0].keys())

    def _test_load_pipeline_raise_error(self, model_id, config_cls, config_kwargs):
        with tempfile.TemporaryDirectory() as tmp_dirname:
            with self.assertRaises(TypeError):
                _ = self._create_pipeline(model_id, config_cls, config_kwargs, tmp_dirname, kwargs={"dummy_arg": True})


class PeftEncoderDecoderModelTester(unittest.TestCase, PeftTextGenerationPipelineTester):
    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_load_pipeline(self, test_name, model_id, config_cls, config_kwargs):
        self._test_load_pipeline(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_load_pipeline_bf16(self, test_name, model_id, config_cls, config_kwargs):
        self._test_load_pipeline_bf16(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_load_pipeline_merged_lora(self, test_name, model_id, config_cls, config_kwargs):
        self._test_load_pipeline_merged_lora(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_load_pipeline_raise_error(self, test_name, model_id, config_cls, config_kwargs):
        self._test_load_pipeline_raise_error(model_id, config_cls, config_kwargs)
