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
import gc
import unittest

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, WhisperForConditionalGeneration

from peft import LoraConfig, PeftModel, get_peft_model
from peft.import_utils import is_bnb_4bit_available, is_bnb_available

from .testing_utils import require_bitsandbytes, require_torch_gpu, require_torch_multi_gpu


if is_bnb_available():
    from peft.tuners.lora import Linear8bitLt

    if is_bnb_4bit_available():
        from peft.tuners.lora import Linear4bit


@require_torch_gpu
class PeftGPUCommonTests(unittest.TestCase):
    r"""
    A common tester to run common operations that are performed on GPU such as generation, loading in 8bit, etc.
    """

    def setUp(self):
        self.seq2seq_model_id = "google/flan-t5-base"
        self.causal_lm_model_id = "facebook/opt-350m"
        self.audio_model_id = "openai/whisper-large"
        self.device = torch.device("cuda:0")

    def tearDown(self):
        r"""
        Efficient mechanism to free GPU memory after each test. Based on
        https://github.com/huggingface/transformers/issues/21094
        """
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    @require_bitsandbytes
    @pytest.mark.multi_gpu_tests
    @pytest.mark.single_gpu_tests
    def test_lora_bnb_8bit_quantization(self):
        r"""
        Test that tests if the 8bit quantization using LoRA works as expected
        """
        whisper_8bit = WhisperForConditionalGeneration.from_pretrained(
            self.audio_model_id,
            device_map="auto",
            load_in_8bit=True,
        )

        opt_8bit = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id,
            device_map="auto",
            load_in_8bit=True,
        )

        flan_8bit = AutoModelForSeq2SeqLM.from_pretrained(
            self.seq2seq_model_id,
            device_map="auto",
            load_in_8bit=True,
        )

        flan_lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
        )

        opt_lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

        flan_8bit = get_peft_model(flan_8bit, flan_lora_config)
        self.assertTrue(isinstance(flan_8bit.base_model.model.encoder.block[0].layer[0].SelfAttention.q, Linear8bitLt))

        opt_8bit = get_peft_model(opt_8bit, opt_lora_config)
        self.assertTrue(isinstance(opt_8bit.base_model.model.model.decoder.layers[0].self_attn.v_proj, Linear8bitLt))

        whisper_8bit = get_peft_model(whisper_8bit, config)
        self.assertTrue(
            isinstance(whisper_8bit.base_model.model.model.decoder.layers[0].self_attn.v_proj, Linear8bitLt)
        )

    @require_bitsandbytes
    @pytest.mark.multi_gpu_tests
    @pytest.mark.single_gpu_tests
    def test_lora_bnb_4bit_quantization_from_pretrained_safetensors(self):
        r"""
        Test that tests if the 4bit quantization using LoRA works as expected with safetensors weights.
        """
        model_id = "facebook/opt-350m"
        peft_model_id = "ybelkada/test-st-lora"

        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True)
        model = PeftModel.from_pretrained(model, peft_model_id)

        _ = model.generate(input_ids=torch.LongTensor([[0, 2, 3, 1]]).to(0))

    @require_bitsandbytes
    @pytest.mark.multi_gpu_tests
    @pytest.mark.single_gpu_tests
    def test_lora_bnb_4bit_quantization(self):
        r"""
        Test that tests if the 4bit quantization using LoRA works as expected
        """
        whisper_4bit = WhisperForConditionalGeneration.from_pretrained(
            self.audio_model_id,
            device_map="auto",
            load_in_4bit=True,
        )

        opt_4bit = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id,
            device_map="auto",
            load_in_4bit=True,
        )

        flan_4bit = AutoModelForSeq2SeqLM.from_pretrained(
            self.seq2seq_model_id,
            device_map="auto",
            load_in_4bit=True,
        )

        flan_lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
        )

        opt_lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

        flan_4bit = get_peft_model(flan_4bit, flan_lora_config)
        self.assertTrue(isinstance(flan_4bit.base_model.model.encoder.block[0].layer[0].SelfAttention.q, Linear4bit))

        opt_4bit = get_peft_model(opt_4bit, opt_lora_config)
        self.assertTrue(isinstance(opt_4bit.base_model.model.model.decoder.layers[0].self_attn.v_proj, Linear4bit))

        whisper_4bit = get_peft_model(whisper_4bit, config)
        self.assertTrue(isinstance(whisper_4bit.base_model.model.model.decoder.layers[0].self_attn.v_proj, Linear4bit))

    @pytest.mark.multi_gpu_tests
    @require_torch_multi_gpu
    def test_lora_causal_lm_mutli_gpu_inference(self):
        r"""
        Test if LORA can be used for inference on multiple GPUs.
        """
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id, device_map="balanced")
        tokenizer = AutoTokenizer.from_pretrained(self.seq2seq_model_id)

        self.assertEqual(set(model.hf_device_map.values()), {0, 1})

        model = get_peft_model(model, lora_config)
        self.assertTrue(isinstance(model, PeftModel))

        dummy_input = "This is a dummy input:"
        input_ids = tokenizer(dummy_input, return_tensors="pt").input_ids.to(self.device)

        # this should work without any problem
        _ = model.generate(input_ids=input_ids)

    @require_torch_multi_gpu
    @pytest.mark.multi_gpu_tests
    @require_bitsandbytes
    def test_lora_seq2seq_lm_mutli_gpu_inference(self):
        r"""
        Test if LORA can be used for inference on multiple GPUs - 8bit version.
        """
        lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(self.seq2seq_model_id, device_map="balanced", load_in_8bit=True)
        tokenizer = AutoTokenizer.from_pretrained(self.seq2seq_model_id)

        self.assertEqual(set(model.hf_device_map.values()), {0, 1})

        model = get_peft_model(model, lora_config)
        self.assertTrue(isinstance(model, PeftModel))
        self.assertTrue(isinstance(model.base_model.model.encoder.block[0].layer[0].SelfAttention.q, Linear8bitLt))

        dummy_input = "This is a dummy input:"
        input_ids = tokenizer(dummy_input, return_tensors="pt").input_ids.to(self.device)

        # this should work without any problem
        _ = model.generate(input_ids=input_ids)
