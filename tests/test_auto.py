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

from peft import (
    AutoPeftModel,
    AutoPeftModelForCausalLM,
    AutoPeftModelForFeatureExtraction,
    AutoPeftModelForQuestionAnswering,
    AutoPeftModelForSeq2SeqLM,
    AutoPeftModelForSequenceClassification,
    AutoPeftModelForTokenClassification,
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForFeatureExtraction,
    PeftModelForQuestionAnswering,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
)


class PeftAutoModelTester(unittest.TestCase):
    def test_peft_causal_lm(self):
        model_id = "peft-internal-testing/tiny-OPTForCausalLM-lora"
        model = AutoPeftModelForCausalLM.from_pretrained(model_id)
        self.assertTrue(isinstance(model, PeftModelForCausalLM))

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)

            model = AutoPeftModelForCausalLM.from_pretrained(model_id)
            self.assertTrue(isinstance(model, PeftModelForCausalLM))

        # check if kwargs are passed correctly
        model = AutoPeftModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.assertTrue(isinstance(model, PeftModelForCausalLM))
        self.assertTrue(model.base_model.lm_head.weight.dtype == torch.bfloat16)

        adapter_name = "default"
        is_trainable = False
        # This should work
        _ = AutoPeftModelForCausalLM.from_pretrained(model_id, adapter_name, is_trainable, torch_dtype=torch.bfloat16)

    def test_peft_seq2seq_lm(self):
        model_id = "peft-internal-testing/tiny_T5ForSeq2SeqLM-lora"
        model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_id)
        self.assertTrue(isinstance(model, PeftModelForSeq2SeqLM))

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)

            model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_id)
            self.assertTrue(isinstance(model, PeftModelForSeq2SeqLM))

        # check if kwargs are passed correctly
        model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.assertTrue(isinstance(model, PeftModelForSeq2SeqLM))
        self.assertTrue(model.base_model.lm_head.weight.dtype == torch.bfloat16)

        adapter_name = "default"
        is_trainable = False
        # This should work
        _ = AutoPeftModelForSeq2SeqLM.from_pretrained(model_id, adapter_name, is_trainable, torch_dtype=torch.bfloat16)

    def test_peft_sequence_cls(self):
        model_id = "peft-internal-testing/tiny_OPTForSequenceClassification-lora"
        model = AutoPeftModelForSequenceClassification.from_pretrained(model_id)
        self.assertTrue(isinstance(model, PeftModelForSequenceClassification))

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)

            model = AutoPeftModelForSequenceClassification.from_pretrained(model_id)
            self.assertTrue(isinstance(model, PeftModelForSequenceClassification))

        # check if kwargs are passed correctly
        model = AutoPeftModelForSequenceClassification.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.assertTrue(isinstance(model, PeftModelForSequenceClassification))
        self.assertTrue(model.score.original_module.weight.dtype == torch.bfloat16)

        adapter_name = "default"
        is_trainable = False
        # This should work
        _ = AutoPeftModelForSequenceClassification.from_pretrained(
            model_id, adapter_name, is_trainable, torch_dtype=torch.bfloat16
        )

    def test_peft_token_classification(self):
        model_id = "peft-internal-testing/tiny_GPT2ForTokenClassification-lora"
        model = AutoPeftModelForTokenClassification.from_pretrained(model_id)
        self.assertTrue(isinstance(model, PeftModelForTokenClassification))

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)

            model = AutoPeftModelForTokenClassification.from_pretrained(model_id)
            self.assertTrue(isinstance(model, PeftModelForTokenClassification))

        # check if kwargs are passed correctly
        model = AutoPeftModelForTokenClassification.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.assertTrue(isinstance(model, PeftModelForTokenClassification))
        self.assertTrue(model.base_model.classifier.original_module.weight.dtype == torch.bfloat16)

        adapter_name = "default"
        is_trainable = False
        # This should work
        _ = AutoPeftModelForTokenClassification.from_pretrained(
            model_id, adapter_name, is_trainable, torch_dtype=torch.bfloat16
        )

    def test_peft_question_answering(self):
        model_id = "peft-internal-testing/tiny_OPTForQuestionAnswering-lora"
        model = AutoPeftModelForQuestionAnswering.from_pretrained(model_id)
        self.assertTrue(isinstance(model, PeftModelForQuestionAnswering))

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)

            model = AutoPeftModelForQuestionAnswering.from_pretrained(model_id)
            self.assertTrue(isinstance(model, PeftModelForQuestionAnswering))

        # check if kwargs are passed correctly
        model = AutoPeftModelForQuestionAnswering.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.assertTrue(isinstance(model, PeftModelForQuestionAnswering))
        self.assertTrue(model.base_model.qa_outputs.original_module.weight.dtype == torch.bfloat16)

        adapter_name = "default"
        is_trainable = False
        # This should work
        _ = AutoPeftModelForQuestionAnswering.from_pretrained(
            model_id, adapter_name, is_trainable, torch_dtype=torch.bfloat16
        )

    def test_peft_feature_extraction(self):
        model_id = "peft-internal-testing/tiny_OPTForFeatureExtraction-lora"
        model = AutoPeftModelForFeatureExtraction.from_pretrained(model_id)
        self.assertTrue(isinstance(model, PeftModelForFeatureExtraction))

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)

            model = AutoPeftModelForFeatureExtraction.from_pretrained(model_id)
            self.assertTrue(isinstance(model, PeftModelForFeatureExtraction))

        # check if kwargs are passed correctly
        model = AutoPeftModelForFeatureExtraction.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.assertTrue(isinstance(model, PeftModelForFeatureExtraction))
        self.assertTrue(model.base_model.model.decoder.embed_tokens.weight.dtype == torch.bfloat16)

        adapter_name = "default"
        is_trainable = False
        # This should work
        _ = AutoPeftModelForFeatureExtraction.from_pretrained(
            model_id, adapter_name, is_trainable, torch_dtype=torch.bfloat16
        )

    def test_peft_whisper(self):
        model_id = "peft-internal-testing/tiny_WhisperForConditionalGeneration-lora"
        model = AutoPeftModel.from_pretrained(model_id)
        self.assertTrue(isinstance(model, PeftModel))

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)

            model = AutoPeftModel.from_pretrained(model_id)
            self.assertTrue(isinstance(model, PeftModel))

        # check if kwargs are passed correctly
        model = AutoPeftModel.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.assertTrue(isinstance(model, PeftModel))
        self.assertTrue(model.base_model.model.model.encoder.embed_positions.weight.dtype == torch.bfloat16)

        adapter_name = "default"
        is_trainable = False
        # This should work
        _ = AutoPeftModel.from_pretrained(model_id, adapter_name, is_trainable, torch_dtype=torch.bfloat16)
