#  Copyright 2025-present the HuggingFace Inc. team.
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
from transformers import AutoModelForSequenceClassification

from peft import (
    AdaLoraConfig,
    BOFTConfig,
    BoneConfig,
    C3AConfig,
    FourierFTConfig,
    HRAConfig,
    IA3Config,
    LoraConfig,
    OFTConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    PromptTuningInit,
    VBLoRAConfig,
    VeraConfig,
)

from .testing_common import PeftCommonTester


PEFT_SEQ_CLS_MODELS_TO_TEST = [
    "hf-internal-testing/tiny-random-BertForSequenceClassification",
    "hf-internal-testing/tiny-random-RobertaForSequenceClassification",
    "trl-internal-testing/tiny-LlamaForSequenceClassification-3.2",
]


ALL_CONFIGS = [
    (
        AdaLoraConfig,
        {
            "task_type": "SEQ_CLS",
            "target_modules": None,
            "total_step": 1,
        },
    ),
    (
        BOFTConfig,
        {
            "task_type": "SEQ_CLS",
            "target_modules": None,
        },
    ),
    (
        BoneConfig,
        {
            "task_type": "SEQ_CLS",
            "target_modules": None,
            "r": 2,
        },
    ),
    (
        FourierFTConfig,
        {
            "task_type": "SEQ_CLS",
            "n_frequency": 10,
            "target_modules": None,
        },
    ),
    (
        HRAConfig,
        {
            "task_type": "SEQ_CLS",
            "target_modules": None,
        },
    ),
    (
        IA3Config,
        {
            "task_type": "SEQ_CLS",
            "target_modules": None,
            "feedforward_modules": None,
        },
    ),
    (
        LoraConfig,
        {
            "task_type": "SEQ_CLS",
            "r": 8,
            "lora_alpha": 32,
            "target_modules": None,
            "lora_dropout": 0.05,
            "bias": "none",
        },
    ),
    #  LoRA + trainable tokens
    (
        LoraConfig,
        {
            "task_type": "SEQ_CLS",
            "r": 8,
            "lora_alpha": 32,
            "target_modules": None,
            "lora_dropout": 0.05,
            "bias": "none",
            "trainable_token_indices": [0, 1, 3],
        },
    ),
    (
        OFTConfig,
        {
            "task_type": "SEQ_CLS",
            "target_modules": None,
        },
    ),
    (
        PrefixTuningConfig,
        {
            "task_type": "SEQ_CLS",
            "num_virtual_tokens": 10,
        },
    ),
    (
        PromptEncoderConfig,
        {
            "task_type": "SEQ_CLS",
            "num_virtual_tokens": 10,
            "encoder_hidden_size": 32,
        },
    ),
    (
        PromptTuningConfig,
        {
            "task_type": "SEQ_CLS",
            "num_virtual_tokens": 10,
        },
    ),
    (
        VBLoRAConfig,
        {
            "task_type": "SEQ_CLS",
            "target_modules": None,
            "vblora_dropout": 0.05,
            "vector_length": 1,
            "num_vectors": 2,
        },
    ),
    (
        VeraConfig,
        {
            "task_type": "SEQ_CLS",
            "r": 8,
            "target_modules": None,
            "vera_dropout": 0.05,
            "projection_prng_key": 0xFF,
            "d_initial": 0.1,
            "save_projection": True,
            "bias": "none",
        },
    ),
    (
        C3AConfig,
        {
            "task_type": "SEQ_CLS",
            "block_size": 1,
            "target_modules": None,
        },
    ),
]


class TestSequenceClassificationModels(PeftCommonTester):
    r"""
    Tests for basic coverage of AutoModelForSequenceClassification and classification-specific cases. Most of the
    functionality is probably already covered by other tests.
    """

    transformers_class = AutoModelForSequenceClassification

    def skipTest(self, reason=""):
        #  for backwards compatibility with unittest style test classes
        pytest.skip(reason)

    def prepare_inputs_for_testing(self):
        input_ids = torch.tensor([[1, 1, 1], [1, 2, 1]]).to(self.torch_device)
        attention_mask = torch.tensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @pytest.mark.parametrize("model_id", PEFT_SEQ_CLS_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_attributes_parametrized(self, model_id, config_cls, config_kwargs):
        self._test_model_attr(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_SEQ_CLS_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_adapter_name(self, model_id, config_cls, config_kwargs):
        self._test_adapter_name(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_SEQ_CLS_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_prepare_for_training_parametrized(self, model_id, config_cls, config_kwargs):
        self._test_prepare_for_training(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_SEQ_CLS_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_prompt_tuning_text_prepare_for_training(self, model_id, config_cls, config_kwargs):
        if config_cls != PromptTuningConfig:
            pytest.skip(f"This test does not apply to {config_cls}")
        config_kwargs = config_kwargs.copy()
        config_kwargs["prompt_tuning_init"] = PromptTuningInit.TEXT
        config_kwargs["prompt_tuning_init_text"] = "This is a test prompt."
        config_kwargs["tokenizer_name_or_path"] = model_id
        self._test_prepare_for_training(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_SEQ_CLS_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained(self, model_id, config_cls, config_kwargs):
        self._test_save_pretrained(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_SEQ_CLS_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained_pickle(self, model_id, config_cls, config_kwargs):
        self._test_save_pretrained(model_id, config_cls, config_kwargs.copy(), safe_serialization=False)

    @pytest.mark.parametrize("model_id", PEFT_SEQ_CLS_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained_selected_adapters(self, model_id, config_cls, config_kwargs):
        self._test_save_pretrained_selected_adapters(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_SEQ_CLS_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained_selected_adapters_pickle(self, model_id, config_cls, config_kwargs):
        self._test_save_pretrained_selected_adapters(
            model_id, config_cls, config_kwargs.copy(), safe_serialization=False
        )

    @pytest.mark.parametrize("model_id", PEFT_SEQ_CLS_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_from_pretrained_config_construction(self, model_id, config_cls, config_kwargs):
        self._test_from_pretrained_config_construction(model_id, config_cls, config_kwargs.copy())
