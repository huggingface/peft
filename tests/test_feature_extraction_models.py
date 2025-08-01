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
import pytest
import torch
from transformers import AutoModel

from peft import (
    AdaLoraConfig,
    BOFTConfig,
    BoneConfig,
    C3AConfig,
    FourierFTConfig,
    HRAConfig,
    IA3Config,
    LoraConfig,
    MissConfig,
    OFTConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptLearningConfig,
    PromptTuningConfig,
    ShiraConfig,
    VBLoRAConfig,
    VeraConfig,
)

from .testing_common import PeftCommonTester
from .testing_utils import set_init_weights_false


PEFT_FEATURE_EXTRACTION_MODELS_TO_TEST = [
    "hf-internal-testing/tiny-random-BertModel",
    "hf-internal-testing/tiny-random-RobertaModel",
    "hf-internal-testing/tiny-random-DebertaModel",
    "hf-internal-testing/tiny-random-DebertaV2Model",
]

# TODO Missing from this list are LoKr, LoHa, LN Tuning, add them
ALL_CONFIGS = [
    (
        AdaLoraConfig,
        {
            "task_type": "FEATURE_EXTRACTION",
            "target_modules": None,
            "total_step": 1,
        },
    ),
    (
        BOFTConfig,
        {
            "task_type": "FEATURE_EXTRACTION",
            "target_modules": None,
        },
    ),
    (
        BoneConfig,
        {
            "task_type": "FEATURE_EXTRACTION",
            "target_modules": None,
            "r": 2,
        },
    ),
    (
        MissConfig,
        {
            "task_type": "FEATURE_EXTRACTION",
            "target_modules": None,
            "r": 2,
        },
    ),
    (
        FourierFTConfig,
        {
            "task_type": "FEATURE_EXTRACTION",
            "n_frequency": 10,
            "target_modules": None,
        },
    ),
    (
        HRAConfig,
        {
            "task_type": "FEATURE_EXTRACTION",
            "target_modules": None,
        },
    ),
    (
        IA3Config,
        {
            "task_type": "FEATURE_EXTRACTION",
            "target_modules": None,
            "feedforward_modules": None,
        },
    ),
    (
        LoraConfig,
        {
            "task_type": "FEATURE_EXTRACTION",
            "r": 8,
            "lora_alpha": 32,
            "target_modules": None,
            "lora_dropout": 0.05,
            "bias": "none",
        },
    ),
    # LoRA + trainable tokens
    (
        LoraConfig,
        {
            "task_type": "FEATURE_EXTRACTION",
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
            "task_type": "FEATURE_EXTRACTION",
            "target_modules": None,
        },
    ),
    (
        PrefixTuningConfig,
        {
            "task_type": "FEATURE_EXTRACTION",
            "num_virtual_tokens": 10,
        },
    ),
    (
        PromptEncoderConfig,
        {
            "task_type": "FEATURE_EXTRACTION",
            "num_virtual_tokens": 10,
            "encoder_hidden_size": 32,
        },
    ),
    (
        PromptTuningConfig,
        {
            "task_type": "FEATURE_EXTRACTION",
            "num_virtual_tokens": 10,
        },
    ),
    (
        ShiraConfig,
        {
            "r": 1,
            "task_type": "FEATURE_EXTRACTION",
            "target_modules": None,
            "init_weights": False,
        },
    ),
    (
        VBLoRAConfig,
        {
            "task_type": "FEATURE_EXTRACTION",
            "target_modules": None,
            "vblora_dropout": 0.05,
            "vector_length": 1,
            "num_vectors": 2,
        },
    ),
    (
        VeraConfig,
        {
            "task_type": "FEATURE_EXTRACTION",
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
            "task_type": "FEATURE_EXTRACTION",
            "block_size": 1,
            "target_modules": None,
        },
    ),
]


def skip_non_prompt_learning(config_cls):
    if not issubclass(config_cls, PromptLearningConfig) or (config_cls == PrefixTuningConfig):
        pytest.skip("Skip tests that are not prompt learning or that are prefix tuning")


def skip_deberta_lora_tests(config_cls, model_id):
    if "deberta" not in model_id.lower():
        return

    to_skip = ["lora", "ia3", "boft", "vera", "fourierft", "hra", "bone", "randlora"]
    config_name = config_cls.__name__.lower()
    if any(k in config_name for k in to_skip):
        pytest.skip(f"Skip tests that use {config_name} for Deberta models")


def skip_deberta_pt_tests(config_cls, model_id):
    if "deberta" not in model_id.lower():
        return

    to_skip = ["prefix"]
    config_name = config_cls.__name__.lower()
    if any(k in config_name for k in to_skip):
        pytest.skip(f"Skip tests that use {config_name} for Deberta models")


class TestPeftFeatureExtractionModel(PeftCommonTester):
    """
    Test if the PeftModel behaves as expected. This includes:
    - test if the model has the expected methods
    """

    transformers_class = AutoModel

    def skipTest(self, reason=""):
        # for backwards compatibility with unittest style test classes
        pytest.skip(reason)

    def prepare_inputs_for_testing(self):
        input_ids = torch.tensor([[1, 1, 1], [1, 2, 1]]).to(self.torch_device)
        attention_mask = torch.tensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)

        input_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        return input_dict

    @pytest.mark.parametrize("model_id", PEFT_FEATURE_EXTRACTION_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_attributes_parametrized(self, model_id, config_cls, config_kwargs):
        self._test_model_attr(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_FEATURE_EXTRACTION_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_adapter_name(self, model_id, config_cls, config_kwargs):
        self._test_adapter_name(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_FEATURE_EXTRACTION_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_prepare_for_training_parametrized(self, model_id, config_cls, config_kwargs):
        self._test_prepare_for_training(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_FEATURE_EXTRACTION_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained(self, model_id, config_cls, config_kwargs):
        self._test_save_pretrained(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_FEATURE_EXTRACTION_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained_selected_adapters(self, model_id, config_cls, config_kwargs):
        self._test_save_pretrained_selected_adapters(model_id, config_cls, config_kwargs)

    def test_load_model_low_cpu_mem_usage(self):
        self._test_load_model_low_cpu_mem_usage(PEFT_FEATURE_EXTRACTION_MODELS_TO_TEST[0], LoraConfig, {})

    @pytest.mark.parametrize("model_id", PEFT_FEATURE_EXTRACTION_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_from_pretrained_config_construction(self, model_id, config_cls, config_kwargs):
        self._test_from_pretrained_config_construction(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_FEATURE_EXTRACTION_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_merge_layers(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_merge_layers(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_FEATURE_EXTRACTION_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_training(self, model_id, config_cls, config_kwargs):
        self._test_training(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_FEATURE_EXTRACTION_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_training_prompt_learning_tasks(self, model_id, config_cls, config_kwargs):
        skip_deberta_pt_tests(config_cls, model_id)
        self._test_training_prompt_learning_tasks(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_FEATURE_EXTRACTION_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_training_layer_indexing(self, model_id, config_cls, config_kwargs):
        self._test_training_layer_indexing(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_FEATURE_EXTRACTION_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_training_gradient_checkpointing(self, model_id, config_cls, config_kwargs):
        skip_deberta_lora_tests(config_cls, model_id)
        self._test_training_gradient_checkpointing(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_FEATURE_EXTRACTION_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_inference_safetensors(self, model_id, config_cls, config_kwargs):
        self._test_inference_safetensors(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_FEATURE_EXTRACTION_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_peft_model_device_map(self, model_id, config_cls, config_kwargs):
        self._test_peft_model_device_map(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_FEATURE_EXTRACTION_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_delete_adapter(self, model_id, config_cls, config_kwargs):
        self._test_delete_adapter(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_FEATURE_EXTRACTION_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_delete_inactive_adapter(self, model_id, config_cls, config_kwargs):
        self._test_delete_inactive_adapter(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_FEATURE_EXTRACTION_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_unload_adapter(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_unload_adapter(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_FEATURE_EXTRACTION_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_weighted_combination_of_adapters(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_weighted_combination_of_adapters(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_FEATURE_EXTRACTION_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_passing_input_embeds_works(self, model_id, config_cls, config_kwargs):
        skip_non_prompt_learning(config_cls)
        self._test_passing_input_embeds_works("test input embeds work", model_id, config_cls, config_kwargs)
