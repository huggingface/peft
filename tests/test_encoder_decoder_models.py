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

import pytest
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForTokenClassification

from peft import (
    AdaLoraConfig,
    BOFTConfig,
    BoneConfig,
    C3AConfig,
    DeloraConfig,
    FourierFTConfig,
    GraloraConfig,
    HRAConfig,
    IA3Config,
    LoraConfig,
    MissConfig,
    OFTConfig,
    OSFConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    RoadConfig,
    ShiraConfig,
    TaskType,
    VBLoRAConfig,
    VeraConfig,
    WaveFTConfig,
    get_peft_model,
)

from .testing_common import PeftCommonTester
from .testing_utils import set_init_weights_false


# Note: models from peft-internal-testing are just the safetensors versions of hf-internal-testing
PEFT_ENCODER_DECODER_MODELS_TO_TEST = [
    "peft-internal-testing/tiny-random-T5ForConditionalGeneration-calibrated",
    "peft-internal-testing/tiny-random-BartForConditionalGeneration",
]

# TODO Missing from this list are LoKr, LoHa, LN Tuning, add them
ALL_CONFIGS = [
    (
        AdaLoraConfig,
        {
            "target_modules": None,
            "total_step": 1,
            "task_type": "SEQ_2_SEQ_LM",
        },
    ),
    (
        BOFTConfig,
        {
            "target_modules": None,
            "task_type": "SEQ_2_SEQ_LM",
        },
    ),
    (
        BoneConfig,
        {
            "target_modules": None,
            "r": 2,
            "task_type": "SEQ_2_SEQ_LM",
        },
    ),
    (
        MissConfig,
        {
            "target_modules": None,
            "r": 2,
            "task_type": "SEQ_2_SEQ_LM",
        },
    ),
    (
        DeloraConfig,
        {
            "task_type": "SEQ_2_SEQ_LM",
            "target_modules": None,
            "r": 2,
        },
    ),
    (
        FourierFTConfig,
        {
            "n_frequency": 10,
            "target_modules": None,
            "task_type": "SEQ_2_SEQ_LM",
        },
    ),
    (
        GraloraConfig,
        {
            "target_modules": None,
            "task_type": "SEQ_2_SEQ_LM",
        },
    ),
    (
        HRAConfig,
        {
            "target_modules": None,
            "task_type": "SEQ_2_SEQ_LM",
        },
    ),
    (
        IA3Config,
        {
            "target_modules": None,
            "feedforward_modules": None,
            "task_type": "SEQ_2_SEQ_LM",
        },
    ),
    (
        LoraConfig,
        {
            "r": 8,
            "lora_alpha": 32,
            "target_modules": None,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "SEQ_2_SEQ_LM",
        },
    ),
    (
        LoraConfig,
        {
            "r": 8,
            "lora_alpha": 32,
            "target_modules": None,
            "lora_dropout": 0.05,
            "bias": "none",
            "trainable_token_indices": [0, 1, 3],
            "task_type": "SEQ_2_SEQ_LM",
        },
    ),
    (
        OFTConfig,
        {
            "target_modules": None,
            "task_type": "SEQ_2_SEQ_LM",
        },
    ),
    (
        PrefixTuningConfig,
        {
            "num_virtual_tokens": 10,
            "task_type": "SEQ_2_SEQ_LM",
        },
    ),
    (
        PromptEncoderConfig,
        {
            "num_virtual_tokens": 10,
            "encoder_hidden_size": 32,
            "task_type": "SEQ_2_SEQ_LM",
        },
    ),
    (
        PromptTuningConfig,
        {
            "num_virtual_tokens": 10,
            "task_type": "SEQ_2_SEQ_LM",
        },
    ),
    (
        RoadConfig,
        {
            "task_type": "SEQ_2_SEQ_LM",
            "variant": "road_1",
            "group_size": 2,
        },
    ),
    (
        ShiraConfig,
        {
            "r": 1,
            "task_type": "SEQ_2_SEQ_LM",
            "target_modules": None,
            "init_weights": False,
        },
    ),
    (
        VBLoRAConfig,
        {
            "target_modules": None,
            "vblora_dropout": 0.05,
            "vector_length": 1,
            "num_vectors": 2,
            "task_type": "SEQ_2_SEQ_LM",
        },
    ),
    (
        VeraConfig,
        {
            "r": 8,
            "target_modules": None,
            "vera_dropout": 0.05,
            "projection_prng_key": 0xFF,
            "d_initial": 0.1,
            "save_projection": True,
            "bias": "none",
            "task_type": "SEQ_2_SEQ_LM",
        },
    ),
    (
        C3AConfig,
        {
            "task_type": "SEQ_2_SEQ_LM",
            "block_size": 1,
            "target_modules": None,
        },
    ),
    (
        WaveFTConfig,
        {
            "task_type": "SEQ_2_SEQ_LM",
            "n_frequency": 8,
            "target_modules": None,
        },
    ),
    (
        OSFConfig,
        {
            "task_type": "SEQ_2_SEQ_LM",
        },
    ),
]


def _skip_osf_disable_adapter_test(config_cls):
    if config_cls is OSFConfig:
        pytest.skip(
            "Skipping OSF for disable_adapter test because OSF uses exact SVD decomposition, so outputs are identical until training."
        )


class TestEncoderDecoderModels(PeftCommonTester):
    transformers_class = AutoModelForSeq2SeqLM

    def prepare_inputs_for_testing(self):
        input_ids = torch.tensor([[1, 1, 1], [1, 2, 1]]).to(self.torch_device)
        decoder_input_ids = torch.tensor([[1, 1, 1], [1, 2, 1]]).to(self.torch_device)
        attention_mask = torch.tensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)

        input_dict = {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
        }

        return input_dict

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_attributes_parametrized(self, model_id, config_cls, config_kwargs):
        self._test_model_attr(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_adapter_name(self, model_id, config_cls, config_kwargs):
        self._test_adapter_name(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_prepare_for_training_parametrized(self, model_id, config_cls, config_kwargs):
        self._test_prepare_for_training(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained(self, model_id, config_cls, config_kwargs):
        self._test_save_pretrained(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained_pickle(self, model_id, config_cls, config_kwargs):
        self._test_save_pretrained(model_id, config_cls, config_kwargs, safe_serialization=False)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained_selected_adapters(self, model_id, config_cls, config_kwargs):
        self._test_save_pretrained_selected_adapters(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained_selected_adapters_pickle(self, model_id, config_cls, config_kwargs):
        self._test_save_pretrained_selected_adapters(model_id, config_cls, config_kwargs, safe_serialization=False)

    def test_load_model_low_cpu_mem_usage(self):
        # Using the first model with LoraConfig and an empty config_kwargs.
        self._test_load_model_low_cpu_mem_usage(PEFT_ENCODER_DECODER_MODELS_TO_TEST[0], LoraConfig, {})

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_from_pretrained_config_construction(self, model_id, config_cls, config_kwargs):
        self._test_from_pretrained_config_construction(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_merge_layers(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_merge_layers(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_mixed_adapter_batches(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_mixed_adapter_batches(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_generate_with_mixed_adapter_batches(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_generate_with_mixed_adapter_batches_and_beam_search(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_generate(self, model_id, config_cls, config_kwargs):
        self._test_generate(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_generate_pos_args(self, model_id, config_cls, config_kwargs):
        self._test_generate_pos_args(model_id, config_cls, config_kwargs, raises_err=True)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_generate_half_prec(self, model_id, config_cls, config_kwargs):
        self._test_generate_half_prec(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_prefix_tuning_half_prec_conversion(self, model_id, config_cls, config_kwargs):
        self._test_prefix_tuning_half_prec_conversion(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_training_encoder_decoders(self, model_id, config_cls, config_kwargs):
        self._test_training(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_training_encoder_decoders_layer_indexing(self, model_id, config_cls, config_kwargs):
        self._test_training_layer_indexing(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    @pytest.mark.parametrize("use_reentrant", [True, False])
    def test_training_encoder_decoders_gradient_checkpointing(
        self, model_id, config_cls, config_kwargs, use_reentrant
    ):
        self._test_training_gradient_checkpointing(model_id, config_cls, config_kwargs, use_reentrant=use_reentrant)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_inference_safetensors(self, model_id, config_cls, config_kwargs):
        self._test_inference_safetensors(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_peft_model_device_map(self, model_id, config_cls, config_kwargs):
        self._test_peft_model_device_map(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_delete_adapter(self, model_id, config_cls, config_kwargs):
        self._test_delete_adapter(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_delete_inactive_adapter(self, model_id, config_cls, config_kwargs):
        self._test_delete_inactive_adapter(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_adding_multiple_adapters_with_bias_raises(self, model_id, config_cls, config_kwargs):
        self._test_adding_multiple_adapters_with_bias_raises(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_unload_adapter(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_unload_adapter(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_weighted_combination_of_adapters(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_weighted_combination_of_adapters(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_training_prompt_learning_tasks(self, model_id, config_cls, config_kwargs):
        self._test_training_prompt_learning_tasks(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_ENCODER_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_disable_adapter(self, model_id, config_cls, config_kwargs):
        _skip_osf_disable_adapter_test(config_cls)
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_disable_adapter(model_id, config_cls, config_kwargs)

    def test_active_adapters_prompt_learning(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "peft-internal-testing/tiny-random-BartForConditionalGeneration"
        ).to(self.torch_device)
        # any prompt learning method would work here
        config = PromptEncoderConfig(task_type=TaskType.SEQ_2_SEQ_LM, num_virtual_tokens=10)
        model = get_peft_model(model, config)
        assert model.active_adapters == ["default"]

    def test_save_shared_tensors(self):
        model_id = "peft-internal-testing/tiny-random-RobertaModel"
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="all",
        )
        model = AutoModelForTokenClassification.from_pretrained(model_id, num_labels=11)
        model = get_peft_model(model, peft_config)
        with tempfile.TemporaryDirectory() as tmp_dir:
            # This should work fine
            model.save_pretrained(tmp_dir, safe_serialization=True)
