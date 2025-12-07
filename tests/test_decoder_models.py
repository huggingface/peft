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
import json
import platform
import tempfile
from unittest.mock import Mock, call, patch

import pytest
import torch
from accelerate.test_utils.testing import get_backend
from safetensors.torch import load_file as safe_load_file
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from peft import (
    AdaLoraConfig,
    BOFTConfig,
    BoneConfig,
    C3AConfig,
    CPTConfig,
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
    PromptEmbedding,
    PromptEncoderConfig,
    PromptTuningConfig,
    PromptTuningInit,
    RoadConfig,
    ShiraConfig,
    TaskType,
    VBLoRAConfig,
    VeraConfig,
    WaveFTConfig,
    get_peft_model,
)

from .testing_common import PeftCommonTester
from .testing_utils import device_count, hub_online_once, load_dataset_english_quotes, set_init_weights_false


# Note: some models from peft-internal-testing are just the safetensors versions of hf-internal-testing
PEFT_DECODER_MODELS_TO_TEST = [
    "peft-internal-testing/tiny-random-OPTForCausalLM",
    "peft-internal-testing/tiny-random-GPT2LMHeadModel",
    "peft-internal-testing/tiny-random-GPTJForCausalLM",
    "trl-internal-testing/tiny-random-LlamaForCausalLM",
    "peft-internal-testing/tiny-dummy-qwen2",
    "hf-internal-testing/tiny-random-Gemma3ForCausalLM",
]

SMALL_GRID_MODELS = [
    "hf-internal-testing/tiny-random-gpt2",
    "peft-internal-testing/tiny-random-OPTForCausalLM",
    "hf-internal-testing/tiny-random-MistralForCausalLM",
    "peft-internal-testing/tiny-dummy-qwen2",
    "trl-internal-testing/tiny-random-LlamaForCausalLM",
]


# TODO Missing from this list are LoKr, LoHa, LN Tuning, add them
# Note: If the PEFT method offers an initialization option to make it an identity transform (typically via the
# init_weights argument), then this option should be set here, if it's not already the default.
ALL_CONFIGS = [
    (
        AdaLoraConfig,
        {
            "task_type": "CAUSAL_LM",
            "target_modules": None,
            "total_step": 1,
        },
    ),
    (
        BOFTConfig,
        {
            "task_type": "CAUSAL_LM",
            "target_modules": None,
        },
    ),
    (
        BoneConfig,
        {
            "task_type": "CAUSAL_LM",
            "target_modules": None,
            "r": 2,
        },
    ),
    (
        MissConfig,
        {
            "task_type": "CAUSAL_LM",
            "target_modules": None,
            "r": 2,
        },
    ),
    (
        CPTConfig,
        {
            "task_type": "CAUSAL_LM",
            "cpt_token_ids": [0, 1, 2, 3, 4, 5, 6, 7],  # Example token IDs for testing
            "cpt_mask": [1, 1, 1, 1, 1, 1, 1, 1],
            "cpt_tokens_type_mask": [1, 2, 2, 2, 3, 3, 4, 4],
        },
    ),
    (
        DeloraConfig,
        {
            "task_type": "CAUSAL_LM",
            "target_modules": None,
            "r": 2,
        },
    ),
    (
        FourierFTConfig,
        {
            "task_type": "CAUSAL_LM",
            "n_frequency": 10,
            "target_modules": None,
        },
    ),
    (
        GraloraConfig,
        {
            "task_type": "CAUSAL_LM",
            "r": 8,
            "alpha": 16,
            "target_modules": None,
            "gralora_dropout": 0.05,
            "gralora_k": 2,
            "hybrid_r": 0,
        },
    ),
    (
        GraloraConfig,
        {
            "task_type": "CAUSAL_LM",
            "r": 16,
            "alpha": 32,
            "target_modules": None,
            "gralora_dropout": 0.05,
            "gralora_k": 4,
            "hybrid_r": 4,
        },
    ),
    (
        HRAConfig,
        {
            "task_type": "CAUSAL_LM",
            "target_modules": None,
        },
    ),
    (
        IA3Config,
        {
            "task_type": "CAUSAL_LM",
            "target_modules": None,
            "feedforward_modules": None,
        },
    ),
    (
        LoraConfig,
        {
            "task_type": "CAUSAL_LM",
            "r": 8,
            "lora_alpha": 32,
            "target_modules": None,
            "lora_dropout": 0.05,
            "bias": "none",
        },
    ),
    # Activated LoRA (aLoRA)
    (
        LoraConfig,
        {
            "task_type": "CAUSAL_LM",
            "r": 8,
            "lora_alpha": 32,
            "target_modules": None,
            "lora_dropout": 0.05,
            "bias": "none",
            "alora_invocation_tokens": [1],
        },
    ),
    (
        LoraConfig,
        {
            "task_type": "CAUSAL_LM",
            "r": 8,
            "lora_alpha": 32,
            "target_modules": None,
            "lora_dropout": 0.05,
            "bias": "none",
            # not one test input sequence will ever have this token, this should do nothing at all
            "alora_invocation_tokens": [1000],
        },
    ),
    # LoRA + trainable tokens
    (
        LoraConfig,
        {
            "task_type": "CAUSAL_LM",
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
            "task_type": "CAUSAL_LM",
            "target_modules": None,
        },
    ),
    (
        PrefixTuningConfig,
        {
            "task_type": "CAUSAL_LM",
            "num_virtual_tokens": 10,
        },
    ),
    (
        PromptEncoderConfig,
        {
            "task_type": "CAUSAL_LM",
            "num_virtual_tokens": 10,
            "encoder_hidden_size": 32,
        },
    ),
    (
        PromptTuningConfig,
        {
            "task_type": "CAUSAL_LM",
            "num_virtual_tokens": 10,
        },
    ),
    (
        RoadConfig,
        {
            "task_type": "CAUSAL_LM",
            "variant": "road_1",
            "group_size": 2,
        },
    ),
    (
        ShiraConfig,
        {
            "r": 1,
            "task_type": "CAUSAL_LM",
            "target_modules": None,
            "init_weights": False,
        },
    ),
    (
        VBLoRAConfig,
        {
            "task_type": "CAUSAL_LM",
            "target_modules": None,
            "vblora_dropout": 0.05,
            "vector_length": 1,
            "num_vectors": 2,
        },
    ),
    (
        VeraConfig,
        {
            "task_type": "CAUSAL_LM",
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
            "task_type": "CAUSAL_LM",
            "block_size": 1,  # Some test cases contain shapes of prime numbers where `block_size` must be 1
            "target_modules": None,
        },
    ),
    (
        WaveFTConfig,
        {
            "task_type": "CAUSAL_LM",
            "n_frequency": 8,
            "target_modules": None,
        },
    ),
    (
        OSFConfig,
        {
            "task_type": "CAUSAL_LM",
        },
    ),
]


def _skip_if_not_conv1d_supported(model_id, config_cls):
    if "GPT2LMHeadModel" in model_id and config_cls in [
        BOFTConfig,
        BoneConfig,
        HRAConfig,
        OFTConfig,
        OSFConfig,
        RoadConfig,
        ShiraConfig,
        C3AConfig,
        MissConfig,
        DeloraConfig,
    ]:
        pytest.skip("Skipping BOFT/HRA/OFT/Bone/Road/SHiRA/C3A/MiSS/OSF/DeLoRA for GPT2LMHeadModel")


def _skip_adalora_oft_hra_bone_for_gpt2(model_id, config_cls):
    if "GPT2LMHeadModel" in model_id and config_cls in [
        AdaLoraConfig,
        BOFTConfig,
        HRAConfig,
        OFTConfig,
        BoneConfig,
        C3AConfig,
        RoadConfig,
        MissConfig,
        DeloraConfig,
    ]:
        pytest.skip("Skipping AdaLora/BOFT/HRA/OFT/Bone/MiSS/DeLoRA for GPT2LMHeadModel")


def _skip_alora_no_activation(config_cls, config_kwargs):
    if config_cls is LoraConfig and config_kwargs.get("alora_invocation_tokens") == [1000]:
        pytest.skip("Skipping aLoRA no-activation-case because the test expects changed output which there won't be.")


def _skip_osf_disable_adapter_test(config_cls):
    if config_cls is OSFConfig:
        pytest.skip(
            "Skipping OSF for disable_adapter test because OSF uses exact SVD decomposition, so outputs are identical until training."
        )


class TestDecoderModels(PeftCommonTester):
    transformers_class = AutoModelForCausalLM

    def prepare_inputs_for_testing(self):
        input_ids = torch.tensor([[1, 1, 1], [1, 2, 1]]).to(self.torch_device)
        attention_mask = torch.tensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_attributes_parametrized(self, model_id, config_cls, config_kwargs):
        _skip_if_not_conv1d_supported(model_id, config_cls)
        self._test_model_attr(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_adapter_name(self, model_id, config_cls, config_kwargs):
        _skip_if_not_conv1d_supported(model_id, config_cls)
        self._test_adapter_name(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_prepare_for_training_parametrized(self, model_id, config_cls, config_kwargs):
        _skip_if_not_conv1d_supported(model_id, config_cls)
        self._test_prepare_for_training(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_prompt_tuning_text_prepare_for_training(self, model_id, config_cls, config_kwargs):
        if config_cls != PromptTuningConfig:
            pytest.skip(f"This test does not apply to {config_cls}")
        config_kwargs = config_kwargs.copy()
        config_kwargs["prompt_tuning_init"] = PromptTuningInit.TEXT
        config_kwargs["prompt_tuning_init_text"] = "This is a test prompt."
        config_kwargs["tokenizer_name_or_path"] = model_id
        self._test_prepare_for_training(model_id, config_cls, config_kwargs.copy())

    def test_prompt_tuning_text_tokenizer_kwargs(self):
        # Allow users to pass additional arguments to Tokenizer.from_pretrained
        # Fix for #1032
        mock = Mock()
        orig_from_pretrained = AutoTokenizer.from_pretrained

        def mock_autotokenizer_from_pretrained(*args, **kwargs):
            mock(*args, **kwargs)
            return orig_from_pretrained(config.tokenizer_name_or_path)

        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        config = PromptTuningConfig(
            base_model_name_or_path=model_id,
            tokenizer_name_or_path=model_id,
            num_virtual_tokens=10,
            prompt_tuning_init=PromptTuningInit.TEXT,
            task_type="CAUSAL_LM",
            prompt_tuning_init_text="This is a test prompt.",
            tokenizer_kwargs={"cache_dir": "/tmp/somewhere", "foo": "bar"},
        )
        model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
        with patch("transformers.AutoTokenizer.from_pretrained", mock_autotokenizer_from_pretrained):
            _ = get_peft_model(model, config)
        expected_call = call(model_id, cache_dir="/tmp/somewhere", foo="bar")
        assert mock.call_args == expected_call

    def test_prompt_tuning_trust_remote_code(self, tmp_path, monkeypatch):
        # See #2888 for details

        # This is a test for a hypothetical exploit that would enable trust_remote_code (and thus RCE) when a user loads
        # a malicious prompt tuning model. This is because PEFT would just pass the on the tokenizer_kwargs defined in
        # the prompt tuning config unsanitzed, which means that if the tokenizer is also malicious, the malicious code
        # would be executed. For this exploit to work, a user cannot load a model using PeftModel.from_pretrained as
        # normal, because the tokenizer is only loaded in training mode. Although the attacker could set
        # inference_mode=True in the adapter_config.json, that would still not work because prompt tuning methods cannot
        # be loaded in inference mode. Therefore, the only way for the exploit to work would be if the user manually
        # loads the model, as is shown below.

        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        with hub_online_once(model_id):
            # crafting the malicious checkpoint:
            model = AutoModelForCausalLM.from_pretrained(model_id)
            config = PromptTuningConfig(
                num_virtual_tokens=10,
                task_type=TaskType.CAUSAL_LM,
                tokenizer_name_or_path=model_id,
                prompt_tuning_init=PromptTuningInit.TEXT,
                prompt_tuning_init_text="hello",
                tokenizer_kwargs={"trust_remote_code": "foobar"},
            )
            model = get_peft_model(model, config)
            model.save_pretrained(tmp_path)

            with open(tmp_path / "adapter_config.json") as f:
                config_dict = json.load(f)
                # disable inference mode
                config_dict["inference_mode"] = False
            with open(tmp_path / "adapter_config.json", "w") as f:
                json.dump(config_dict, f)

            del model

            # applying a mock to check the used parameters
            used_args = []
            used_kwargs = {}

            orig_from_pretrained = AutoTokenizer.from_pretrained

            def fake_from_pretrained(*args, **kwargs):
                used_args.extend(args)
                used_kwargs.update(kwargs)
                return orig_from_pretrained(*args, **kwargs)

            monkeypatch.setattr(AutoTokenizer, "from_pretrained", fake_from_pretrained)

            # user code: loading the malicious checkpoint
            model = AutoModelForCausalLM.from_pretrained(model_id)
            config = PromptTuningConfig.from_pretrained(tmp_path)
            PromptEmbedding(config, model.model.decoder.embed_tokens)

            # check that neither args nor kwargs used trust_remote_code='foobar'
            assert "foobar" not in used_args
            assert used_kwargs.get("trust_remote_code") != "foobar"

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_prompt_tuning_sample_vocab_prepare_for_training(self, model_id, config_cls, config_kwargs):
        if config_cls != PromptTuningConfig:
            pytest.skip(f"This test does not apply to {config_cls}")

        config_kwargs = config_kwargs.copy()
        config_kwargs["prompt_tuning_init"] = PromptTuningInit.SAMPLE_VOCAB
        config_kwargs["tokenizer_name_or_path"] = model_id

        self._test_prepare_for_training(model_id, config_cls, config_kwargs.copy())

    def test_prompt_tuning_config_invalid_args(self):
        # Raise an error when tokenizer_kwargs is used with prompt_tuning_init!='TEXT', because this argument has no
        # function in that case
        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
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

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained(self, model_id, config_cls, config_kwargs):
        _skip_if_not_conv1d_supported(model_id, config_cls)
        self._test_save_pretrained(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained_pickle(self, model_id, config_cls, config_kwargs):
        _skip_if_not_conv1d_supported(model_id, config_cls)
        self._test_save_pretrained(model_id, config_cls, config_kwargs.copy(), safe_serialization=False)

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained_selected_adapters(self, model_id, config_cls, config_kwargs):
        _skip_if_not_conv1d_supported(model_id, config_cls)
        self._test_save_pretrained_selected_adapters(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained_selected_adapters_pickle(self, model_id, config_cls, config_kwargs):
        _skip_if_not_conv1d_supported(model_id, config_cls)
        self._test_save_pretrained_selected_adapters(
            model_id, config_cls, config_kwargs.copy(), safe_serialization=False
        )

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_from_pretrained_config_construction(self, model_id, config_cls, config_kwargs):
        _skip_if_not_conv1d_supported(model_id, config_cls)
        self._test_from_pretrained_config_construction(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_merge_layers(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_merge_layers(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_merge_layers_multi(self, model_id, config_cls, config_kwargs):
        _skip_if_not_conv1d_supported(model_id, config_cls)
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_merge_layers_multi(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_merge_layers_nan(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_merge_layers_nan(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_mixed_adapter_batches(self, model_id, config_cls, config_kwargs):
        if config_cls != LoraConfig:
            pytest.skip("Mixed adapter batches not supported for this config.")
        _skip_alora_no_activation(config_cls, config_kwargs)
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_mixed_adapter_batches(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_generate_with_mixed_adapter_batches(self, model_id, config_cls, config_kwargs):
        if config_cls != LoraConfig:
            pytest.skip("Mixed adapter batches not supported for this config.")
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_generate_with_mixed_adapter_batches_and_beam_search(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_generate(self, model_id, config_cls, config_kwargs):
        _skip_if_not_conv1d_supported(model_id, config_cls)
        self._test_generate(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_generate_pos_args(self, model_id, config_cls, config_kwargs):
        _skip_if_not_conv1d_supported(model_id, config_cls)
        self._test_generate_pos_args(model_id, config_cls, config_kwargs.copy(), raises_err=False)

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_merge_layers_fp16(self, model_id, config_cls, config_kwargs):
        self._test_merge_layers_fp16(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_generate_half_prec(self, model_id, config_cls, config_kwargs):
        self._test_generate_half_prec(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_prefix_tuning_half_prec_conversion(self, model_id, config_cls, config_kwargs):
        self._test_prefix_tuning_half_prec_conversion(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_training_decoders(self, model_id, config_cls, config_kwargs):
        _skip_if_not_conv1d_supported(model_id, config_cls)
        self._test_training(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_training_decoders_layer_indexing(self, model_id, config_cls, config_kwargs):
        self._test_training_layer_indexing(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    @pytest.mark.parametrize("use_reentrant", [True, False])
    def test_training_decoders_gradient_checkpointing(self, model_id, config_cls, config_kwargs, use_reentrant):
        _skip_if_not_conv1d_supported(model_id, config_cls)
        self._test_training_gradient_checkpointing(
            model_id, config_cls, config_kwargs.copy(), use_reentrant=use_reentrant
        )

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_inference_safetensors(self, model_id, config_cls, config_kwargs):
        _skip_if_not_conv1d_supported(model_id, config_cls)
        self._test_inference_safetensors(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_peft_model_device_map(self, model_id, config_cls, config_kwargs):
        self._test_peft_model_device_map(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_delete_adapter(self, model_id, config_cls, config_kwargs):
        _skip_if_not_conv1d_supported(model_id, config_cls)
        self._test_delete_adapter(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_delete_inactive_adapter(self, model_id, config_cls, config_kwargs):
        _skip_if_not_conv1d_supported(model_id, config_cls)
        self._test_delete_inactive_adapter(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_adding_multiple_adapters_with_bias_raises(self, model_id, config_cls, config_kwargs):
        _skip_if_not_conv1d_supported(model_id, config_cls)
        self._test_adding_multiple_adapters_with_bias_raises(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_unload_adapter(self, model_id, config_cls, config_kwargs):
        _skip_adalora_oft_hra_bone_for_gpt2(model_id, config_cls)
        _skip_if_not_conv1d_supported(model_id, config_cls)
        _skip_alora_no_activation(config_cls, config_kwargs)
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_unload_adapter(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_weighted_combination_of_adapters(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_weighted_combination_of_adapters(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_training_prompt_learning_tasks(self, model_id, config_cls, config_kwargs):
        self._test_training_prompt_learning_tasks(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_disable_adapter(self, model_id, config_cls, config_kwargs):
        _skip_if_not_conv1d_supported(model_id, config_cls)
        _skip_alora_no_activation(config_cls, config_kwargs)
        _skip_osf_disable_adapter_test(config_cls)
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_disable_adapter(model_id, config_cls, config_kwargs.copy())

    def test_generate_adalora_no_dropout(self):
        # test for issue #730
        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        config_kwargs = {
            "target_modules": None,
            "task_type": "CAUSAL_LM",
            "lora_dropout": 0.0,
            "total_step": 1,
        }
        self._test_generate(model_id, AdaLoraConfig, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_passing_input_embeds_works(self, model_id, config_cls, config_kwargs):
        _skip_if_not_conv1d_supported(model_id, config_cls)
        if (platform.system() == "Darwin") and (config_cls == PrefixTuningConfig):
            # the error is:
            # > RuntimeError: unsupported operation: more than one element of the written-to tensor refers to a single
            # > memory location. Please clone() the tensor before performing the operation.
            # in transformers sdpa_mask_older_torch. As we (currently) cannot upgrade PyTorch on MacOS GH runners, we're
            # stuck with this error.
            # TODO: remove if torch can be upgraded on MacOS or if MacOS CI is removed
            pytest.skip("Prefix tuning fails on MacOS in this case, not worth fixing")
        self._test_passing_input_embeds_works("", model_id, config_cls, config_kwargs.copy())

    def test_lora_layer_replication(self):
        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"
        config_kwargs = {
            "target_modules": ["down_proj", "up_proj"],
            "task_type": "CAUSAL_LM",
            "lora_dropout": 0.0,
            "layer_replication": [[0, 1], [0, 2], [1, 2]],
        }
        model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
        config = LoraConfig(base_model_name_or_path=model_id, **config_kwargs)

        assert len(model.model.layers), "Expected 2 layers in original model." == 2
        model = get_peft_model(model, config)
        layers = model.base_model.model.model.layers
        assert len(layers) == 4, "Expected 4 layers in adapted model."
        assert (
            layers[0].mlp.up_proj.base_layer.weight.data.storage().data_ptr()
            == layers[1].mlp.up_proj.base_layer.weight.data.storage().data_ptr()
            and layers[2].mlp.up_proj.base_layer.weight.data.storage().data_ptr()
            == layers[3].mlp.up_proj.base_layer.weight.data.storage().data_ptr()
        ), "Expected layers 0-1 and 2-3 to share weights"
        assert (
            layers[0].mlp.up_proj.base_layer.weight.data.storage().data_ptr()
            != layers[2].mlp.up_proj.base_layer.weight.data.storage().data_ptr()
        ), "Expected layers 0 and 2 to have different weights"
        assert (
            layers[0].mlp.up_proj.lora_A.default.weight.data.storage().data_ptr()
            != layers[1].mlp.up_proj.lora_A.default.weight.data.storage().data_ptr()
            and layers[2].mlp.up_proj.lora_A.default.weight.data.storage().data_ptr()
            != layers[3].mlp.up_proj.lora_A.default.weight.data.storage().data_ptr()
        ), "Expected all LoRA adapters to have distinct weights"
        assert len([n for n, _ in model.named_parameters() if ".lora_A." in n]) == 8, (
            "Expected 8 LoRA adapters since we are adding one each for up and down."
        )
        self._test_prepare_for_training(model_id, LoraConfig, config_kwargs.copy())
        self._test_generate(model_id, LoraConfig, config_kwargs.copy())

    def test_prefix_tuning_qwen2_with_grouped_query_attention(self):
        # See 1901, fixes a bug with handling GQA
        model_id = "peft-internal-testing/tiny-dummy-qwen2"
        with hub_online_once(model_id):
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            peft_config = PrefixTuningConfig(num_virtual_tokens=10, task_type="CAUSAL_LM")
            model = get_peft_model(base_model, peft_config)
            x = torch.tensor([[1, 2, 3]])
            # does not raise
            model(x)

    def test_prefix_tuning_qwen3_with_grouped_query_attention(self):
        # See 2881, fixes a bug with handling GQA
        model_id = "trl-internal-testing/tiny-Qwen3ForCausalLM"
        with hub_online_once(model_id):
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            peft_config = PrefixTuningConfig(num_virtual_tokens=10, task_type="CAUSAL_LM")
            model = get_peft_model(base_model, peft_config)
            x = torch.tensor([[1, 2, 3]])
            # does not raise
            model(x)

    def test_prefix_tuning_mistral(self):
        # See issue 869, 1962
        _, device_count, _ = get_backend()
        if device_count > 1:
            pytest.skip("PEFT Mistral training with DP does not work, skipping")

        model_id = "hf-internal-testing/tiny-random-MistralForCausalLM"
        base_model = AutoModelForCausalLM.from_pretrained(model_id)
        peft_config = PrefixTuningConfig(num_virtual_tokens=10, task_type="CAUSAL_LM")
        model = get_peft_model(base_model, peft_config)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        def process(samples):
            tokenized = tokenizer(samples["quote"], truncation=True, max_length=128)
            return tokenized

        data = load_dataset_english_quotes()
        data = data.map(process, batched=True)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            trainer = Trainer(
                model=model,
                train_dataset=data["train"],
                args=TrainingArguments(
                    num_train_epochs=1,
                    max_steps=5,
                    per_device_train_batch_size=4,
                    output_dir=tmp_dirname,
                ),
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
            trainer.train()

    @pytest.mark.parametrize("model_id", SMALL_GRID_MODELS)
    @pytest.mark.parametrize(
        "config_cls,config_kwargs",
        [
            (
                PromptTuningConfig,
                {
                    "num_virtual_tokens": 10,
                    "task_type": "CAUSAL_LM",
                },
            ),
            (
                PrefixTuningConfig,
                {
                    "num_virtual_tokens": 10,
                    "task_type": "CAUSAL_LM",
                },
            ),
            (
                PromptEncoderConfig,
                {
                    "num_virtual_tokens": 10,
                    "encoder_hidden_size": 32,
                    "task_type": "CAUSAL_LM",
                },
            ),
            (
                CPTConfig,
                {
                    "task_type": "CAUSAL_LM",
                    "cpt_token_ids": [0, 1, 2, 3, 4, 5, 6, 7],  # Example token IDs for testing
                    "cpt_mask": [1, 1, 1, 1, 1, 1, 1, 1],
                    "cpt_tokens_type_mask": [1, 2, 2, 2, 3, 3, 4, 4],
                },
            ),
        ],
    )
    def test_prompt_learning_with_gradient_checkpointing(self, model_id, config_cls, config_kwargs):
        # See issue 869
        # Test prompt learning methods with gradient checkpointing in a semi realistic setting.
        # Prefix tuning does not work if the model uses the new caching implementation. In that case, a helpful error
        # should be raised.

        # skip if multi GPU, since this results in DataParallel usage by Trainer, which fails with "CUDA device
        # assertion", breaking subsequent tests
        if device_count > 1:
            pytest.skip("Skip on multi-GPU setups")
        peft_config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        base_model = self.transformers_class.from_pretrained(model_id)
        base_model.gradient_checkpointing_enable()

        try:
            model = get_peft_model(base_model, peft_config)
        except ValueError as exc:
            # Some methods will raise a helpful error. After this, exit the test, as training would fail.
            assert config_cls == PrefixTuningConfig
            assert "Prefix tuning does not work with gradient checkpointing" in str(exc)
            return

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        def process(samples):
            tokenized = tokenizer(samples["quote"], truncation=True, max_length=128)
            return tokenized

        data = load_dataset_english_quotes()
        data = data.map(process, batched=True)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            trainer = Trainer(
                model=model,
                train_dataset=data["train"],
                args=TrainingArguments(
                    num_train_epochs=1,
                    max_steps=3,
                    per_device_train_batch_size=4,
                    output_dir=tmp_dirname,
                ),
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
            trainer.train()

    @pytest.mark.parametrize("save_embedding_layers", ["auto", True, False])
    @pytest.mark.parametrize(
        "peft_config",
        [
            (LoraConfig(target_modules=["lin0", "embed_tokens"], init_lora_weights=False)),
            (LoraConfig(target_modules=r".*\.embed_tokens", init_lora_weights=False)),
        ],
    )
    def test_save_pretrained_targeting_lora_to_embedding_layer(self, save_embedding_layers, tmp_path, peft_config):
        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"

        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id)
            model = get_peft_model(model, peft_config)

            if save_embedding_layers == "auto":
                # assert warning
                msg_start = "Setting `save_embedding_layers` to `True` as embedding layers found in `target_modules`."
                with pytest.warns(UserWarning, match=msg_start):
                    model.save_pretrained(tmp_path, save_embedding_layers=save_embedding_layers)
            else:
                model.save_pretrained(tmp_path, save_embedding_layers=save_embedding_layers)

            state_dict = safe_load_file(tmp_path / "adapter_model.safetensors")
            contains_embedding = "base_model.model.model.embed_tokens.base_layer.weight" in state_dict

            if save_embedding_layers in ["auto", True]:
                assert contains_embedding
                assert torch.allclose(
                    model.base_model.model.model.embed_tokens.base_layer.weight,
                    state_dict["base_model.model.model.embed_tokens.base_layer.weight"],
                )
            else:
                assert not contains_embedding

    @pytest.mark.parametrize("use_dora", [False, True])
    def test_lora_embed_scale_is_applied(self, use_dora):
        """Test that LoRA correctly handles embeddings with scaling (e.g., Gemma3)."""
        model_id = "hf-internal-testing/tiny-random-Gemma3ForCausalLM"
        with hub_online_once(model_id):
            base_model = AutoModelForCausalLM.from_pretrained(model_id).to(self.torch_device)
            orig_embedding = base_model.get_input_embeddings()

            peft_config = LoraConfig(target_modules=["embed_tokens"], init_lora_weights=False, use_dora=use_dora)
            peft_model = get_peft_model(base_model, peft_config)

            x = torch.arange(10).to(self.torch_device)
            peft_embedding = peft_model.base_model.model.get_input_embeddings()
            embedding_output = peft_embedding(x)
            max_embedding_output = embedding_output.abs().max(0)[0]
            assert (max_embedding_output < 100.0).all()
            peft_model.merge_adapter()
            embedding_merged = peft_embedding(x)
            assert torch.allclose(embedding_output, embedding_merged, atol=1e-5, rtol=1e-5)
            peft_model.unmerge_adapter()

            # set embed_scale to an absurdly high value, then check that the embedding output is also scaled to a high
            # value
            orig_embedding.embed_scale.fill_(10000.0)
            max_embedding_output = peft_embedding(x).abs().max(0)[0]
            assert (max_embedding_output > 100.0).all()

            # set embed_scale to zero, then check that the embedding output is also zero
            orig_embedding.embed_scale.fill_(0)
            embedding_output = peft_embedding(x)
            assert (embedding_output == 0.0).all()

    def test_lora_embed_scale_is_applied_mixed_batch(self):
        """Test that LoRA correctly handles embeddings with scaling in mixed batch mode."""
        model_id = "hf-internal-testing/tiny-random-Gemma3ForCausalLM"
        with hub_online_once(model_id):
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            orig_embedding = base_model.get_input_embeddings()

            peft_config = LoraConfig(target_modules=["embed_tokens"], init_lora_weights=False)
            peft_model = get_peft_model(base_model, peft_config)
            peft_model.add_adapter("adapter2", peft_config)

            # sanity check: with the default embed_scale, the embedding output should be reasonably sized
            peft_embedding = peft_model.base_model.model.get_input_embeddings()
            input_ids = torch.arange(10).unsqueeze(0).repeat(2, 1)
            adapter_names = ["default", "adapter2"]
            max_embedding_output = peft_embedding(input_ids, adapter_names=adapter_names).abs().max()
            assert max_embedding_output < 100.0

            # set embed_scale to an absurdly high value, then check that the embedding output is also scaled to a high
            # value
            orig_embedding.embed_scale.fill_(10000.0)
            max_embedding_output = peft_embedding(input_ids, adapter_names=adapter_names).abs().max()
            assert max_embedding_output > 100.0

            # set embed_scale to zero, then check that the embedding output is also zero
            orig_embedding.embed_scale.fill_(0)
            embedding_output = peft_embedding(input_ids, adapter_names=adapter_names)
            assert (embedding_output == 0.0).all()

    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_set_requires_grad_prompt_learning_raises(self, config_cls, config_kwargs):
        # Test that for prompt learning, calling set_requires_grad raises an error with an appropriate error message.
        # Note that for non-prompt learning methods, set_requires_grad is being tested for custom models, so there is no
        # specific test here.
        model_id = PEFT_DECODER_MODELS_TO_TEST[0]  # it's enough to test this with one model
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        if not config.is_prompt_learning:
            pytest.skip("This test is only for prompt learning methods.")

        with hub_online_once(model_id + config_kwargs.get("tokenizer_name_or_path", "")):
            model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
            model = get_peft_model(model, config)
            msg = "Setting `requires_grad` is not supported for prompt learning methods like"
            with pytest.raises(TypeError, match=msg):
                model.set_requires_grad(adapter_names="adpater0")
