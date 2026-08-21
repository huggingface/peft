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
    AdamssConfig,
    BeftConfig,
    BOFTConfig,
    C3AConfig,
    DeftConfig,
    DeloraConfig,
    FourierFTConfig,
    FrodConfig,
    GloraConfig,
    GraloraConfig,
    HiraConfig,
    HRAConfig,
    IA3Config,
    LilyConfig,
    LoraConfig,
    MissConfig,
    OFTConfig,
    PeanutConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    PromptTuningInit,
    PsoftConfig,
    RandLoraConfig,
    RoadConfig,
    ShadowConfig,
    ShiraConfig,
    SupertuningConfig,
    TinyLoraConfig,
    VBLoRAConfig,
    VeraConfig,
    WaveFTConfig,
    get_peft_model,
)
from peft.utils.other import ModulesToSaveWrapper

from .testing_common import PeftCommonTester
from .testing_utils import hub_online_once, set_init_weights_false


# Note: models from peft-internal-testing are just the safetensors versions of hf-internal-testing
PEFT_SEQ_CLS_MODELS_TO_TEST = [
    "peft-internal-testing/tiny-random-BertForSequenceClassification",
    "peft-internal-testing/tiny-random-RobertaForSequenceClassification",
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
        BeftConfig,
        {
            "task_type": "SEQ_CLS",
            "target_modules": None,
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
        MissConfig,
        {
            "task_type": "SEQ_CLS",
            "target_modules": None,
            "r": 2,
        },
    ),
    (
        DeftConfig,
        {
            "task_type": "SEQ_CLS",
            "target_modules": None,
        },
    ),
    (
        DeloraConfig,
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
        FrodConfig,
        {
            "task_type": "SEQ_CLS",
            "target_modules": None,
            "sparse_rate": 0.01,
        },
    ),
    (
        GloraConfig,
        {
            "task_type": "SEQ_CLS",
            "target_modules": None,
        },
    ),
    (
        GraloraConfig,
        {
            "task_type": "SEQ_CLS",
            "target_modules": None,
        },
    ),
    (
        HiraConfig,
        {
            "task_type": "SEQ_CLS",
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
        LilyConfig,
        {
            "task_type": "SEQ_CLS",
            "target_modules": None,
            "r": 8,
            "stride_A": 1,
            "num_B": 2,
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
        PsoftConfig,
        {
            "task_type": "SEQ_CLS",
            "r": 16,  # tiny llama has hidden size 16, so don't choose a greater value
            "psoft_alpha": 16,
            "target_modules": None,
        },
    ),
    (
        PeanutConfig,
        {
            "r": 8,
            "depth": 1,
            "act_fn": "relu",
            "task_type": "SEQ_CLS",
            "target_modules": None,
        },
    ),
    (
        RandLoraConfig,
        {
            "task_type": "SEQ_CLS",
            "target_modules": None,
            "r": 8,
            "randlora_alpha": 1,
        },
    ),
    (
        RoadConfig,
        {
            "task_type": "SEQ_CLS",
            "variant": "road_1",
            "group_size": 2,
        },
    ),
    (
        ShiraConfig,
        {
            "r": 1,
            "task_type": "SEQ_CLS",
            "target_modules": None,
            "init_weights": False,
        },
    ),
    (
        ShadowConfig,
        {
            "task_type": "SEQ_CLS",
            "r": 2,
            "shadow_num_hidden_layers": 1,
        },
    ),
    (
        SupertuningConfig,
        {
            "sparsity": 0.9,
            "task_type": "SEQ_CLS",
            "target_modules": None,
            "init_weights": False,
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
        TinyLoraConfig,
        {
            "task_type": "SEQ_CLS",
            "target_modules": None,
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
    (
        WaveFTConfig,
        {
            "task_type": "SEQ_CLS",
            "n_frequency": 8,
            "target_modules": None,
        },
    ),
    (
        AdamssConfig,
        {
            "task_type": "SEQ_CLS",
            "target_modules": None,
            "r": 8,
        },
    ),
]


def _skip_encoder_models(model_id, config_cls):
    # ShadowPEFT rides a contiguous decoder stack; encoder-only classifiers (BERT/RoBERTa) are unsupported.
    if config_cls is ShadowConfig and ("Bert" in model_id or "Roberta" in model_id):
        pytest.skip("ShadowPEFT requires a decoder-only backbone")


class TestSequenceClassificationModels(PeftCommonTester):
    r"""
    Tests for basic coverage of AutoModelForSequenceClassification and classification-specific cases. Most of the
    functionality is probably already covered by other tests.
    """

    transformers_class = AutoModelForSequenceClassification

    def prepare_inputs_for_testing(self):
        input_ids = torch.tensor([[1, 1, 1], [1, 2, 1]]).to(self.torch_device)
        attention_mask = torch.tensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @pytest.mark.parametrize("model_id", PEFT_SEQ_CLS_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_attributes_parametrized(self, model_id, config_cls, config_kwargs):
        _skip_encoder_models(model_id, config_cls)
        self._test_model_attr(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_SEQ_CLS_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_adapter_name(self, model_id, config_cls, config_kwargs):
        _skip_encoder_models(model_id, config_cls)
        self._test_adapter_name(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_SEQ_CLS_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_prepare_for_training_parametrized(self, model_id, config_cls, config_kwargs):
        _skip_encoder_models(model_id, config_cls)
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
        _skip_encoder_models(model_id, config_cls)
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_save_pretrained(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_SEQ_CLS_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained_pickle(self, model_id, config_cls, config_kwargs):
        _skip_encoder_models(model_id, config_cls)
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_save_pretrained(model_id, config_cls, config_kwargs.copy(), safe_serialization=False)

    @pytest.mark.parametrize("model_id", PEFT_SEQ_CLS_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained_selected_adapters(self, model_id, config_cls, config_kwargs):
        _skip_encoder_models(model_id, config_cls)
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_save_pretrained_selected_adapters(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_SEQ_CLS_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained_selected_adapters_pickle(self, model_id, config_cls, config_kwargs):
        _skip_encoder_models(model_id, config_cls)
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_save_pretrained_selected_adapters(
            model_id, config_cls, config_kwargs.copy(), safe_serialization=False
        )

    @pytest.mark.parametrize("model_id", PEFT_SEQ_CLS_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_from_pretrained_config_construction(self, model_id, config_cls, config_kwargs):
        _skip_encoder_models(model_id, config_cls)
        self._test_from_pretrained_config_construction(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_SEQ_CLS_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_modules_to_save_correctly_set(self, model_id, config_cls, config_kwargs):
        # tests for a regression, introduced via #2220, where modules_to_save was not applied to prompt learning methods
        _skip_encoder_models(model_id, config_cls)
        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = get_peft_model(model, config)
            base_model = model.get_base_model()
            # classifier layer is called either "classifier" or "score"
            classifier = getattr(base_model, "classifier", getattr(base_model, "score", None))
            if classifier is None:
                raise ValueError(f"Could not determine classifier layer name for {model_id}, please fix the test")
            assert isinstance(classifier, ModulesToSaveWrapper)

    @pytest.mark.parametrize("model_id", PEFT_SEQ_CLS_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_forward_with_labels(self, model_id, config_cls, config_kwargs):
        # Check the full forward pass including the loss computation. This is especially relevant for prompt learning
        # methods, whose sequence classification forward (including the _prefix_tuning_forward fallback for models whose
        # forward does not accept past_key_values) is implemented in PeftModelForSequenceClassification itself.
        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)

            if getattr(model.config, "pad_token_id", None) is None:
                # needed for a batched forward pass with sequence classification models like Llama
                model.config.pad_token_id = 0

            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = get_peft_model(model, config).to(self.torch_device)
            model.eval()

            inputs = self.prepare_inputs_for_testing()
            num_labels = model.config.num_labels
            if num_labels == 1:
                # a single label means that transformers infers regression as the problem type and uses an MSE loss on
                # float labels; this is the case for the tiny Llama model, whose head has a single output
                labels = torch.tensor([0.5, -0.5]).to(self.torch_device)
            else:
                labels = torch.tensor([0, num_labels - 1]).to(self.torch_device)

            with torch.no_grad():
                output = model(**inputs, labels=labels)

            assert output.loss is not None
            assert torch.isfinite(output.loss)
            assert output.logits.shape == (2, num_labels)

            if num_labels == 1:
                expected_loss = torch.nn.functional.mse_loss(output.logits.squeeze().float(), labels)
            else:
                # int labels and num_labels > 1 result in single label classification, i.e. plain cross entropy
                expected_loss = torch.nn.functional.cross_entropy(output.logits.float(), labels)
            # ensure same dtype for allclose call
            expected_loss = expected_loss.to(dtype=output.loss.dtype)

            if config_cls == AdaLoraConfig:
                # AdaLora adds an orthogonal regularization term to the loss, so it does not equal the plain task loss
                assert output.loss > expected_loss
            else:
                assert torch.allclose(output.loss, expected_loss, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize(
        "config_cls,config_kwargs",
        [
            (PrefixTuningConfig, {"task_type": "SEQ_CLS", "num_virtual_tokens": 4}),
            (PromptEncoderConfig, {"task_type": "SEQ_CLS", "num_virtual_tokens": 4, "encoder_hidden_size": 32}),
            (PromptTuningConfig, {"task_type": "SEQ_CLS", "num_virtual_tokens": 4}),
        ],
    )
    def test_prompt_learning_forward_with_inputs_embeds(self, config_cls, config_kwargs):
        # Passing inputs_embeds instead of input_ids should be equivalent.
        model_id = PEFT_SEQ_CLS_MODELS_TO_TEST[0]
        with hub_online_once(model_id):
            base_model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.torch_device)
            model = get_peft_model(base_model, config_cls(base_model_name_or_path=model_id, **config_kwargs))
            model.eval()

            input_ids = torch.tensor([[1, 1, 1], [1, 2, 1]]).to(self.torch_device)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                output_ids = model(input_ids=input_ids, attention_mask=attention_mask)
                inputs_embeds = model.get_input_embeddings()(input_ids)
                output_embeds = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            assert torch.allclose(output_ids.logits, output_embeds.logits, atol=1e-5, rtol=1e-5)
