# Copyright 2025-present the HuggingFace Inc. team.
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
from transformers import AutoModelForCausalLM

from peft import LoraConfig, get_peft_model

from .testing_common import PeftCommonTester
from .testing_utils import set_init_weights_false


PEFT_DECODER_MODELS_TO_TEST = [
    "trl-internal-testing/tiny-Llama4ForCausalLM",
]

# TODO Missing from this list are LoKr, LoHa, LN Tuning, add them
ALL_CONFIGS = [
    # target one layer
    (
        LoraConfig,
        {
            "task_type": "CAUSAL_LM",
            "r": 8,
            "lora_alpha": 32,
            "target_modules": None,
            "lora_dropout": 0.0,
            "bias": "none",
            "target_parameters": [
                "feed_forward.experts.down_proj",
            ],
        },
    ),
    # target two layers
    (
        LoraConfig,
        {
            "task_type": "CAUSAL_LM",
            "r": 8,
            "lora_alpha": 32,
            "target_modules": None,
            "lora_dropout": 0.0,
            "bias": "none",
            "target_parameters": [
                "feed_forward.experts.gate_up_proj",
                "feed_forward.experts.down_proj",
            ],
        },
    ),
]


class MyAutoModelForCausalLM(AutoModelForCausalLM):
    # TODO
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(*args, **kwargs)
        # model contains weights with values ~1e36 or nan
        with torch.no_grad():
            for param in model.parameters():
                param.data = torch.randn(param.shape)
        return model


class TestDecoderModelsTargetParameters(PeftCommonTester):
    transformers_class = MyAutoModelForCausalLM

    def skipTest(self, reason=""):
        # for backwards compatibility with unittest style test classes
        pytest.skip(reason)

    def prepare_inputs_for_testing(self):
        input_ids = torch.tensor([[1, 1, 1], [1, 2, 1]]).to(self.torch_device)
        attention_mask = torch.tensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_attributes_parametrized(self, model_id, config_cls, config_kwargs):
        self._test_model_attr(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_adapter_name(self, model_id, config_cls, config_kwargs):
        self._test_adapter_name(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_prepare_for_training_parametrized(self, model_id, config_cls, config_kwargs):
        self._test_prepare_for_training(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained(self, model_id, config_cls, config_kwargs):
        self._test_save_pretrained(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained_pickle(self, model_id, config_cls, config_kwargs):
        self._test_save_pretrained(model_id, config_cls, config_kwargs.copy(), safe_serialization=False)

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained_selected_adapters(self, model_id, config_cls, config_kwargs):
        self._test_save_pretrained_selected_adapters(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained_selected_adapters_pickle(self, model_id, config_cls, config_kwargs):
        self._test_save_pretrained_selected_adapters(
            model_id, config_cls, config_kwargs.copy(), safe_serialization=False
        )

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_from_pretrained_config_construction(self, model_id, config_cls, config_kwargs):
        self._test_from_pretrained_config_construction(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_merge_layers(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_merge_layers(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_merge_layers_multi(self, model_id, config_cls, config_kwargs):
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
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        msg = "lora.ParamWrapper does not support mixed adapter batches yet."
        with pytest.raises(ValueError, match=msg):
            self._test_mixed_adapter_batches(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_generate_with_mixed_adapter_batches(self, model_id, config_cls, config_kwargs):
        if config_cls != LoraConfig:
            pytest.skip("Mixed adapter batches not supported for this config.")
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)

        msg = "lora.ParamWrapper does not support mixed adapter batches yet."
        with pytest.raises(ValueError, match=msg):
            self._test_generate_with_mixed_adapter_batches_and_beam_search(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_generate(self, model_id, config_cls, config_kwargs):
        self._test_generate(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_generate_pos_args(self, model_id, config_cls, config_kwargs):
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
        self._test_training(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_training_decoders_gradient_checkpointing(self, model_id, config_cls, config_kwargs):
        self._test_training_gradient_checkpointing(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_inference_safetensors(self, model_id, config_cls, config_kwargs):
        self._test_inference_safetensors(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_peft_model_device_map(self, model_id, config_cls, config_kwargs):
        self._test_peft_model_device_map(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_delete_adapter(self, model_id, config_cls, config_kwargs):
        self._test_delete_adapter(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_delete_inactive_adapter(self, model_id, config_cls, config_kwargs):
        self._test_delete_inactive_adapter(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_adding_multiple_adapters_with_bias_raises(self, model_id, config_cls, config_kwargs):
        self._test_adding_multiple_adapters_with_bias_raises(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_unload_adapter(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_unload_adapter(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_weighted_combination_of_adapters(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        msg = "add_weighted_adapter does not support targeting nn.Parameter"
        with pytest.raises(ValueError, match=msg):
            self._test_weighted_combination_of_adapters(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_training_prompt_learning_tasks(self, model_id, config_cls, config_kwargs):
        self._test_training_prompt_learning_tasks(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_disable_adapter(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_disable_adapter(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_passing_input_embeds_works(self, model_id, config_cls, config_kwargs):
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
