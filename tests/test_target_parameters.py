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

from .testing_common import PeftCommonTester, hub_online_once
from .testing_utils import set_init_weights_false


PEFT_DECODER_MODELS_TO_TEST = [
    "trl-internal-testing/tiny-Llama4ForCausalLM",
]

# TODO Missing from this list are LoKr, LoHa, LN Tuning, add them
ALL_CONFIGS = [
    # target down_proj
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
    # target gate_up_proj and down_proj (but not on the same module!)
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
                "0.feed_forward.experts.gate_up_proj",
                "1.feed_forward.experts.down_proj",
            ],
        },
    ),
    # target q_proj, v_proj as modules, and down_proj as parameter
    (
        LoraConfig,
        {
            "task_type": "CAUSAL_LM",
            "r": 8,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.0,
            "bias": "none",
            "target_parameters": [
                "feed_forward.experts.down_proj",
            ],
        },
    ),
]


class MyAutoModelForCausalLM(AutoModelForCausalLM):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(*args, **kwargs)

        # check that we load the original model, not, say, a trained checkpoint
        if args[0] == "trl-internal-testing/tiny-Llama4ForCausalLM":
            # model contains weights with values ~1e36 or nan, so we need to reinitialize with sane values
            with torch.no_grad():
                for param in model.parameters():
                    param.data = torch.randn(param.shape)
        return model


class TestDecoderModelsTargetParameters(PeftCommonTester):
    # This is more or less a copy of TestDecoderModels at the time of the PR being added. Unnecessary code is removed,
    # like code required for testing non-LoRA methods. The tests being included are not selected to test specific
    # functionality of targeting nn.Parameters, they (together with the tests in test_custom_models.py) just ensure that
    # generally, nothing is broken.
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
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        msg = "lora.ParamWrapper does not support mixed adapter batches yet."
        with pytest.raises(ValueError, match=msg):
            self._test_mixed_adapter_batches(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id", PEFT_DECODER_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", ALL_CONFIGS)
    def test_generate_with_mixed_adapter_batches(self, model_id, config_cls, config_kwargs):
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


class TestTargetParameter:
    def test_targeting_module_and_targeting_param_equivalent(self):
        # note: we purposely target the gate_proj because its weight is not square (unlike q_proj, ...), this makes it
        # easier to catch shape errors
        torch.manual_seed(0)
        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        with hub_online_once(model_id):
            model0 = AutoModelForCausalLM.from_pretrained(model_id)
            x = torch.arange(10).view(2, 5)
            with torch.inference_mode():
                out_base = model0(x, output_hidden_states=True).hidden_states[-1]

            # targeting the module
            config0 = LoraConfig(target_modules=["gate_proj"], init_lora_weights=False)
            model0 = get_peft_model(model0, config0)

            # targeting the parameter
            model1 = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
            config1 = LoraConfig(target_modules=[], target_parameters=["gate_proj.weight"], init_lora_weights=False)
            model1 = get_peft_model(model1, config1)

            gate_proj_0_0 = model0.base_model.model.model.layers[0].mlp.gate_proj
            gate_proj_0_1 = model0.base_model.model.model.layers[1].mlp.gate_proj
            gate_proj_1_0 = model1.base_model.model.model.layers[0].mlp.gate_proj
            gate_proj_1_1 = model1.base_model.model.model.layers[1].mlp.gate_proj

            # ensure that the randomly initialized LoRA weights are identical
            gate_proj_1_0.lora_A.default.weight.data.copy_(gate_proj_0_0.lora_A.default.weight.data)
            gate_proj_1_1.lora_A.default.weight.data.copy_(gate_proj_0_1.lora_A.default.weight.data)
            gate_proj_1_0.lora_B.default.weight.data.copy_(gate_proj_0_0.lora_B.default.weight.data)
            gate_proj_1_1.lora_B.default.weight.data.copy_(gate_proj_0_1.lora_B.default.weight.data)

            with torch.inference_mode():
                out_lora_0 = model0(x, output_hidden_states=True).hidden_states[-1]
                out_lora_1 = model1(x, output_hidden_states=True).hidden_states[-1]

            # sanity check: basemodel outputs should be different

            atol, rtol = 1e-6, 1e-6
            assert not torch.allclose(out_base, out_lora_0, atol=atol, rtol=rtol)

            # LoRA outputs should be the same
            assert torch.allclose(out_lora_0, out_lora_1, atol=atol, rtol=rtol)

    def test_target_multiple_parameters_on_same_module(self):
        # for now, it is not supported to target multiple parameters from the same module with the same adapter,
        # however, it is possible to target multiple parameters from same module with different adapters
        torch.manual_seed(0)
        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id)
            x = torch.arange(10).view(2, 5)
            with torch.inference_mode():
                out_base = model(x, output_hidden_states=True).hidden_states[-1]

            # targeting gate_up_proj
            config0 = LoraConfig(target_parameters=["feed_forward.experts.gate_up_proj"], init_lora_weights=False)
            model = get_peft_model(model, config0)
            with torch.inference_mode():
                out_lora_0 = model(x, output_hidden_states=True).hidden_states[-1]
            atol, rtol = 1e-6, 1e-6
            assert not torch.allclose(out_base, out_lora_0, atol=atol, rtol=rtol)

            # targeting down_proj
            config1 = LoraConfig(target_parameters=["feed_forward.experts.down_proj"], init_lora_weights=False)
            model.add_adapter("other", config1)
            model.set_adapter("other")
            with torch.inference_mode():
                out_lora_1 = model(x, output_hidden_states=True).hidden_states[-1]
            assert not torch.allclose(out_base, out_lora_1, atol=atol, rtol=rtol)
            assert not torch.allclose(out_lora_0, out_lora_1, atol=atol, rtol=rtol)

            # targeting both gate_up_proj and down_proj
            model.base_model.set_adapter(["default", "other"])
            with torch.inference_mode():
                out_lora_01 = model(x, output_hidden_states=True).hidden_states[-1]
            assert not torch.allclose(out_base, out_lora_01, atol=atol, rtol=rtol)
            assert not torch.allclose(out_lora_0, out_lora_01, atol=atol, rtol=rtol)
            assert not torch.allclose(out_lora_1, out_lora_01, atol=atol, rtol=rtol)
