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

import re

import pytest
import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

import peft
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft.tuners.lora.layer import Linear, LoraLayer, ParamWrapper

from .testing_common import PeftCommonTester
from .testing_utils import hub_online_once, set_init_weights_false


ALL_CONFIGS = [
    ##########
    # Llama4 #
    ##########
    # target down_proj
    (
        "trl-internal-testing/tiny-Llama4ForCausalLM",
        LoraConfig,
        {
            "task_type": TaskType.CAUSAL_LM,
            "target_modules": [],
            "lora_dropout": 0.0,
            "target_parameters": [
                "feed_forward.experts.down_proj",
            ],
        },
    ),
    # target gate_up_proj and down_proj, but not on the same module
    (
        "trl-internal-testing/tiny-Llama4ForCausalLM",
        LoraConfig,
        {
            "task_type": TaskType.CAUSAL_LM,
            "target_modules": [],
            "lora_dropout": 0.0,
            "target_parameters": [
                "0.feed_forward.experts.gate_up_proj",
                "1.feed_forward.experts.down_proj",
            ],
        },
    ),
    # target down_proj and gate_up_proj on the same module
    (
        "trl-internal-testing/tiny-Llama4ForCausalLM",
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
                "feed_forward.experts.gate_up_proj",
            ],
        },
    ),
    # target q_proj, v_proj as modules, and down_proj as parameter
    (
        "trl-internal-testing/tiny-Llama4ForCausalLM",
        LoraConfig,
        {
            "task_type": TaskType.CAUSAL_LM,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.0,
            "target_parameters": [
                "feed_forward.experts.down_proj",
            ],
        },
    ),
    ###########
    # gpt-oss #
    ###########
    # target down_proj
    (
        "trl-internal-testing/tiny-GptOssForCausalLM",
        LoraConfig,
        {
            "task_type": TaskType.CAUSAL_LM,
            "target_modules": [],
            "lora_dropout": 0.0,
            "target_parameters": [
                "mlp.experts.down_proj",
            ],
        },
    ),
    # target gate_up_proj and down_proj, but not on the same module
    (
        "trl-internal-testing/tiny-GptOssForCausalLM",
        LoraConfig,
        {
            "task_type": TaskType.CAUSAL_LM,
            "target_modules": [],
            "lora_dropout": 0.0,
            "target_parameters": [
                "0.mlp.experts.gate_up_proj",
                "1.mlp.experts.down_proj",
            ],
        },
    ),
    # target down_proj and gate_up_proj on the same module
    (
        "trl-internal-testing/tiny-GptOssForCausalLM",
        LoraConfig,
        {
            "task_type": "CAUSAL_LM",
            "r": 8,
            "lora_alpha": 32,
            "target_modules": None,
            "lora_dropout": 0.0,
            "bias": "none",
            "target_parameters": [
                "mlp.experts.down_proj",
                "mlp.experts.gate_up_proj",
            ],
        },
    ),
    # target q_proj, v_proj as modules, and down_proj as parameter
    (
        "trl-internal-testing/tiny-GptOssForCausalLM",
        LoraConfig,
        {
            "task_type": TaskType.CAUSAL_LM,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.0,
            "target_parameters": [
                "mlp.experts.down_proj",
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
        elif args[0] == "trl-internal-testing/tiny-GptOssForCausalLM":
            # model is bf16, which trips up some tests that require tight tolerances
            with torch.no_grad():
                model.float()
        return model


def test_rank_pattern_for_moe_target_parameters(tmp_path):
    model_id = "trl-internal-testing/tiny-Llama4ForCausalLM"
    with hub_online_once(model_id):
        model = MyAutoModelForCausalLM.from_pretrained(model_id)
        num_experts = getattr(model.config, "num_local_experts", None) or getattr(model.config, "num_experts", None)
        assert num_experts is not None
        r = 8
        effective_r = max(1, r // num_experts)
        config = LoraConfig(
            r=r,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            target_parameters=["feed_forward.experts.gate_up_proj"],
            rank_pattern={
                "experts.gate_up_proj": effective_r,
            },
            init_lora_weights=False,
        )
        model = get_peft_model(model, config)

        wrappers = [
            module
            for module in model.modules()
            if isinstance(module, ParamWrapper) and module.parameter_name == "gate_up_proj"
        ]
        assert wrappers, "Expected to find ParamWrapper for gate_up_proj."
        lora_module = wrappers[0]
        assert lora_module.r["default"] == effective_r
        assert lora_module.lora_A["default"].weight.shape[0] == effective_r * num_experts
        assert lora_module.scaling["default"] == config.lora_alpha / effective_r
        assert config.r == r


class TestDecoderModelsTargetParameters(PeftCommonTester):
    # This is more or less a copy of TestDecoderModels at the time of the PR being added. Unnecessary code is removed,
    # like code required for testing non-LoRA methods. The tests being included are not selected to test specific
    # functionality of targeting nn.Parameters, they (together with the tests in test_custom_models.py) just ensure that
    # generally, nothing is broken.
    transformers_class = MyAutoModelForCausalLM

    def prepare_inputs_for_testing(self):
        input_ids = torch.tensor([[1, 1, 1], [1, 2, 1]]).to(self.torch_device)
        attention_mask = torch.tensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_attributes_parametrized(self, model_id, config_cls, config_kwargs):
        self._test_model_attr(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_adapter_name(self, model_id, config_cls, config_kwargs):
        self._test_adapter_name(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_prepare_for_training_parametrized(self, model_id, config_cls, config_kwargs):
        self._test_prepare_for_training(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_save_pretrained(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained_pickle(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_save_pretrained(model_id, config_cls, config_kwargs.copy(), safe_serialization=False)

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained_selected_adapters(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_save_pretrained_selected_adapters(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_save_pretrained_selected_adapters_pickle(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_save_pretrained_selected_adapters(
            model_id, config_cls, config_kwargs.copy(), safe_serialization=False
        )

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_from_pretrained_config_construction(self, model_id, config_cls, config_kwargs):
        self._test_from_pretrained_config_construction(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_merge_layers(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_merge_layers(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_merge_layers_multi(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_merge_layers_multi(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_merge_layers_nan(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_merge_layers_nan(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_mixed_adapter_batches(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        msg = "lora.ParamWrapper does not support mixed adapter batches yet."
        with pytest.raises(ValueError, match=msg):
            self._test_mixed_adapter_batches(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_generate_with_mixed_adapter_batches(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        msg = "lora.ParamWrapper does not support mixed adapter batches yet."
        with pytest.raises(ValueError, match=msg):
            self._test_generate_with_mixed_adapter_batches_and_beam_search(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_generate(self, model_id, config_cls, config_kwargs):
        self._test_generate(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_generate_pos_args(self, model_id, config_cls, config_kwargs):
        self._test_generate_pos_args(model_id, config_cls, config_kwargs.copy(), raises_err=False)

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_merge_layers_fp16(self, model_id, config_cls, config_kwargs):
        self._test_merge_layers_fp16(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_generate_half_prec(self, model_id, config_cls, config_kwargs):
        self._test_generate_half_prec(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_training_decoders(self, model_id, config_cls, config_kwargs):
        self._test_training(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_training_decoders_gradient_checkpointing(self, model_id, config_cls, config_kwargs):
        self._test_training_gradient_checkpointing(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_inference_safetensors(self, model_id, config_cls, config_kwargs):
        self._test_inference_safetensors(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_peft_model_device_map(self, model_id, config_cls, config_kwargs):
        self._test_peft_model_device_map(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_delete_adapter(self, model_id, config_cls, config_kwargs):
        self._test_delete_adapter(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_delete_inactive_adapter(self, model_id, config_cls, config_kwargs):
        self._test_delete_inactive_adapter(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_adding_multiple_adapters_with_bias_raises(self, model_id, config_cls, config_kwargs):
        self._test_adding_multiple_adapters_with_bias_raises(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_unload_adapter(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_unload_adapter(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_weighted_combination_of_adapters(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        msg = "add_weighted_adapter does not support targeting nn.Parameter"
        with pytest.raises(ValueError, match=msg):
            self._test_weighted_combination_of_adapters(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_training_prompt_learning_tasks(self, model_id, config_cls, config_kwargs):
        self._test_training_prompt_learning_tasks(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_disable_adapter(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_disable_adapter(model_id, config_cls, config_kwargs.copy())

    @pytest.mark.parametrize("model_id,config_cls,config_kwargs", ALL_CONFIGS)
    def test_passing_input_embeds_works(self, model_id, config_cls, config_kwargs):
        self._test_passing_input_embeds_works("", model_id, config_cls, config_kwargs.copy())


class TestTargetParameters:
    # Tests specifically designed for target_parameters
    def test_targeting_module_and_targeting_param_equivalent(self):
        # Test that using LoRA with target_modules vs target_parameters yields identical results.
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

    def test_target_multiple_parameters_on_same_module(self, monkeypatch):
        # test that if we target multiple nn.Parameters on the same module, all of them are being used during the
        # forward pass
        torch.manual_seed(0)
        model_id = "trl-internal-testing/tiny-Llama4ForCausalLM"
        with hub_online_once(model_id):
            x = torch.arange(10).view(2, 5)
            model = MyAutoModelForCausalLM.from_pretrained(model_id)
            shape_gate_up_proj = model.model.layers[0].feed_forward.experts.gate_up_proj.shape
            shape_down_proj = model.model.layers[0].feed_forward.experts.down_proj.shape
            num_layers = len(model.model.layers)

            target_parameters = ["feed_forward.experts.gate_up_proj", "feed_forward.experts.down_proj"]
            num_params = len(target_parameters)
            config = LoraConfig(target_parameters=target_parameters, init_lora_weights=False)
            model = get_peft_model(model, config)

            # CHECK FORWARD CALLS

            # log the weights seen during the forward call
            weights = []

            def mock_forward(self, W):
                weights.append(W)
                return orig_forward(self, W)

            from peft.tuners.lora.layer import _LoraFactorsProxy

            orig_forward = _LoraFactorsProxy.forward
            monkeypatch.setattr(_LoraFactorsProxy, "forward", mock_forward)

            num_steps = 3
            with torch.inference_mode():
                for _ in range(num_steps):
                    out_base = model(x, output_hidden_states=True).hidden_states[-1]

            actual_call_count = len(weights)
            # Note: We call forward twice per step, once to create the parametrization and once for the actual forward
            # step. This may be a bit wasteful but it's not clear how to prevent this and overall is probably negligible
            num_forward_per_step = 2
            # Since https://github.com/huggingface/transformers/pull/39501, one of the parameters is accessed twice per
            # forward call, but we cache all calls after the first.
            expected_call_count = num_steps * num_layers * num_params * num_forward_per_step
            assert actual_call_count == expected_call_count

            actual_shapes = {W.shape for W in weights}
            expected_shapes = {shape_gate_up_proj, shape_down_proj}
            assert actual_shapes == expected_shapes

            # CHECK WEIGHT UPDATES

            lora_weights_before = {
                k: v.clone() for k, v in model.named_parameters() if "lora_A.default" in k or "lora_B.default" in k
            }
            # sanity check:
            assert len(lora_weights_before) == 2 * num_layers * num_params
            # train
            optim = torch.optim.SGD(model.parameters(), lr=0.01)
            for _ in range(10):
                optim.zero_grad()
                out = model(x)
                loss = out.logits.sum()
                loss.backward()
                optim.step()

            lora_weights_after = {
                k: v for k, v in model.named_parameters() if "lora_A.default" in k or "lora_B.default" in k
            }
            assert lora_weights_before.keys() == lora_weights_after.keys()
            atol, rtol = 0.1, 0.1
            for key in lora_weights_before.keys():
                assert not torch.allclose(lora_weights_before[key], lora_weights_after[key], atol=atol, rtol=rtol)

    def test_target_parameters_forward_under_autocast(self, monkeypatch):
        # Folding the LoRA update into the parameter uses baddbmm, which autocast casts down to the autocast dtype.
        # Since a parametrization may not change the dtype of the parameter, registering it then failed, see #3601.
        torch.manual_seed(0)
        model_id = "trl-internal-testing/tiny-Llama4ForCausalLM"
        with hub_online_once(model_id):
            model = MyAutoModelForCausalLM.from_pretrained(model_id)
            x = torch.arange(10).view(2, 5)
            config = LoraConfig(target_parameters=["feed_forward.experts.gate_up_proj"], init_lora_weights=False)
            model = get_peft_model(model, config)

            # log the dtypes of the folded weights seen during the forward call
            dtypes = []

            def mock_forward(self, W):
                out = orig_forward(self, W)
                dtypes.append(out.dtype)
                return out

            from peft.tuners.lora.layer import _LoraFactorsProxy

            orig_forward = _LoraFactorsProxy.forward
            monkeypatch.setattr(_LoraFactorsProxy, "forward", mock_forward)

            with torch.inference_mode(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                model(x)

            # the fold must preserve the dtype of the targeted parameter, even under autocast
            assert dtypes
            assert set(dtypes) == {torch.float32}

    def test_target_parameters_works_with_existing_parametrization(self):
        # When a parameter is already parametrized, we want the LoRA parametrization to work with it correctly.
        class MyLinear(nn.Linear):
            # For testing purposes, define a linear layer with 2 parameters: weight and other_weight.
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                nn.init.ones_(self.weight)
                self.other_weight = nn.Parameter(torch.ones(self.weight.shape))

        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = MyLinear(2, 2, bias=False)

            def forward(self, x):
                return self.lin(x)

        class MyParametrization(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + 1

        # base model
        model = MyModule()
        x = torch.ones((2, 2))

        # sanity check: result should be 1*1 + 1*1 == 2
        output_base = model(x)
        assert torch.all(output_base == 2)

        # add parametrization to the weight
        nn.utils.parametrize.register_parametrization(model.lin, "weight", MyParametrization())

        # result should be (1+1)*1 + (1+1)*1 == 4
        output_parametrized = model(x)
        assert torch.all(output_parametrized == 4)

        # add LoRA parametrization to the weight
        config = LoraConfig(r=2, lora_alpha=6, target_parameters=["lin.weight"], init_lora_weights=False)
        model = get_peft_model(model, config)
        # manually set LoRA weights to ones
        nn.init.ones_(model.base_model.model.lin.lora_A["default"].weight)
        nn.init.ones_(model.base_model.model.lin.lora_B["default"].weight)

        output_lora = model(x)
        # delta_weight should be: (1+1) * lora_scale = (1+1) * (alpha / rank) = 2 * (6 / 2) = 6
        # result should be: (1+1+6)*1 + (1+1+6)*1 == 8 + 8 == 16
        assert torch.all(output_lora == 16)

        # calling twice should yield the same result
        output_lora2 = model(x)
        assert torch.allclose(output_lora, output_lora2)

        # Adding another adapter that targets a *different* parameter is not allowed: all adapters that use
        # target_parameters must target the same set of parameters.
        config = LoraConfig(r=2, lora_alpha=6, target_parameters=["lin.other_weight"], init_lora_weights=False)
        msg = "all adapters must target the same set of parameters"
        with pytest.raises(ValueError, match=msg):
            model.add_adapter("other", config)
        # the rejected adapter was not added
        assert "other" not in model.peft_config

        # after unloading, the output should be the same as before LoRA was applied
        unloaded = model.unload()
        output_unloaded = unloaded(x)
        assert torch.all(output_unloaded == output_parametrized)

    def test_target_parameter_result_caching_works(self, monkeypatch):
        # See 2912
        # There was an issue with the caching of _LoraParameterProxy not working correctly. This test checks that the
        # results returned from the forward call are all identical to ensure they're not recomputed each time.
        torch.manual_seed(0)
        model_id = "trl-internal-testing/tiny-GptOssForCausalLM"

        tensor_storage = []

        def store_tensors_deco(fn):
            def wrapper(*args, **kwargs):
                result = fn(*args, **kwargs)
                tensor_storage.append(result)
                return result

            return wrapper

        monkeypatch.setattr(
            peft.tuners.lora.layer._LoraFactorsProxy,
            "forward",
            store_tensors_deco(peft.tuners.lora.layer._LoraFactorsProxy.forward),
        )

        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id)
            config = LoraConfig(
                target_modules=[],
                # for simplicity, only target a single layer
                target_parameters=["0.mlp.experts.gate_up_proj"],
            )
            model = get_peft_model(model, config)
            x = torch.arange(100).view(2, 50)  # larger input to hit many experts

            # forward is called twice, once at initialization of the parametrization and once during the forward pass,
            # after which it is cached; without caching, it would be called 25 times.
            output = model(x, output_hidden_states=True)
            assert len(set(map(id, tensor_storage))) == 2

            # sanity check: a second forward call _does_ trigger a new forward
            output = model(x, output_hidden_states=True)
            assert len(set(map(id, tensor_storage))) == 4

    def test_target_parameter_init_does_not_warn_about_unknown_layer_type(self, recwarn):
        # For target parameters, the layer type is not known. This is fine, as the in_features and out_features are
        # derived from the targeted parameter shape. But we need to ensure that there is no warning about the unknown
        # layer type.
        model_id = "trl-internal-testing/tiny-GptOssForCausalLM"
        with hub_online_once(model_id):
            model0 = AutoModelForCausalLM.from_pretrained(model_id)
            config = LoraConfig(
                target_modules=[],
                target_parameters=["0.mlp.experts.gate_up_proj"],
            )
            model = get_peft_model(model0, config)
            warn_messages = (w.message.args[0] for w in recwarn.list)
            msg_start = "Unsupported layer type"
            assert not any(msg.startswith(msg_start) for msg in warn_messages)

    def test_adding_second_adapter_reuses_param_wrapper(self):
        # Adding a second adapter that targets the same parameters must reuse the existing (possibly nested)
        # ParamWrapper(s) instead of nesting new ones. As a result, the number of ParamWrappers stays constant and each
        # of them holds both adapters.
        torch.manual_seed(0)
        model_id = "trl-internal-testing/tiny-Llama4ForCausalLM"
        target_parameters = ["feed_forward.experts.gate_up_proj", "feed_forward.experts.down_proj"]
        with hub_online_once(model_id):
            model = MyAutoModelForCausalLM.from_pretrained(model_id)
            config = LoraConfig(target_modules=[], target_parameters=target_parameters, init_lora_weights=False)
            model = get_peft_model(model, config)
            num_wrappers_single = sum(isinstance(m, ParamWrapper) for m in model.modules())

            config_other = LoraConfig(target_modules=[], target_parameters=target_parameters, init_lora_weights=False)
            model.add_adapter("other", config_other)
            num_wrappers_multi = sum(isinstance(m, ParamWrapper) for m in model.modules())

            # the number of ParamWrappers does not change when adding a second adapter
            assert num_wrappers_single > 0
            assert num_wrappers_multi == num_wrappers_single

            # every ParamWrapper holds both adapters
            for module in model.modules():
                if isinstance(module, ParamWrapper):
                    assert set(module.lora_A.keys()) == {"default", "other"}
                    assert set(module.lora_B.keys()) == {"default", "other"}

    def test_multiple_adapters_load_order_independent(self, tmp_path):
        # Regression test: when multiple adapters target parameters, the saved checkpoint must load correctly regardless
        # of the order in which the adapters are loaded. This is important to test because a previous attempt at
        # implementing multiple target_parameters adapters made use of nesting, so had something like:
        #   wrapper-default (wrapper-other (base-layer))
        # which meant that the state dict for 'other' would contain an extra base layer, which meant it could not be
        # loaded unless the default adapter was loaded first.
        torch.manual_seed(0)
        model_id = "trl-internal-testing/tiny-Llama4ForCausalLM"
        target_parameters = ["feed_forward.experts.gate_up_proj", "feed_forward.experts.down_proj"]
        x = torch.arange(10).view(2, 5)
        with hub_online_once(model_id):
            model = MyAutoModelForCausalLM.from_pretrained(model_id)
            config = LoraConfig(target_modules=[], target_parameters=target_parameters, init_lora_weights=False)
            model = get_peft_model(model, config)
            config_other = LoraConfig(target_modules=[], target_parameters=target_parameters, init_lora_weights=False)
            model.add_adapter("other", config_other)

            # collect the reference outputs of both adapters
            outputs = {}
            for adapter in ["default", "other"]:
                model.set_adapter(adapter)
                with torch.inference_mode():
                    outputs[adapter] = model(x).logits.clone()

            # 'default' is saved to the root, 'other' to a subfolder
            model.save_pretrained(tmp_path)
            del model

            # load in *reverse* order: load 'other' first, then 'default'
            model = MyAutoModelForCausalLM.from_pretrained(model_id)
            model = PeftModel.from_pretrained(model, str(tmp_path / "other"), adapter_name="other")
            load_result = model.load_adapter(str(tmp_path), adapter_name="default")

            assert not load_result.missing_keys
            assert not load_result.unexpected_keys

            for adapter in ["default", "other"]:
                model.set_adapter(adapter)
                with torch.inference_mode():
                    out = model(x).logits
                assert torch.allclose(out, outputs[adapter], atol=1e-5, rtol=1e-5)

    def test_target_parameter_on_top_level_module_raises(self):
        # nn.Parameters that are registered directly on the top-level module (i.e. the module passed to get_peft_model)
        # cannot be targeted. Wrapping the parameter would require replacing the module that holds it with
        # lora.ParamWrapper, but that module is its own parent, so the wrapper ends up registered as a submodule of the
        # very module it wraps. This creates a cyclic module graph, resulting in an error.

        class MyModule(nn.Module):
            # module with a 2d and a 3d nn.Parameter registered directly on the top-level module
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.zeros(10, 10))

        config = LoraConfig(target_parameters=["param"])
        msg = re.escape("Targeting an nn.Parameter on the top-level module is not supported (parameter 'param')")
        with pytest.raises(ValueError, match=msg):
            get_peft_model(MyModule(), config)

    @pytest.fixture
    def deepseek_model(self):
        # docstyle-ignore
        """
        DeepseekV3ForCausalLM(
          (model): DeepseekV3Model(
            (embed_tokens): Embedding(32, 32)
            (layers): ModuleList(
              (0): DeepseekV3DecoderLayer(
                (self_attn): DeepseekV3Attention(
                  (q_a_proj): Linear(in_features=32, out_features=16, bias=False)
                  (q_a_layernorm): DeepseekV3RMSNorm((16,), eps=1e-06)
                  (q_b_proj): Linear(in_features=16, out_features=32, bias=False)
                  (kv_a_proj_with_mqa): Linear(in_features=32, out_features=12, bias=False)
                  (kv_a_layernorm): DeepseekV3RMSNorm((8,), eps=1e-06)
                  (kv_b_proj): Linear(in_features=8, out_features=48, bias=False)
                  (o_proj): Linear(in_features=32, out_features=32, bias=False)
                )
                (mlp): DeepseekV3MLP(
                  (gate_proj): Linear(in_features=32, out_features=32, bias=False)
                  (up_proj): Linear(in_features=32, out_features=32, bias=False)
                  (down_proj): Linear(in_features=32, out_features=32, bias=False)
                  (act_fn): SiLUActivation()
                )
                (input_layernorm): DeepseekV3RMSNorm((32,), eps=1e-06)
                (post_attention_layernorm): DeepseekV3RMSNorm((32,), eps=1e-06)
              )
              (1): DeepseekV3DecoderLayer(
                (self_attn): DeepseekV3Attention(
                  (q_a_proj): Linear(in_features=32, out_features=16, bias=False)
                  (q_a_layernorm): DeepseekV3RMSNorm((16,), eps=1e-06)
                  (q_b_proj): Linear(in_features=16, out_features=32, bias=False)
                  (kv_a_proj_with_mqa): Linear(in_features=32, out_features=12, bias=False)
                  (kv_a_layernorm): DeepseekV3RMSNorm((8,), eps=1e-06)
                  (kv_b_proj): Linear(in_features=8, out_features=48, bias=False)
                  (o_proj): Linear(in_features=32, out_features=32, bias=False)
                )
                (mlp): DeepseekV3MoE(
                  (experts): DeepseekV3Experts(
                    (act_fn): SiLUActivation()
                  )
                  (gate): DeepseekV3TopkRouter()
                  (shared_experts): DeepseekV3MLP(
                    (gate_proj): Linear(in_features=32, out_features=16, bias=False)
                    (up_proj): Linear(in_features=32, out_features=16, bias=False)
                    (down_proj): Linear(in_features=16, out_features=32, bias=False)
                    (act_fn): SiLUActivation()
                  )
                )
                (input_layernorm): DeepseekV3RMSNorm((32,), eps=1e-06)
                (post_attention_layernorm): DeepseekV3RMSNorm((32,), eps=1e-06)
              )
            )
            (norm): DeepseekV3RMSNorm((32,), eps=1e-06)
            (rotary_emb): DeepseekV3RotaryEmbedding()
          )
          (lm_head): Linear(in_features=32, out_features=32, bias=False)
        )
        """
        torch.manual_seed(0)
        config = DeepseekV3Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=32,
            moe_intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            q_lora_rank=16,
            kv_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=8,
            n_routed_experts=4,
            n_shared_experts=1,
            n_group=2,  # V3's router takes the top two experts within each group.
            topk_group=1,
            num_experts_per_tok=2,
            first_k_dense_replace=1,  # first block is dense, second is MoE
            max_position_embeddings=32,
            num_mtp_layers=0,
            use_cache=False,
            attn_implementation="eager",
            experts_implementation="eager",
        )
        return DeepseekV3ForCausalLM(config)

    @pytest.mark.parametrize(
        "target_parameters,target_modules",
        [
            pytest.param(
                target_parameters,
                target_modules,
                # Bare module names without parameter targets still use legacy MoE conversion, this is not fixed (yet)
                # but also not very likely to to be used an issue in practice; users can qualify the module names if
                # needed
                marks=(
                    pytest.mark.xfail(strict=True, reason="Bare targets without parameters use legacy MoE conversion")
                    if not target_parameters and all("." not in name for name in target_modules)
                    else ()
                ),
            )
            for target_parameters in [
                [],
                ["experts.down_proj"],
                ["experts.gate_up_proj", "experts.down_proj"],
            ]
            for target_modules in [
                ["down_proj"],  # this xfails for target_parameters=[]
                ["gate_proj", "down_proj"],  # this xfails for target_parameters=[]
                ["shared_experts.down_proj"],
                ["shared_experts.gate_proj", "shared_experts.down_proj"],
                ["mlp.shared_experts.down_proj"],
                ["mlp.shared_experts.gate_proj", "mlp.shared_experts.down_proj"],
            ]
        ],
    )
    def test_deepseek_v3_target_modules_and_target_parameters_name_overlap(
        self, deepseek_model, target_modules, target_parameters
    ):
        # see #3711: down_proj can both be nn.Linear and an nn.Parameter, don't accidentally target with the wrong type
        mlp0 = deepseek_model.model.layers[0].mlp  # layer 0: all dense
        mlp1 = deepseek_model.model.layers[1].mlp  # layer 1: shared experts dense, experts MoE
        # sanity check
        assert isinstance(mlp0.down_proj, nn.Linear)
        assert isinstance(mlp0.gate_proj, nn.Linear)
        assert isinstance(mlp1.shared_experts.down_proj, nn.Linear)
        assert isinstance(mlp1.shared_experts.gate_proj, nn.Linear)
        assert isinstance(mlp1.experts.down_proj, nn.Parameter)
        assert isinstance(mlp1.experts.gate_up_proj, nn.Parameter)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=2,
            lora_alpha=2,
            lora_dropout=0.0,
            target_modules=target_modules,
            target_parameters=target_parameters,
            init_lora_weights=False,
        )
        peft_model = get_peft_model(deepseek_model, lora_config)

        # check the rewritten config
        assert peft_model.peft_config["default"].target_modules == set(target_modules)
        assert peft_model.peft_config["default"].target_parameters == set(target_parameters or [])

        # check the lora modules
        # for some modules, whether they're targeted depends on the parametrization
        assert isinstance(mlp0.gate_proj, (Linear, nn.Linear))
        assert isinstance(mlp0.down_proj, (Linear, nn.Linear))
        assert isinstance(mlp1.shared_experts.gate_proj, (Linear, nn.Linear))
        # this must be targeted no matter what parametrization
        assert isinstance(mlp1.shared_experts.down_proj, Linear)

        # remember: for the param wrapper, the *parent module* is wrapped
        if len(target_parameters or []) == 0:
            assert not isinstance(mlp1.experts, ParamWrapper)
        else:
            assert isinstance(mlp1.experts, ParamWrapper)
            assert mlp1.experts.parameter_name == "down_proj"
        if len(target_parameters or []) > 1:
            # 2 targets: -> nested ParamWrapper
            assert isinstance(mlp1.experts.base_layer, ParamWrapper)
            assert mlp1.experts.base_layer.parameter_name == "gate_up_proj"

    @pytest.fixture
    def qwen_model(self):
        # docstyle-ignore
        """
        Qwen model mixing dense and MoE layers with ambiguous names:
        Qwen3MoeForCausalLM(
          (model): Qwen3MoeModel(
            (embed_tokens): Embedding(32, 16)
            (layers): ModuleList(
              (0): Qwen3MoeDecoderLayer(
                (self_attn): Qwen3MoeAttention(
                  (q_proj): Linear(in_features=16, out_features=16, bias=False)
                  (k_proj): Linear(in_features=16, out_features=16, bias=False)
                  (v_proj): Linear(in_features=16, out_features=16, bias=False)
                  (o_proj): Linear(in_features=16, out_features=16, bias=False)
                  (q_norm): Qwen3MoeRMSNorm((8,), eps=1e-06)
                  (k_norm): Qwen3MoeRMSNorm((8,), eps=1e-06)
                )
                (mlp): Qwen3MoeMLP(
                  (gate_proj): Linear(in_features=16, out_features=32, bias=False)
                  (up_proj): Linear(in_features=16, out_features=32, bias=False)
                  (down_proj): Linear(in_features=32, out_features=16, bias=False)
                  (act_fn): SiLUActivation()
                )
                (input_layernorm): Qwen3MoeRMSNorm((16,), eps=1e-06)
                (post_attention_layernorm): Qwen3MoeRMSNorm((16,), eps=1e-06)
              )
              (1-2): 2 x Qwen3MoeDecoderLayer(
                (self_attn): Qwen3MoeAttention(
                  (q_proj): Linear(in_features=16, out_features=16, bias=False)
                  (k_proj): Linear(in_features=16, out_features=16, bias=False)
                  (v_proj): Linear(in_features=16, out_features=16, bias=False)
                  (o_proj): Linear(in_features=16, out_features=16, bias=False)
                  (q_norm): Qwen3MoeRMSNorm((8,), eps=1e-06)
                  (k_norm): Qwen3MoeRMSNorm((8,), eps=1e-06)
                )
                (mlp): Qwen3MoeSparseMoeBlock(
                  (experts): Qwen3MoeExperts(
                    (act_fn): SiLUActivation()
                  )
                  (gate): Qwen3MoeTopKRouter()
                )
                (input_layernorm): Qwen3MoeRMSNorm((16,), eps=1e-06)
                (post_attention_layernorm): Qwen3MoeRMSNorm((16,), eps=1e-06)
              )
            )
            (norm): Qwen3MoeRMSNorm((16,), eps=1e-06)
            (rotary_emb): Qwen3MoeRotaryEmbedding()
          )
          (lm_head): Linear(in_features=16, out_features=32, bias=False)
        )
        """
        torch.manual_seed(0)
        config = Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_experts=2,
            num_experts_per_tok=2,
            mlp_only_layers=[0],
            max_position_embeddings=32,
            use_cache=False,
            attn_implementation="eager",
            experts_implementation="eager",
        )
        return Qwen3MoeForCausalLM(config)

    @pytest.mark.parametrize("prefix", ["model.layers.0.mlp", "mlp"], ids=["full-path", "suffix"])
    @pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
    def test_qwen_linear_module_target_is_preserved(self, qwen_model, prefix, projection):
        # The dense MLP has ordinary linear projections with the same leaf names as the legacy expert projections.
        # Preserve its module target alongside an explicit, layer-specific fused parameter target.
        target_module = f"{prefix}.{projection}"
        target_parameter = "model.layers.1.mlp.experts.down_proj"
        config = LoraConfig(r=2, target_modules=[target_module], target_parameters=[target_parameter])
        model = get_peft_model(qwen_model, config)

        assert config.target_modules == {target_module}
        assert set(config.target_parameters) == {target_parameter}

        layers = model.get_base_model().model.layers
        assert isinstance(getattr(layers[0].mlp, projection), Linear)
        assert isinstance(layers[1].mlp.experts, ParamWrapper)
        assert layers[1].mlp.experts.parameter_name == "down_proj"
        assert not isinstance(layers[2].mlp.experts, LoraLayer)
        assert len([module for module in model.modules() if isinstance(module, LoraLayer)]) == 2

    @pytest.mark.parametrize(
        "target_modules",
        [
            [
                "1.self_attn.q_proj",
                "1.mlp.gate",
                "1.mlp.experts.gate_proj",
                "1.mlp.experts.up_proj",
                "1.mlp.experts.down_proj",
            ],
            r"model\.layers\.1\.(self_attn\.q_proj|mlp\.(gate|experts\.(gate_proj|up_proj|down_proj)))",
            r"model\.layers\.1\.(self_attn\.q_proj|mlp\.(gate|experts\.\d+\.(gate_proj|up_proj|down_proj)))",
        ],
        ids=["suffixes", "regex", "numbered-expert-regex"],
    )
    @pytest.mark.xfail(strict=True, reason="Legacy expert conversion still loses layer scope")
    def test_qwen_legacy_targets_install_parameter_adapters_on_correct_layer(self, qwen_model, target_modules):
        # Looking only at named_modules() would either reject the custom router or silently drop the expert targets
        # while still adapting q_proj. Check actual wrappers and gradients, including the fused gate/up rank.
        config = LoraConfig(r=2, lora_alpha=2, target_modules=target_modules)
        model = get_peft_model(qwen_model, config)

        # only layer 1 should be targeted but right now, all layers are targeted -> xfail
        assert config.target_parameters != {"down_proj", "gate_up_proj", "gate.weight"}

        layers = model.get_base_model().model.layers
        assert isinstance(layers[1].self_attn.q_proj, Linear)
        assert isinstance(layers[1].mlp.gate, ParamWrapper)
        assert layers[1].mlp.gate.parameter_name == "weight"

        experts = layers[1].mlp.experts
        assert isinstance(experts, ParamWrapper)
        assert experts.parameter_name == "down_proj"
        assert isinstance(experts.base_layer, ParamWrapper)
        assert experts.base_layer.parameter_name == "gate_up_proj"
        # fused weight:
        assert experts.base_layer.r["default"] == 4
        assert experts.base_layer.lora_alpha["default"] == 4

        # layers[2] should not be updated but it is -> xfail
        assert not isinstance(layers[2].mlp.experts, LoraLayer)
        assert not isinstance(layers[2].mlp.gate, LoraLayer)
        assert not isinstance(layers[0].mlp.down_proj, LoraLayer)

        # expected: q_proj, gate, gate_up_proj, down_proj for one layer -> xfail
        adapters = [module for module in model.modules() if isinstance(module, LoraLayer)]
        assert len(adapters) == 4

        # Resolve a fresh legacy config against the already wrapped architecture, without targeting adapter internals.
        model.add_adapter("other", LoraConfig(r=2, lora_alpha=2, target_modules=target_modules))
        # same issue as above: should be only 4 targets -> xfail
        assert len([module for module in model.modules() if isinstance(module, LoraLayer)]) == 4
        assert all("other" in adapter.lora_A for adapter in adapters)

    @pytest.mark.parametrize("target_modules", [["down_proj"], r".*\.down_proj"], ids=["suffix", "regex"])
    @pytest.mark.xfail(strict=True, reason="Broad MoE targets still omit dense modules")
    def test_qwen_target_can_match_dense_modules_and_expert_parameters(self, qwen_model, target_modules):
        # down_proj can refer to both nn.Linear in layer 0 and MoE in layer 1 & 2
        config = LoraConfig(r=2, target_modules=target_modules)
        model = get_peft_model(qwen_model, config)

        # layer 0: normal target_modules -> xfail
        assert len(config.target_modules) == 1
        assert all(".0." in key for key in config.target_modules)
        # layers 1 and 2: target_parameters -> xfail
        assert len(config.target_parameters) == 2
        assert all(".1." in key or ".2." in key for key in config.target_parameters)

        layers = model.get_base_model().model.layers
        # not targeted -> xfail
        assert isinstance(layers[0].mlp.down_proj, Linear)

        # parameters are correctly targeted:
        for idx in (1, 2):
            assert isinstance(layers[idx].mlp.experts, ParamWrapper)
            assert layers[idx].mlp.experts.parameter_name == "down_proj"

        # one for each layer but layer 0 is not targeted -> xfail
        assert len([module for module in model.modules() if isinstance(module, LoraLayer)]) == 3

    @pytest.mark.parametrize(
        "kwargs",
        [
            # gate_proj and up_proj would be fused to gate_up_proj
            {"target_modules": ["1.mlp.experts.0.down_proj"]},
            {"target_modules": ["1.mlp.experts.gate_proj", "2.mlp.experts.up_proj"]},
            # with fused experts, it's no longer possible to exclude one specific expert, so this can no longer be
            # expressed
            {"target_modules": ["down_proj"], "exclude_modules": ["1.mlp.experts.0.down_proj"]},
        ],
        ids=["target-subset", "non-overlapping-targets", "exclude-expert"],
    )
    @pytest.mark.xfail(strict=True, reason="Legacy conversion still broadens numbered expert selections")
    def test_targeting_subset_of_layers_to_be_fused_raises(self, qwen_model, kwargs):
        # when trying to target a subset of modules from layers that will be fused, an error should be raised
        config = LoraConfig(**kwargs)
        with pytest.raises(ValueError, match="Cannot convert a subset of experts"):
            get_peft_model(qwen_model, config)
