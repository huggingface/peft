#!/usr/bin/env python3

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
import copy
import re

import pytest
import torch
from diffusers import StableDiffusionPipeline
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

from peft import (
    AdaLoraConfig,
    IA3Config,
    LoKrConfig,
    LoraConfig,
    RandLoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    inject_adapter_in_model,
    set_peft_model_state_dict,
)
from peft.tuners import lora
from peft.utils import ModulesToSaveWrapper

from .testing_utils import hub_online_once


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 10)
        self.linear = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10, bias=True)
        self.lm_head = torch.nn.Linear(10, 10)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.linear(x)
        x = self.lm_head(x)
        return x


class TestLowLevelFunctional:
    # Some simple tests for the low level API
    @pytest.fixture
    def model(self):
        model = DummyModel()

        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            target_modules=["linear"],
        )

        return inject_adapter_in_model(lora_config, model)

    def test_inject_adapter_in_model(self, model):
        dummy_inputs = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]])
        _ = model(dummy_inputs)

        for name, module in model.named_modules():
            if name == "linear":
                assert hasattr(module, "lora_A")
                assert hasattr(module, "lora_B")

    def test_get_peft_model_state_dict(self, model):
        peft_state_dict = get_peft_model_state_dict(model)

        for key in peft_state_dict.keys():
            assert "lora" in key

    def test_modules_to_save(self):
        model = DummyModel()

        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            target_modules=["linear"],
            modules_to_save=["embedding", "linear2"],
        )

        model = inject_adapter_in_model(lora_config, model)

        for name, module in model.named_modules():
            if name == "linear":
                assert hasattr(module, "lora_A")
                assert hasattr(module, "lora_B")
            elif name in ["embedding", "linear2"]:
                assert isinstance(module, ModulesToSaveWrapper)

        state_dict = get_peft_model_state_dict(model)

        assert "embedding.weight" in state_dict.keys()

        assert hasattr(model.embedding, "weight")

        assert hasattr(model.linear2, "weight")
        assert hasattr(model.linear2, "bias")


class TestInjectAdapterFromStateDict:
    # The inject_adapter_in_model function can determine the target modules based on the LoraConfig (default) or based
    # on a state_dict (or rather, the state_dict keys). Here we test that the latter works as expected.

    # We test a subset of model classes and PEFT configs, testing everything would be excessive
    @pytest.mark.parametrize(
        "model_cls_and_id",
        [
            (AutoModelForCausalLM, "trl-internal-testing/tiny-random-LlamaForCausalLM"),
            (AutoModel, "peft-internal-testing/tiny-random-BertModel"),
            (AutoModelForSeq2SeqLM, "peft-internal-testing/tiny-random-BartForConditionalGeneration"),
            (AutoModelForSequenceClassification, "peft-internal-testing/tiny-random-RobertaForSequenceClassification"),
        ],
        ids=["Llama", "Bert", "Bart", "Roberta"],
    )
    @pytest.mark.parametrize(
        "config",
        [
            AdaLoraConfig(total_step=5),
            IA3Config(),
            LoKrConfig(),
            LoraConfig(),
            RandLoraConfig(),
        ],
        ids=["AdaLoRA", "IA3", "LoKr", "LoRA", "RandLoRA"],
    )
    def test_inject_from_state_dict_and_from_config_target_same_layers(self, model_cls_and_id, config, recwarn):
        model_cls, model_id = model_cls_and_id
        config = copy.deepcopy(config)  # since PEFT may mutate it

        with hub_online_once(model_id):
            # use config for injection
            model = model_cls.from_pretrained(model_id)
            model = inject_adapter_in_model(config, model)
            sd_before = get_peft_model_state_dict(model)
            del model

            model = model_cls.from_pretrained(model_id)
            # get other warnings, if any, out of the way
            recwarn.clear()
            # assure that this doesn't cause any warnings
            model = inject_adapter_in_model(config, model, state_dict=sd_before)
            assert not recwarn.list

            sd_after = get_peft_model_state_dict(model)

            # We exepct the same keys and the same shapes of the weights. Don't check the values: injection is only
            # about creating the PEFT adapter, not about loading the actual weights
            assert len(sd_before) > 0
            assert sd_before.keys() == sd_after.keys()
            for key in sd_before.keys():
                assert sd_before[key].shape == sd_after[key].shape

    def test_inject_from_state_dict_transformers(self):
        model_id = "peft-internal-testing/opt-125m"
        config = LoraConfig()

        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id)
            model.add_adapter(config)
            sd_before = get_peft_model_state_dict(model)
            del model

            model = AutoModelForCausalLM.from_pretrained(model_id)
            model = inject_adapter_in_model(config, model, state_dict=sd_before)
            sd_after = get_peft_model_state_dict(model)

            # We exepct the same keys and the same shapes of the weights. Don't check the values: injection is only
            # about creating the PEFT adapter, not about loading the actual weights
            assert len(sd_before) > 0
            assert sd_before.keys() == sd_after.keys()
            for key in sd_before.keys():
                assert sd_before[key].shape == sd_after[key].shape

    def test_inject_from_state_dict_transformers_irregular_targets(self):
        # ensure that this works even if an "irregular" pattern is used, i.e. only targeting some modules on some layers
        model_id = "peft-internal-testing/opt-125m"
        config = LoraConfig(
            target_modules=r".*\.[0-5]\.self_attn\.v_proj|.*\.[4-7]\.self_attn\.k_proj",
        )

        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id)
            model.add_adapter(config)
            sd_before = get_peft_model_state_dict(model)
            del model

            model = AutoModelForCausalLM.from_pretrained(model_id)
            model = inject_adapter_in_model(config, model, state_dict=sd_before)
            sd_after = get_peft_model_state_dict(model)

            # We exepct the same keys and the same shapes of the weights. Don't check the values: injection is only
            # about creating the PEFT adapter, not about loading the actual weights
            assert len(sd_before) > 0
            assert sd_before.keys() == sd_after.keys()
            for key in sd_before.keys():
                assert sd_before[key].shape == sd_after[key].shape

    def test_inject_from_state_dict_transformers_target_parameters_raises(self):
        # Injecting from state_dict does not correctly identify target_parameters. This is because, just from looking at
        # the state_dict, we cannot tell if the user intends to use target_modules or target_parameters. Currently, we
        # just assume the former, thus applying normal lora.Linear etc. layers instead of lora.ParamWrapper. When we
        # detect that the user tries to do this, we raise an error.
        model_id = "peft-internal-testing/opt-125m"
        config = LoraConfig(target_modules=[], target_parameters=["q_proj.weight", "v_proj.weight"])

        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id)
            model.add_adapter(config)
            sd = get_peft_model_state_dict(model)
            del model

            model = AutoModelForCausalLM.from_pretrained(model_id)
            msg = "Trying to inject a PEFT adapter from a state_dict but the PEFT config uses `target_parameters`"
            with pytest.raises(ValueError, match=msg):
                inject_adapter_in_model(config, model, state_dict=sd)

    @pytest.mark.xfail(
        reason="Loading from state_dict with target_parameters fails", raises=AssertionError, strict=True
    )
    def test_inject_from_state_dict_transformers_target_parameters_fails(self):
        # Injecting from state_dict does not correctly identify target_parameters. This is because, just from looking at
        # the state_dict, we cannot tell if the user intends to use target_modules or target_parameters. Currently, we
        # just assume the former, thus applying normal lora.Linear etc. layers instead of lora.ParamWrapper. When we
        # don't detect that the user tries to do this, there is nothing that can be done.
        model_id = "peft-internal-testing/opt-125m"
        config = LoraConfig(target_modules=[], target_parameters=["q_proj.weight", "v_proj.weight"])

        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id)
            model.add_adapter(config)
            # sanity check:
            for name, module in model.named_modules():
                if name.endswith((".q_proj", ".v_proj")):
                    assert isinstance(module, lora.ParamWrapper)

            sd_before = get_peft_model_state_dict(model)
            del model

            model = AutoModelForCausalLM.from_pretrained(model_id)
            config = LoraConfig()  # no target_parameters defined, we cannot know the original intent
            model = inject_adapter_in_model(config, model, state_dict=sd_before)
            sd_after = get_peft_model_state_dict(model)

            # this fails, we get lora.Linear instances
            for name, module in model.named_modules():
                if name.endswith((".q_proj", ".v_proj")):
                    assert isinstance(module, lora.ParamWrapper)

    def test_inject_from_state_dict_stable_diffusion(self):
        # same test as above, but with stable diffusion model and only testing LoRA
        model_id = "hf-internal-testing/tiny-sd-pipe"
        config_text_encoder = LoraConfig(target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"])
        config_unet = LoraConfig(
            target_modules=[
                "proj_in",
                "proj_out",
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
                "ff.net.0.proj",
                "ff.net.2",
            ]
        )
        with hub_online_once(model_id):
            pipe = StableDiffusionPipeline.from_pretrained(model_id)
            pipe.text_encoder.add_adapter(config_text_encoder)
            pipe.unet.add_adapter(config_unet)

            sd_te_before = get_peft_model_state_dict(pipe.text_encoder)
            sd_unet_before = get_peft_model_state_dict(pipe.unet)
            del pipe

            pipe = StableDiffusionPipeline.from_pretrained(model_id)
            inject_adapter_in_model(config_text_encoder, pipe.text_encoder, state_dict=sd_te_before)
            inject_adapter_in_model(config_unet, pipe.unet, state_dict=sd_unet_before)

            sd_te_after = get_peft_model_state_dict(pipe.text_encoder)
            sd_unet_after = get_peft_model_state_dict(pipe.unet)

            # We exepct the same keys and the same shapes of the weights. Don't check the values: injection is only
            # about creating the PEFT adapter, not about loading the actual weights
            assert len(sd_te_before) > 0
            assert sd_te_before.keys() == sd_te_after.keys()
            for key in sd_te_before.keys():
                assert sd_te_before[key].shape == sd_te_after[key].shape

            assert len(sd_unet_before) > 0
            assert sd_unet_before.keys() == sd_unet_after.keys()
            for key in sd_unet_before.keys():
                assert sd_unet_before[key].shape == sd_unet_after[key].shape

    def test_inject_from_state_dict_low_cpu_mem_usage(self):
        model_id = "peft-internal-testing/opt-125m"
        config = LoraConfig()

        with hub_online_once(model_id):
            # use config for injection
            model = AutoModelForCausalLM.from_pretrained(model_id)
            model = inject_adapter_in_model(config, model)
            sd_before = get_peft_model_state_dict(model)
            del model

            model = AutoModelForCausalLM.from_pretrained(model_id)
            model = inject_adapter_in_model(config, model, state_dict=sd_before, low_cpu_mem_usage=True)
            # all PEFT parameters should be on meta device
            assert {p.device.type for p in get_peft_model_state_dict(model).values()} == {"meta"}

    def test_inject_from_state_dict_missing_keys_warning(self):
        # check that if the PEFT config specifies **more** taget modules than the state_dict, we get a warning for that
        model_id = "peft-internal-testing/opt-125m"
        config = LoraConfig()

        with hub_online_once(model_id):
            # use config for injection
            model = AutoModelForCausalLM.from_pretrained(model_id)
            model = inject_adapter_in_model(config, model)
            sd_before = get_peft_model_state_dict(model)
            del model

            # delete a keys for one module from state_dict
            del sd_before["model.decoder.layers.5.self_attn.q_proj.lora_A.weight"]
            del sd_before["model.decoder.layers.5.self_attn.q_proj.lora_B.weight"]

            model = AutoModelForCausalLM.from_pretrained(model_id)
            msg = re.escape(
                "While injecting the PEFT adapters, an inconsistency was discovered between the PEFT config and "
                "the provided state_dict. This is not necessarily an issue and can be ignored if this was the "
                "intent. The PEFT config contained these additional target modules: "
                "['model.decoder.layers.5.self_attn.q_proj']. "
            )

            with pytest.warns(RuntimeWarning, match=msg):  # as rec:#(UserWarning, match=msg) as rec:
                model = inject_adapter_in_model(config, model, state_dict=sd_before, low_cpu_mem_usage=True)

            # besides the warning, the rest of the injection should work
            sd_after = get_peft_model_state_dict(model)
            assert len(sd_before) > 0
            assert sd_before.keys() == sd_after.keys()
            for key in sd_before.keys():
                assert sd_before[key].shape == sd_after[key].shape

    def test_inject_from_state_dict_extra_keys_warning(self):
        # check that if the PEFT config specifies **fewer** taget modules than the state_dict, we get a warning for that
        model_id = "peft-internal-testing/opt-125m"
        config = LoraConfig()

        with hub_online_once(model_id):
            # use config for injection
            model = AutoModelForCausalLM.from_pretrained(model_id)
            model = inject_adapter_in_model(config, model)
            sd_before = get_peft_model_state_dict(model)
            del model

            # remove q_proj of layer 5 from the PEFT config
            config.exclude_modules = ["model.decoder.layers.5.self_attn.q_proj"]

            model = AutoModelForCausalLM.from_pretrained(model_id)
            msg = re.escape(
                "While injecting the PEFT adapters, an inconsistency was discovered between the PEFT config and "
                "the provided state_dict. This is not necessarily an issue and can be ignored if this was the "
                "intent. The state_dict contained these additional target modules: "
                "['model.decoder.layers.5.self_attn.q_proj']. "
            )

            with pytest.warns(RuntimeWarning, match=msg):
                model = inject_adapter_in_model(config, model, state_dict=sd_before, low_cpu_mem_usage=True)

            # besides the warning, the rest of the injection should work
            sd_after = get_peft_model_state_dict(model)
            assert len(sd_before) > 0
            assert sd_before.keys() == sd_after.keys()
            for key in sd_before.keys():
                assert sd_before[key].shape == sd_after[key].shape


class TestPeftStateDict:
    # Test some edge cases around getting and setting the PEFT state_dict. There are potential sources of errors there
    # because the adapter_name is removed from/added to the state_dict keys.
    def test_get_peft_model_state_dict_removes_adapter_name(self):
        # ensure that the adapter name, "default", is removed from the state_dict
        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id)

        # note: lora targets q_proj and v_proj; add in an auxiliary module for good measure
        model = get_peft_model(model, LoraConfig(modules_to_save=["lm_head"]))
        sd = get_peft_model_state_dict(model)
        assert len(sd) > 1  # sanity check
        assert not any("default" in key for key in sd)

    def test_get_peft_model_state_dict_removes_non_defaul_adapter_name(self):
        # ensure that the adapter name is removed from the state_dict, even if it's not "default"
        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id)

        model = get_peft_model(model, LoraConfig(modules_to_save=["lm_head"]), adapter_name="other")
        sd = get_peft_model_state_dict(model, adapter_name="other")
        assert len(sd) > 1  # sanity check
        assert not any("other" in key for key in sd)

    def test_get_peft_model_state_dict_removes_adapter_name_when_same_as_module_name(self):
        # here the adapter is named "v_proj", which is the same name as some modules targeted with lora in the model,
        # which is nefarious
        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id)

        config = LoraConfig(modules_to_save=["lm_head"], target_modules=["v_proj"])
        model = get_peft_model(model, config, adapter_name="v_proj")
        sd = get_peft_model_state_dict(model, adapter_name="v_proj")
        assert len(sd) > 1  # sanity check
        for key in sd:
            # assert that the adapter_name was indeed removed
            assert not key.endswith("lora_A.v_proj.weight")
            assert not key.endswith("lora_B.v_proj.weight")
            assert not key.endswith("modules_to_save.v_proj.weight")
            # assert that the module name was not stripped completely from the key
            assert ("v_proj" in key) or ("q_proj" in key) or ("lm_head") in key

    def check_peft_model_weights_loaded_correctly(self, inner_model_cls, config, nested, adapter_name="default"):
        # Runs checks that a roundtrip of get_peft_model_state_dict and set_peft_model_state_dict results in the same
        # model (same outputs and same weights).
        class Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = inner_model_cls()

            def forward(self, x):
                return self.inner(x)

        if nested:
            # add another layer of nesting
            model_cls = Outer
        else:
            model_cls = inner_model_cls

        x = torch.randn(1, 5)

        torch.manual_seed(0)
        base_model = model_cls()
        with torch.inference_mode():
            base_out = base_model(x)

        torch.manual_seed(42)
        model = get_peft_model(base_model, config, adapter_name=adapter_name)
        with torch.inference_mode():
            peft_out = model(x)
        # sanity check: peft adapter has an effect
        assert not torch.allclose(base_out, peft_out, atol=1e-6)

        sd = get_peft_model_state_dict(model, adapter_name=adapter_name)

        torch.manual_seed(0)
        base_model = model_cls()
        torch.manual_seed(42 + 1)  # ensure we start with a different, randomly initialized PEFT model
        model_new = get_peft_model(base_model, config, adapter_name=adapter_name)
        with torch.inference_mode():
            peft_new = model_new(x)
        assert not torch.allclose(peft_out, peft_new, atol=1e-6)

        set_peft_model_state_dict(model_new, sd, adapter_name=adapter_name)
        with torch.inference_mode():
            peft_out_loaded = model_new(x)
        assert torch.allclose(peft_out, peft_out_loaded, atol=1e-6)

        sd_new = get_peft_model_state_dict(model, adapter_name=adapter_name)
        assert sd.keys() == sd_new.keys()
        for key, val in sd.items():
            val_new = sd_new[key]
            torch.allclose(val, val_new)

    @pytest.mark.parametrize("nested", [False, True])
    def test_get_and_set_peft_model_state_dict_normal_names(self, nested):
        # In this test, there is no edge case. Therefore, this test is basically the "control group" for the subsequent
        # tests (if this test were to fail, it means the testing code itself is wrong).
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.foo_linear = nn.Linear(5, 5)
                self.foo_baz = nn.Linear(5, 5)
                self.baz_foo = nn.Linear(5, 5)
                self.foo_baz_foo = nn.Linear(5, 5)
                self.baz_foo_baz = nn.Linear(5, 5)

            def forward(self, x):
                x = self.foo_linear(x)
                x = self.foo_baz(x)
                x = self.baz_foo(x)
                x = self.foo_baz_foo(x)
                x = self.baz_foo_baz(x)
                return x

        config = LoraConfig(
            target_modules=["foo_linear", "foo_baz", "baz_foo", "foo_baz_foo", "baz_foo_baz"], init_lora_weights=False
        )
        self.check_peft_model_weights_loaded_correctly(MyModel, config, nested=nested)

    @pytest.mark.parametrize("nested", [False, True])
    def test_get_and_set_peft_model_state_dict_peft_prefix_in_module_name(self, nested):
        # Here we have a model with some modules containing "lora" in their name.
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.foo_linear = nn.Linear(5, 5)
                self.foo_lora = nn.Linear(5, 5)
                self.lora_foo = nn.Linear(5, 5)
                self.foo_lora_foo = nn.Linear(5, 5)
                self.lora_foo_lora = nn.Linear(5, 5)

            def forward(self, x):
                x = self.foo_linear(x)
                x = self.foo_lora(x)
                x = self.lora_foo(x)
                x = self.foo_lora_foo(x)
                x = self.lora_foo_lora(x)
                return x

        config = LoraConfig(
            target_modules=["foo_linear", "foo_lora", "lora_foo", "foo_lora_foo", "lora_foo_lora"],
            init_lora_weights=False,
        )
        self.check_peft_model_weights_loaded_correctly(MyModel, config, nested=nested)

    @pytest.mark.parametrize("nested", [False, True])
    def test_get_and_set_peft_model_state_dict_weight_in_module_name(self, nested):
        # Here we have a model with some modules containing "weight" in their name.
        # See #2772
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.foo_linear = nn.Linear(5, 5)
                self.foo_weight = nn.Linear(5, 5)
                self.weight_foo = nn.Linear(5, 5)
                self.foo_weight_foo = nn.Linear(5, 5)
                self.weight_foo_weight = nn.Linear(5, 5)

            def forward(self, x):
                x = self.foo_linear(x)
                x = self.foo_weight(x)
                x = self.weight_foo(x)
                x = self.foo_weight_foo(x)
                x = self.weight_foo_weight(x)
                return x

        config = LoraConfig(
            target_modules=["foo_linear", "foo_weight", "weight_foo", "foo_weight_foo", "weight_foo_weight"],
            init_lora_weights=False,
        )
        self.check_peft_model_weights_loaded_correctly(MyModel, config, nested=nested)

    @pytest.mark.parametrize("nested", [False, True])
    def test_get_and_set_peft_model_state_dict_bias_in_module_name(self, nested):
        # Here we have a model with some modules containing "bias" in their name.
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.foo_linear = nn.Linear(5, 5)
                self.foo_bias = nn.Linear(5, 5)
                self.bias_foo = nn.Linear(5, 5)
                self.foo_bias_foo = nn.Linear(5, 5)
                self.bias_foo_bias = nn.Linear(5, 5)

            def forward(self, x):
                x = self.foo_linear(x)
                x = self.foo_bias(x)
                x = self.bias_foo(x)
                x = self.foo_bias_foo(x)
                x = self.bias_foo_bias(x)
                return x

        config = LoraConfig(
            target_modules=["foo_linear", "foo_bias", "bias_foo", "foo_bias_foo", "bias_foo_bias"],
            init_lora_weights=False,
            bias="lora_only",
        )
        self.check_peft_model_weights_loaded_correctly(MyModel, config, nested=nested)

    @pytest.mark.parametrize("nested", [False, True])
    def test_get_and_set_peft_model_state_dict_adapter_name_same_as_module_name(self, nested):
        # Here we choose a module name that is identical to the name of one of the adapters.
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = nn.Linear(5, 5)
                self.foo_baz = nn.Linear(5, 5)
                self.baz_foo = nn.Linear(5, 5)
                self.foo_baz_foo = nn.Linear(5, 5)
                self.baz_foo_baz = nn.Linear(5, 5)

            def forward(self, x):
                x = self.foo(x)
                x = self.foo_baz(x)
                x = self.baz_foo(x)
                x = self.foo_baz_foo(x)
                x = self.baz_foo_baz(x)
                return x

        config = LoraConfig(
            target_modules=["foo", "foo_baz", "baz_foo", "foo_baz_foo", "baz_foo_baz"], init_lora_weights=False
        )
        self.check_peft_model_weights_loaded_correctly(MyModel, config, nested=nested, adapter_name="foo")
