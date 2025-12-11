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

import copy
import platform
import re

import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM

from peft import (
    C3AConfig,
    IA3Config,
    LoKrConfig,
    LoraConfig,
    PeftModel,
    PrefixTuningConfig,
    convert_to_lora,
    get_peft_model,
    get_peft_model_state_dict,
    save_as_lora,
    set_peft_model_state_dict,
)
from peft.utils import infer_device

from .testing_utils import hub_online_once


class TestLoraConversion:
    """Test functionality to convert non-LoRA adapters to LoRA adapters

    This is mainly testing with LoKr, as it would be wasteful to test with all compatible PEFT methods in detail. For a
    broad suite of tests across PEFT methods, check test_decoder_models.py::test_lora_conversion.

    We mainly use convert_to_lora and not save_as_lora here, as is just a thin wrapper around convert_to_lora and
    involves disk IO, which we want to avoid as much as possible. For most users, save_as_lora will most likely be the
    main entry point,

    For comparing outputs, it's not ideal to check the logits, as most of them are close to zero and we cannot use
    torch.allclose, as a certain deviation is expected from conversion. A robust way would be to check the hidden
    states after subtracting the base model's hidden states (since the contribution of the adapter is what we want to
    compare).
    """

    model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
    torch_device = infer_device()
    base_model = None

    def get_base_model(self):
        if self.base_model is None:
            with hub_online_once(self.model_id):
                self.base_model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.torch_device)
        return copy.deepcopy(self.base_model)

    @pytest.fixture
    def lokr_model(self):
        torch.manual_seed(0)
        return get_peft_model(self.get_base_model(), LoKrConfig(init_weights=False))

    @staticmethod
    def get_mse(output1, output2):
        return nn.functional.mse_loss(output1.hidden_states[-1], output2.hidden_states[-1]).item()

    def test_no_peft_layer_raises(self):
        # Model without any PEFT layer should raise
        base_model = self.get_base_model()
        msg = "Could not detect any layer that supports LoRA conversion"
        with pytest.raises(TypeError, match=msg):
            convert_to_lora(base_model, rank=8)

    def test_prompt_learning_model_raises(self):
        # Prefix Tuning does not support LoRA conversion
        base_model = self.get_base_model()
        config = PrefixTuningConfig(num_virtual_tokens=10, task_type="CAUSAL_LM")
        prefix_model = get_peft_model(base_model, config).eval()
        assert not prefix_model.supports_lora_conversion()

        msg = "Could not detect any layer that supports LoRA conversion"
        with pytest.raises(TypeError, match=msg):
            convert_to_lora(prefix_model, rank=8)

    def test_peft_model_but_no_support_raises(self):
        # IA3 has BaseTunerLayers but does not support LoRA conversion
        base_model = self.get_base_model()
        ia3_model = get_peft_model(base_model, IA3Config()).eval()
        assert not ia3_model.supports_lora_conversion()

        msg = "Some module types on this model do not support LoRA conversion"
        with pytest.raises(TypeError, match=msg):
            convert_to_lora(ia3_model, rank=8)

    def test_model_with_unsupported_layers_raises(self):
        # conv layers do not support LoRA conversion (yet)
        # note: change this test if we add support for conv layer conversion
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(16, 16, 3)
                self.lin = nn.Linear(16, 16)

        lokr_model = get_peft_model(MyModule(), LoKrConfig(target_modules=["conv", "lin"])).eval()
        assert not lokr_model.supports_lora_conversion()

        msg = "Some module types on this model do not support LoRA conversion"
        with pytest.raises(TypeError, match=msg):
            convert_to_lora(lokr_model, rank=8)

    def test_targeted_modules_identical(self, lokr_model):
        lora_config, lora_state_dict = convert_to_lora(lokr_model, rank=8)
        lokr_state_dict = lokr_model.state_dict()

        # LoRA should have an entry for each layer targeted by LoKr
        # cut off parameter name and PEFT method specific part of the name to obtain module name
        modules_lokr = {k.rsplit(".", 2)[0] for k in lokr_state_dict.keys() if ".lokr" in k}
        modules_lora = {k.rsplit(".", 2)[0] for k in lora_state_dict.keys() if ".lora" in k}
        assert modules_lokr == modules_lora

        # creating a new LoRA model based on the returned config should give the same state dict keys
        base_model = self.get_base_model()
        new_lora_model = get_peft_model(base_model, lora_config).eval()
        new_lora_state_dict = get_peft_model_state_dict(new_lora_model)
        assert lora_state_dict.keys() == new_lora_state_dict.keys()

    def test_targeted_modules_identical_target_modules_str(self):
        base_model = self.get_base_model()
        lokr_config = LoKrConfig(target_modules=r".*\.q_proj", r=16, init_weights=False)
        lokr_model = get_peft_model(base_model, lokr_config).eval()
        lora_config, lora_state_dict = convert_to_lora(lokr_model, rank=8)
        lokr_state_dict = lokr_model.state_dict()

        # LoRA should have an entry for each layer targeted by LoKr
        # cut off parameter name and PEFT method specific part of the name to obtain module name
        modules_lokr = {k.rsplit(".", 2)[0] for k in lokr_state_dict.keys() if ".lokr" in k}
        modules_lora = {k.rsplit(".", 2)[0] for k in lora_state_dict.keys() if ".lora" in k}
        assert modules_lokr == modules_lora

        # creating a new LoRA model based on the returned config should give the same state dict keys
        base_model = self.get_base_model()
        new_lora_model = get_peft_model(base_model, lora_config).eval()
        new_lora_state_dict = get_peft_model_state_dict(new_lora_model)
        assert lora_state_dict.keys() == new_lora_state_dict.keys()

    def test_fixed_rank_lora_config(self, lokr_model):
        # with a fixed rank, we expect target_modules to be set on the LoRA config but not rank_pattern, alpha_pattern
        lora_config, _ = convert_to_lora(lokr_model, rank=8)
        assert isinstance(lora_config, LoraConfig)
        assert lora_config.r == 8
        assert lora_config.lora_alpha == 8
        assert lora_config.target_modules
        assert not lora_config.rank_pattern
        assert not lora_config.alpha_pattern

    def test_dynamic_rank_lora_config(self, lokr_model):
        # with a dynmaic rank, we expect rank_pattern and alpha_pattern to be set
        lora_config, state_dict = convert_to_lora(lokr_model, rank=0.5)
        assert lora_config.r == 1  # dummy value
        assert lora_config.lora_alpha == 1  # dummy value
        assert lora_config.rank_pattern
        assert lora_config.alpha_pattern

        # rank and alpha are always the same, i.e. scaling is 1
        assert lora_config.rank_pattern == lora_config.alpha_pattern
        # for each module, two LoRA weights
        assert 2 * len(lora_config.rank_pattern) == len(state_dict)

    def test_dynamic_rank_1_lora_config(self, lokr_model):
        # with a dynmaic rank, we expect rank_pattern and alpha_pattern to be set
        lora_config, state_dict = convert_to_lora(lokr_model, rank=1.0)
        assert lora_config.r == 1  # dummy value
        assert lora_config.lora_alpha == 1  # dummy value
        assert lora_config.rank_pattern
        assert lora_config.alpha_pattern

        # rank and alpha are always the same, i.e. scaling is 1
        assert lora_config.rank_pattern == lora_config.alpha_pattern
        # for each module, two LoRA weights
        assert 2 * len(lora_config.rank_pattern) == len(state_dict)

    def test_threshold_wrong_value_raises(self, lokr_model):
        # if a threshold is used, it must be between 0 and 1
        msg = "If rank is a float, it is interpreted as a threshold. It must be between 0 and 1 but got 123.0"
        with pytest.raises(ValueError, match=msg):
            convert_to_lora(lokr_model, rank=123.0)

        msg = "If rank is a float, it is interpreted as a threshold. It must be between 0 and 1 but got -0.5"
        with pytest.raises(ValueError, match=msg):
            convert_to_lora(lokr_model, rank=-0.5)

    def test_rank_higher_than_weight_dim_raises(self, lokr_model):
        # if the requested rank is higher than the weight dimension, we should raise
        msg = re.escape("The chosen rank 123 is larger than the weight shape (16), please choose a lower rank")
        with pytest.raises(ValueError, match=msg):
            convert_to_lora(lokr_model, rank=123)

    def test_fixed_rank_0_raises(self, lokr_model):
        msg = "Passing a rank of 0 doesn't make sense"
        with pytest.raises(ValueError, match=msg):
            convert_to_lora(lokr_model, rank=0)

    def test_converting_transformers_model_works(self, lokr_model, tmp_path):
        # test that we can convert a transformers model that has loaded LoKr directly
        assert lokr_model.supports_lora_conversion()

        inputs = torch.arange(10).view(1, -1).to(self.torch_device)
        with torch.inference_mode():
            output_lokr = lokr_model(inputs, output_hidden_states=True)

        lokr_model.save_pretrained(tmp_path)
        # load directly with transformers
        loaded_model = AutoModelForCausalLM.from_pretrained(tmp_path).to(self.torch_device)
        with torch.inference_mode():
            output_loaded = loaded_model(inputs, output_hidden_states=True)

        # sanity check
        atol, rtol = 1e-4, 1e-4
        assert torch.allclose(output_lokr.logits, output_loaded.logits, atol=atol, rtol=rtol)

        save_as_lora(tmp_path / "converted", lokr_model, rank=8)
        lora_model = AutoModelForCausalLM.from_pretrained(tmp_path / "converted").to(self.torch_device)

        # With from_pretrained, we don't get a load_result and thus cannot check for missing keys. As a proxy,
        # instead check that no LoRA weight is all zeros (which would indicate a missing weight)
        for name, param in lora_model.named_parameters():
            if (".lora_A" in name) or (".lora_B" in name):
                assert not torch.all(param == 0)

        with torch.inference_mode():
            output_converted = lora_model(inputs, output_hidden_states=True)
        assert 0.0 < self.get_mse(output_converted, output_lokr) < 0.1

    def test_converted_lora_approximates_original_adapter(self, lokr_model):
        inputs = torch.arange(10).view(1, -1).to(self.torch_device)
        with torch.inference_mode():
            with lokr_model.disable_adapter():
                output_base = lokr_model(inputs, output_hidden_states=True)
            output_lokr = lokr_model(inputs, output_hidden_states=True)

        # sanity check
        atol, rtol = 1e-4, 1e-4
        assert not torch.allclose(output_base.logits, output_lokr.logits, atol=atol, rtol=rtol)

        ##############
        # fixed rank #
        ##############

        lora_config, state_dict = convert_to_lora(lokr_model, rank=8)
        base_model = self.get_base_model()
        lora_model = get_peft_model(base_model, lora_config).eval()

        # by default, the LoRA model should be an identity transform
        with torch.inference_mode():
            output_lora = lora_model(inputs, output_hidden_states=True)
        assert torch.allclose(output_base.logits, output_lora.logits, atol=atol, rtol=rtol)

        # load the converted LoRA weights
        load_result = set_peft_model_state_dict(lora_model, state_dict)
        assert not load_result.unexpected_keys

        # sanity check the number of trainable parameters
        num_train_params, total_params = lora_model.get_nb_trainable_parameters()
        assert 100 < num_train_params < 0.1 * total_params

        with torch.inference_mode():
            output_converted = lora_model(inputs, output_hidden_states=True)

        mse_lora = self.get_mse(output_lora, output_lokr)
        mse_converted = self.get_mse(output_converted, output_lokr)
        assert mse_lora > 0.5
        assert 0.0 < mse_converted < 0.1

        ###############################
        # this time with dynamic rank #
        ###############################

        lora_config, state_dict = convert_to_lora(lokr_model, rank=0.9)
        base_model = self.get_base_model()
        lora_model = get_peft_model(base_model, lora_config).eval()
        load_result = set_peft_model_state_dict(lora_model, state_dict)
        assert not load_result.unexpected_keys

        # sanity check the number of trainable parameters
        num_train_params, total_params = lora_model.get_nb_trainable_parameters()
        assert 100 < num_train_params < 0.1 * total_params

        with torch.inference_mode():
            output_converted = lora_model(inputs, output_hidden_states=True)
        mse_converted = self.get_mse(output_converted, output_lokr)
        assert 0.0 < mse_converted < 0.1

    def test_with_tqdm_works(self, lokr_model, capsys):
        # pass progressbar=True to use tqdm
        convert_to_lora(lokr_model, rank=8, progressbar=True)
        captured = capsys.readouterr()
        assert "Converting to LoRA" in captured.err

    def test_save_as_lora(self, lokr_model, tmp_path):
        # whether using save_as_lora gives the same result as convert_to_lora
        inputs = torch.arange(10).view(1, -1).to(self.torch_device)
        atol, rtol = 1e-4, 1e-4

        lora_config, state_dict = convert_to_lora(lokr_model, rank=8)
        base_model = self.get_base_model()
        lora_model = get_peft_model(base_model, lora_config).eval()
        load_result = set_peft_model_state_dict(lora_model, state_dict)
        assert not load_result.unexpected_keys

        with torch.inference_mode():
            output_before = lora_model(inputs).logits

        # test that save_as_lora works as expected
        save_as_lora(tmp_path, lokr_model, rank=8)
        base_model = self.get_base_model()
        loaded_model = PeftModel.from_pretrained(base_model, tmp_path).to(self.torch_device)

        with torch.inference_mode():
            output_after = loaded_model(inputs).logits

        assert torch.allclose(output_before, output_after, atol=atol, rtol=rtol)

    def test_model_without_peft_config(self, lokr_model):
        # Conversion also works with models that don't have a PeftConfig on them. This is a bit of a convoluted case,
        # but conversion doesn't strictly rely on an existing peft_config, so it should still work.
        def unwrap(peft_model):
            unwrapped = peft_model.get_base_model()
            del unwrapped.peft_config
            return unwrapped

        inputs = torch.arange(10).view(1, -1).to(self.torch_device)
        with torch.inference_mode():
            output_lokr = lokr_model(inputs, output_hidden_states=True)

        # remove the PeftModel wrapper and the peft_config attribute -- this should still work
        unwrapped_lokr_model = unwrap(lokr_model)
        lora_config, state_dict = convert_to_lora(unwrapped_lokr_model, rank=8)

        base_model = self.get_base_model()
        lora_model = get_peft_model(base_model, lora_config).eval()
        unwrapped_lora_model = unwrap(lora_model)

        # Note: On the unwrapped model, we cannot use set_peft_model_state_dict, as that requires a peft_config. Thus,
        # we need to manually inject the adapter name into state_dict keys, which is done automatically when using
        # set_peft_model_state_dict.
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace(".lora_A.weight", ".lora_A.default.weight")
            new_k = new_k.replace(".lora_B.weight", ".lora_B.default.weight")
            new_state_dict[new_k] = v

        load_result = unwrapped_lora_model.load_state_dict(new_state_dict, strict=False)
        assert not load_result.unexpected_keys

        with torch.inference_mode():
            output_converted = lora_model(inputs, output_hidden_states=True)

        mse = self.get_mse(output_converted, output_lokr)
        assert 0.0 < mse < 0.1

    def test_converted_lora_to_lora_works_and_warns(self):
        # In general, there is no need to convert LoRA to LoRA, but it should still work. One possible use case would be
        # to shrink the rank of an existing LoRA adapter. The resulting correlation in this test is surprisingly high,
        # probably because the initial LoRA was not trained but initialized with random weights.
        inputs = torch.arange(10).view(1, -1).to(self.torch_device)
        base_model = self.get_base_model()
        with torch.inference_mode():
            output_base = base_model(inputs, output_hidden_states=True)

        orig_lora_config = LoraConfig(r=16, init_lora_weights=False)
        orig_lora_model = get_peft_model(base_model, orig_lora_config).eval()

        with torch.inference_mode():
            output_orig_lora = orig_lora_model(inputs, output_hidden_states=True)

        # sanity check
        atol, rtol = 1e-4, 1e-4
        assert not torch.allclose(output_base.logits, output_orig_lora.logits)

        # convert from rank 16 to rank 8
        msg = "Converting a PEFT adapter to LoRA that is already a LoRA adapter"
        with pytest.warns(UserWarning, match=msg):
            # check that a warning was issued
            lora_config, state_dict = convert_to_lora(orig_lora_model, rank=8)

        base_model = self.get_base_model()
        lora_model = get_peft_model(base_model, lora_config).eval()

        # load the converted LoRA weights
        load_result = set_peft_model_state_dict(lora_model, state_dict)
        assert not load_result.unexpected_keys

        with torch.inference_mode():
            output_converted = lora_model(inputs, output_hidden_states=True)

        mse_converted = self.get_mse(output_converted, output_orig_lora)
        assert 0.0 < mse_converted < 0.1

    def test_converted_lora_with_multiple_adapters(self, lokr_model):
        # ensure that we can convert specific adapters when multiple are present
        lokr_config = LoKrConfig(r=16, init_weights=False)
        lokr_model.add_adapter("other", lokr_config)

        inputs = torch.arange(10).view(1, -1).to(self.torch_device)
        with torch.inference_mode():
            output_lokr_default = lokr_model(inputs, output_hidden_states=True)
            lokr_model.set_adapter("other")
            output_lokr_other = lokr_model(inputs, output_hidden_states=True)

        # sanity check
        atol, rtol = 1e-4, 1e-4
        assert not torch.allclose(output_lokr_default.logits, output_lokr_other.logits, atol=atol, rtol=rtol)

        # convert the default adapter
        lora_config_default, state_dict_default = convert_to_lora(lokr_model, rank=8)
        base_model = self.get_base_model()
        lora_model_default = get_peft_model(base_model, lora_config_default).eval()

        # load the converted LoRA weights for the default adapter
        load_result = set_peft_model_state_dict(lora_model_default, state_dict_default)
        assert not load_result.unexpected_keys
        with torch.inference_mode():
            output_converted_default = lora_model_default(inputs, output_hidden_states=True)

        # convert the other adapter
        lora_config_other, state_dict_other = convert_to_lora(lokr_model, rank=8, adapter_name="other")
        base_model = self.get_base_model()
        lora_model_other = get_peft_model(base_model, lora_config_other).eval()
        # load the converted LoRA weights for the other adapter
        load_result = set_peft_model_state_dict(lora_model_other, state_dict_other)
        assert not load_result.unexpected_keys
        with torch.inference_mode():
            output_converted_other = lora_model_other(inputs, output_hidden_states=True)

        mse_default_default = self.get_mse(output_converted_default, output_lokr_default)
        mse_other_other = self.get_mse(output_converted_other, output_lokr_other)
        mse_default_other = self.get_mse(output_converted_default, output_lokr_other)
        mse_other_default = self.get_mse(output_converted_other, output_lokr_default)

        assert 0.0 < mse_default_default < 0.1
        assert 0.0 < mse_other_other < 0.1
        assert mse_default_other > 0.5
        assert mse_other_default > 0.5

    def test_convert_model_with_modules_to_save(self):
        # If the original adapter defines modules_to_save, these need to be included in the LoRA adapter
        model = self.get_base_model()
        inputs = torch.arange(10).view(1, -1).to(self.torch_device)
        with torch.inference_mode():
            output_base = model(inputs, output_hidden_states=True)

        # lokr is initialized as identity transform to ensure that modules_to_save is the thing that impacts the output
        lokr_config = LoKrConfig(modules_to_save=["0.fc1"])
        lokr_model = get_peft_model(model, lokr_config)

        # ensure that the modules_to_save affects the output
        lokr_model.base_model.model.model.decoder.layers[0].fc1.modules_to_save.default.weight.data.mul_(-10.0)
        lokr_model.base_model.model.model.decoder.layers[0].fc1.modules_to_save.default.bias.data.mul_(-10.0)

        with torch.inference_mode():
            output_lokr = lokr_model(inputs, output_hidden_states=True)

        # sanity check
        atol, rtol = 1e-4, 1e-4
        assert not torch.allclose(output_base.logits, output_lokr.logits, atol=atol, rtol=rtol)

        lora_config, state_dict = convert_to_lora(lokr_model, rank=8)
        assert lora_config.modules_to_save == lokr_config.modules_to_save

        base_model = self.get_base_model()
        lora_model = get_peft_model(base_model, lora_config).eval()

        # load the converted LoRA weights
        load_result = set_peft_model_state_dict(lora_model, state_dict)
        assert not load_result.unexpected_keys

        with torch.inference_mode():
            output_converted = lora_model(inputs, output_hidden_states=True)

        mse_converted = self.get_mse(output_converted, output_lokr)
        # here we expect an actual loss of 0, since only the modules_to_save affect the result, and those are identical
        assert mse_converted == 0.0

    @pytest.mark.parametrize("bias", ["c3a_only", "all"])
    def test_convert_model_with_trainable_bias_raises(self, bias):
        # If the original adapter includes trainable bias terms, we raise. LoKr doesn't support this, so taking C3A
        model = self.get_base_model()
        inputs = torch.arange(10).view(1, -1).to(self.torch_device)

        c3a_config = C3AConfig(block_size=4, bias=bias)
        c3a_model = get_peft_model(model, c3a_config)

        msg = "The adapter's config sets bias"
        with pytest.raises(ValueError, match=msg):
            convert_to_lora(c3a_model, rank=8)

    @pytest.mark.skipif(platform.system() != "Linux", reason="Running test involving torch.compile only on Linux.")
    def test_with_torch_compile(self, lokr_model):
        # ensure that we can call lora conversion with compilation
        lora_config_no_comp, state_dict_no_comp = convert_to_lora(lokr_model, rank=8)
        lora_config_comp, state_dict_comp = convert_to_lora(
            lokr_model, rank=8, compile_kwargs={"mode": "max-autotune-no-cudagraphs"}
        )

        assert lora_config_no_comp.to_dict() == lora_config_comp.to_dict()
        assert state_dict_no_comp.keys() == state_dict_comp.keys()
        for key, weight_no_comp in state_dict_no_comp.items():
            weight_comp = state_dict_comp[key]
            assert torch.allclose(weight_comp, weight_no_comp)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_convert_float16_dtype(self, dtype):
        inputs = torch.arange(10).view(1, -1).to(self.torch_device)

        torch.manual_seed(0)
        base_model = self.get_base_model().to(dtype)
        with torch.inference_mode():
            output_base = base_model(inputs, output_hidden_states=True)

        # load a LoKr model with 16 bit precision
        lokr_model = get_peft_model(base_model, LoKrConfig(init_weights=False), autocast_adapter_dtype=False)

        with torch.inference_mode():
            output_lokr = lokr_model(inputs, output_hidden_states=True)

        # sanity check
        atol, rtol = 1e-4, 1e-4
        assert not torch.allclose(output_base.logits, output_lokr.logits, atol=atol, rtol=rtol)

        lora_config, state_dict = convert_to_lora(lokr_model, rank=8)
        for weight in state_dict.values():
            assert weight.dtype == dtype

        base_model = self.get_base_model().to(dtype)
        lora_model = get_peft_model(base_model, lora_config, autocast_adapter_dtype=False).eval()

        # load the converted LoRA weights
        load_result = set_peft_model_state_dict(lora_model, state_dict)
        assert not load_result.unexpected_keys

        with torch.inference_mode():
            output_converted = lora_model(inputs, output_hidden_states=True)

        mse_converted = self.get_mse(output_converted, output_lokr)
        assert 0.0 < mse_converted < 0.1
