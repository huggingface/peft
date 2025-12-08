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
import re

import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM

from peft import (
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

    This is mainly testing with LoKr, as it would be wasteful to test with all compatible PEFT methods.

    We mainly use convert_to_lora and not save_as_lora here, as is just a thin wrapper around convert_to_lora and
    involves disk IO, which we want to avoid as much as possible. For most users, save_as_lora will most likely be the
    main entry point,

    """

    model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
    torch_device = infer_device()
    with hub_online_once(model_id):
        base_model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)

    def get_base_model(self):
        return copy.deepcopy(self.base_model)

    @pytest.fixture
    def lokr_model(self):
        torch.manual_seed(0)
        return get_peft_model(self.get_base_model(), LoKrConfig(init_weights=False))

    def test_no_peft_layer_raises(self):
        # Model without any PEFT layer should raise
        base_model = self.get_base_model()
        msg = "Could not detect any layer that supports LoRA conversion"
        with pytest.raises(TypeError, match=msg):
            convert_to_lora(base_model, rank=8)

    def test_prompt_learning_model_raises(self):
        # Prefix Tuning does not support LoRA conversion
        base_model = self.get_base_model()
        prefix_model = get_peft_model(base_model, PrefixTuningConfig(num_virtual_tokens=10, task_type="CAUSAL_LM"))
        msg = "Could not detect any layer that supports LoRA conversion"
        with pytest.raises(TypeError, match=msg):
            convert_to_lora(prefix_model, rank=8)

    def test_peft_model_but_no_support_raises(self):
        # IA3 has BaseTunerLayers but does not support LoRA conversion
        base_model = self.get_base_model()
        ia3_model = get_peft_model(base_model, IA3Config())
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

        lokr_model = get_peft_model(MyModule(), LoKrConfig(target_modules=["conv", "lin"]))
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
        new_lora_model = get_peft_model(base_model, lora_config)
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
        msg = re.escape("The chosen rank 123 is larger then the weight shape (16), please choose a lower rank")
        with pytest.raises(ValueError, match=msg):
            convert_to_lora(lokr_model, rank=123)

    def test_converting_transformers_model_works(self, lokr_model, tmp_path):
        # test that we can convert a transformers model that has loaded LoKr directly
        inputs = torch.arange(10).view(1, -1).to(self.torch_device)
        with torch.inference_mode():
            output_lokr = lokr_model(inputs).logits

        lokr_model.save_pretrained(tmp_path)
        # load directly with transformers
        loaded_model = AutoModelForCausalLM.from_pretrained(tmp_path).to(self.torch_device)
        with torch.inference_mode():
            output_loaded = loaded_model(inputs).logits

        # sanity check
        atol, rtol = 1e-4, 1e-4
        assert torch.allclose(output_lokr, output_loaded, atol=atol, rtol=rtol)

        save_as_lora(tmp_path / "converted", lokr_model, rank=8)
        lora_model = AutoModelForCausalLM.from_pretrained(tmp_path / "converted").to(self.torch_device)

        # With from_pretrained, we don't get a load_result and thus cannot check for missing keys. As a proxy,
        # instead check that no LoRA weight is all zeros (which would indicate a missing weight)
        for name, param in lora_model.named_parameters():
            if (".lora_A" in name) or (".lora_B" in name):
                assert not torch.all(param == 0)

        with torch.inference_mode():
            output_converted = lora_model(inputs).logits
        corr_converted = torch.corrcoef(torch.stack((output_lokr.flatten(), output_converted.flatten())))[0, 1]
        # the converted LoRA's correlation should be very high
        assert corr_converted > 0.9

    def test_converted_lora_approximates_original_adapter(self, lokr_model):
        inputs = torch.arange(10).view(1, -1).to(self.torch_device)
        with torch.inference_mode():
            with lokr_model.disable_adapter():
                output_base = lokr_model(inputs).logits
            output_lokr = lokr_model(inputs).logits

        # sanity check
        atol, rtol = 1e-4, 1e-4
        assert not torch.allclose(output_base, output_lokr, atol=atol, rtol=rtol)

        ##############
        # fixed rank #
        ##############

        lora_config, state_dict = convert_to_lora(lokr_model, rank=8)
        base_model = self.get_base_model()
        lora_model = get_peft_model(base_model, lora_config)

        # by default, the LoRA model should be an identity transform
        with torch.inference_mode():
            output_lora = lora_model(inputs).logits
        assert torch.allclose(output_base, output_lora, atol=atol, rtol=rtol)

        # load the converted LoRA weights
        load_result = set_peft_model_state_dict(lora_model, state_dict)
        assert not load_result.unexpected_keys

        # sanity check the number of trainable parameters
        num_train_params, total_params = lora_model.get_nb_trainable_parameters()
        assert 100 < num_train_params < 0.1 * total_params

        with torch.inference_mode():
            output_converted = lora_model(inputs).logits

        # note the corr coeff matrix is 2x2, we want the off-diagonal entry
        corr_lora = torch.corrcoef(torch.stack((output_lokr.flatten(), output_lora.flatten())))[0, 1]
        corr_converted = torch.corrcoef(torch.stack((output_lokr.flatten(), output_converted.flatten())))[0, 1]

        # sanity check: the base LoRA's correlation should not be too high
        assert corr_lora < 0.8
        # the converted LoRA's correlation should be very high
        assert corr_converted > 0.9

        ###############################
        # this time with dynamic rank #
        ###############################

        lora_config, state_dict = convert_to_lora(lokr_model, rank=0.9)
        base_model = self.get_base_model()
        lora_model = get_peft_model(base_model, lora_config)
        load_result = set_peft_model_state_dict(lora_model, state_dict)
        assert not load_result.unexpected_keys

        # sanity check the number of trainable parameters
        num_train_params, total_params = lora_model.get_nb_trainable_parameters()
        assert 100 < num_train_params < 0.1 * total_params

        with torch.inference_mode():
            output_converted = lora_model(inputs).logits
        corr_converted = torch.corrcoef(torch.stack((output_lokr.flatten(), output_converted.flatten())))[0, 1]
        assert corr_converted > 0.9

    def test_with_tqdm_works(self, lokr_model, capsys):
        # pass progressbar=True to use tqdm
        convert_to_lora(lokr_model, rank=8, progressbar=True)
        captured = capsys.readouterr()
        assert "Converting to LoRA" in captured.err

    def test_save_as_lora(self, lokr_model, tmp_path):
        inputs = torch.arange(10).view(1, -1).to(self.torch_device)
        atol, rtol = 1e-4, 1e-4

        lora_config, state_dict = convert_to_lora(lokr_model, rank=8)
        base_model = self.get_base_model()
        lora_model = get_peft_model(base_model, lora_config)
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
            output_lokr = lokr_model(inputs).logits

        # remove the PeftModel wrapper and the peft_config attribute -- this should still work
        unwrapped_lokr_model = unwrap(lokr_model)
        lora_config, state_dict = convert_to_lora(unwrapped_lokr_model, rank=8)

        base_model = self.get_base_model()
        lora_model = get_peft_model(base_model, lora_config)
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
            output_converted = lora_model(inputs).logits

        corr_converted = torch.corrcoef(torch.stack((output_lokr.flatten(), output_converted.flatten())))[0, 1]
        # the converted LoRA's correlation should be very high
        assert corr_converted > 0.9
