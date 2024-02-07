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
import itertools
import os
import re
import tempfile
import unittest

import pytest
import torch
from parameterized import parameterized
from torch import nn
from transformers import AutoModelForCausalLM

from peft import (
    AdaLoraConfig,
    LoHaConfig,
    LoKrConfig,
    LoraConfig,
    OFTConfig,
    PeftMixedModel,
    PrefixTuningConfig,
    get_peft_model,
)
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import infer_device


class SimpleNet(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        # note: out_features must be > rank or else OFT will be an identity transform
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(20, 16, bias=bias)

    def forward(self, X):
        X = X.float()
        X = self.lin0(X)
        X = self.relu(X)
        X = self.lin1(X)
        return X


def _param_name_func(testcase_func, param_num, params):
    # for parameterized tests in TextMixedAdapterTypes
    config0, config1 = params[0]
    name0 = config0.__class__.__name__[: -len("Config")]
    name1 = config1.__class__.__name__[: -len("Config")]
    if name0 != name1:
        return f"{testcase_func.__name__}_{param_num}_{name0}_{name1}"
    return f"{testcase_func.__name__}_{param_num}_{name0}_x2"


class TestMixedAdapterTypes(unittest.TestCase):
    torch_device = infer_device()

    def _get_model(self, model_cls, peft_config=None, adapter_name=None, seed=0, mixed=True):
        torch.manual_seed(0)  # always use seed 0 for base model, seed for adapters may differ
        base_model = model_cls().eval().to(self.torch_device)
        if peft_config is None:
            return base_model

        torch.manual_seed(seed)
        assert adapter_name is not None
        peft_model = get_peft_model(base_model, peft_config, adapter_name=adapter_name, mixed=mixed)
        return peft_model.eval().to(self.torch_device)

    def _check_mixed_outputs(self, model_cls, config0, config1, input, *, is_commutative):
        # This test checks different combinations of adapter0, adapter1, or combinations of the two, and whether
        # outputs are the same/different, depending on context. If we pass is_commutative=True, it means that the order
        # of adapters does not matter, and we expect the same output regardless of the order in which adapters are
        # applied.
        # We have to very careful with resetting the random seed each time it is used, otherwise the adapters may be
        # initialized with different values, and the test will fail.

        atol = 1e-5
        rtol = 1e-5
        seed0 = 0
        seed1 = 1

        # base model
        base_model = self._get_model(model_cls)
        output_base = base_model(input)
        assert torch.isfinite(output_base).all()

        # adapter 0
        peft_model_0 = self._get_model(model_cls, config0, "adapter0", seed=seed0)
        output_config0 = peft_model_0(input)

        assert torch.isfinite(output_config0).all()
        assert not torch.allclose(output_base, output_config0, atol=atol, rtol=rtol)

        # adapter 1
        peft_model_1 = self._get_model(model_cls, config1, "adapter1", seed=seed1)
        output_config1 = peft_model_1(input)

        assert torch.isfinite(output_config1).all()
        assert not torch.allclose(output_base, output_config1, atol=atol, rtol=rtol)
        assert not torch.allclose(output_config0, output_config1, atol=atol, rtol=rtol)

        # adapter 0 + 1
        peft_model_01 = self._get_model(model_cls, config0, "adapter0", seed=seed0)
        torch.manual_seed(seed1)
        peft_model_01.add_adapter("adapter1", config1)
        peft_model_01.set_adapter(["adapter0", "adapter1"])
        output_mixed_01 = peft_model_01(input)

        # check the number of tuner layer types
        tuner_layers = [mod for mod in peft_model_01.modules() if isinstance(mod, BaseTunerLayer)]
        tuner_types = {type(tuner_layer) for tuner_layer in tuner_layers}
        if type(config0) == type(config1):
            assert len(tuner_types) == 1
        else:
            assert len(tuner_types) == 2

        assert peft_model_01.active_adapters == ["adapter0", "adapter1"]
        assert torch.isfinite(output_mixed_01).all()
        assert not torch.allclose(output_config0, output_mixed_01, atol=atol, rtol=rtol)
        assert not torch.allclose(output_config1, output_mixed_01, atol=atol, rtol=rtol)
        if is_commutative:
            delta0 = output_config0 - output_base
            delta1 = output_config1 - output_base
            delta_mixed_01 = output_mixed_01 - output_base
            assert torch.allclose((delta0 + delta1), delta_mixed_01, atol=atol, rtol=rtol)

        # adapter 1 + 0
        peft_model_10 = self._get_model(model_cls, config1, "adapter1", seed=seed1)
        torch.manual_seed(seed0)
        peft_model_10.add_adapter("adapter0", config0)
        peft_model_10.set_adapter(["adapter1", "adapter0"])
        output_mixed_10 = peft_model_10(input)

        # check the number of tuner layer types
        tuner_layers = [mod for mod in peft_model_10.modules() if isinstance(mod, BaseTunerLayer)]
        tuner_types = {type(tuner_layer) for tuner_layer in tuner_layers}
        if type(config0) == type(config1):
            assert len(tuner_types) == 1
        else:
            assert len(tuner_types) == 2

        assert peft_model_10.active_adapters == ["adapter1", "adapter0"]
        assert torch.isfinite(output_mixed_10).all()
        assert not torch.allclose(output_config0, output_mixed_10, atol=atol, rtol=rtol)
        assert not torch.allclose(output_config1, output_mixed_10, atol=atol, rtol=rtol)
        if is_commutative:
            assert torch.allclose(output_mixed_01, output_mixed_10, atol=atol, rtol=rtol)

        # turn around the order of the adapters of the 0 + 1 mixed model, should behave like the 0 + 1 mixed model
        peft_model_10.set_adapter(["adapter0", "adapter1"])
        output_mixed_reversed = peft_model_10(input)

        # check the number of tuner layer types
        tuner_layers = [mod for mod in peft_model_10.modules() if isinstance(mod, BaseTunerLayer)]
        tuner_types = {type(tuner_layer) for tuner_layer in tuner_layers}
        if type(config0) == type(config1):
            assert len(tuner_types) == 1
        else:
            assert len(tuner_types) == 2

        assert peft_model_10.active_adapters == ["adapter0", "adapter1"]
        assert torch.isfinite(output_mixed_reversed).all()
        assert not torch.allclose(output_mixed_reversed, output_config0, atol=atol, rtol=rtol)
        assert not torch.allclose(output_mixed_reversed, output_config1, atol=atol, rtol=rtol)
        if is_commutative:
            assert torch.allclose(output_mixed_reversed, output_mixed_01, atol=atol, rtol=rtol)
            assert torch.allclose(output_mixed_reversed, output_mixed_10, atol=atol, rtol=rtol)

    def _check_merging(self, model_cls, config0, config1, input):
        # Ensure that when merging mixed adapters, the result is the same as when applying the adapters separately.
        # Merging requires a bit higher tolerance for some adapters, which can also vary depending on CPU vs GPU.
        atol = 1e-4
        rtol = 1e-4
        seed0 = 0
        seed1 = 1

        # adapter 0 + 1
        peft_model_01 = self._get_model(model_cls, config0, "adapter0", seed=seed0)
        torch.manual_seed(seed1)
        peft_model_01.add_adapter("adapter1", config1)
        peft_model_01.set_adapter(["adapter0", "adapter1"])
        output_mixed_01 = peft_model_01(input)

        model_merged_01 = peft_model_01.merge_and_unload()
        output_merged_01 = model_merged_01(input)
        assert torch.allclose(output_mixed_01, output_merged_01, atol=atol, rtol=rtol)

        # adapter 1 + 0
        peft_model_10 = self._get_model(model_cls, config1, "adapter1", seed=seed1)
        torch.manual_seed(seed0)
        peft_model_10.add_adapter("adapter0", config0)
        peft_model_10.set_adapter(["adapter1", "adapter0"])
        output_mixed_10 = peft_model_10(input)

        model_merged_10 = peft_model_10.merge_and_unload()
        output_merged_10 = model_merged_10(input)
        assert torch.allclose(output_mixed_10, output_merged_10, atol=atol, rtol=rtol)

    def _check_unload(self, model_cls, config0, config1, input):
        # Ensure that we can unload the base model without merging
        atol = 1e-5
        rtol = 1e-5
        seed0 = 0
        seed1 = 1

        base_model = self._get_model(model_cls)
        output_base = base_model(input)

        # adapter 0 + 1
        peft_model_01 = self._get_model(model_cls, config0, "adapter0", seed=seed0)
        torch.manual_seed(seed1)
        peft_model_01.add_adapter("adapter1", config1)
        peft_model_01.set_adapter(["adapter0", "adapter1"])
        output_mixed = peft_model_01(input)

        # unload
        model_unloaded = peft_model_01.unload()
        output_unloaded = model_unloaded(input)

        assert not torch.allclose(output_mixed, output_unloaded, atol=atol, rtol=rtol)
        assert torch.allclose(output_base, output_unloaded, atol=atol, rtol=rtol)

    def _check_disable(self, model_cls, config0, config1, input):
        # Ensure that we can disable adapters
        atol = 1e-5
        rtol = 1e-5
        seed0 = 0
        seed1 = 1

        # base model
        base_model = self._get_model(model_cls)
        output_base = base_model(input)

        # adapter 0
        peft_model_0 = self._get_model(model_cls, config0, "adapter0", seed=seed0)
        output_config0 = peft_model_0(input)
        with peft_model_0.disable_adapter():
            output_disabled0 = peft_model_0(input)

        assert not torch.allclose(output_base, output_config0, atol=atol, rtol=rtol)
        assert torch.allclose(output_base, output_disabled0, atol=atol, rtol=rtol)

        # adapter 1
        peft_model_1 = self._get_model(model_cls, config1, "adapter1", seed=seed1)
        output_config1 = peft_model_1(input)
        with peft_model_1.disable_adapter():
            output_disabled1 = peft_model_1(input)

        assert not torch.allclose(output_base, output_config1, atol=atol, rtol=rtol)
        assert torch.allclose(output_base, output_disabled1, atol=atol, rtol=rtol)

        # adapter 0 + 1
        peft_model_01 = self._get_model(model_cls, config0, "adapter0", seed=seed0)
        torch.manual_seed(seed1)
        peft_model_01.add_adapter("adapter1", config1)
        peft_model_01.set_adapter(["adapter0", "adapter1"])
        output_mixed_01 = peft_model_01(input)
        with peft_model_01.disable_adapter():
            output_disabled01 = peft_model_01(input)

        assert not torch.allclose(output_base, output_mixed_01, atol=atol, rtol=rtol)
        assert torch.allclose(output_base, output_disabled01, atol=atol, rtol=rtol)

        # adapter 1 + 0
        peft_model_10 = self._get_model(model_cls, config1, "adapter1", seed=seed1)
        torch.manual_seed(seed0)
        peft_model_10.add_adapter("adapter0", config0)
        peft_model_10.set_adapter(["adapter1", "adapter0"])
        output_mixed_10 = peft_model_10(input)
        with peft_model_10.disable_adapter():
            output_disabled10 = peft_model_10(input)

        assert not torch.allclose(output_base, output_mixed_10, atol=atol, rtol=rtol)
        assert torch.allclose(output_base, output_disabled10, atol=atol, rtol=rtol)

    def _check_loading(self, model_cls, config0, config1, input, *, is_commutative):
        # Check that we can load two adapters into the same model
        # Note that we save the adapters using a normal PeftModel because PeftMixModel doesn't support saving yet
        atol = 1e-5
        rtol = 1e-5
        seed0 = 0
        seed1 = 1

        with tempfile.TemporaryDirectory() as tmp_dirname:
            # SAVING
            # adapter 0: note that we set mixed=False because mixed models don't support saving (yet)
            peft_model_0 = self._get_model(model_cls, config0, "adapter0", seed=seed0, mixed=False)
            output_config0 = peft_model_0(input)
            peft_model_0.save_pretrained(os.path.join(tmp_dirname, "adapter0"))

            # adapter 1: note that we set mixed=False because mixed models don't support saving (yet)
            peft_model_1 = self._get_model(model_cls, config1, "adapter1", seed=seed1, mixed=False)
            output_config1 = peft_model_1(input)
            peft_model_1.save_pretrained(os.path.join(tmp_dirname, "adapter1"))

            # adapter 0 + 1
            peft_model_01 = self._get_model(model_cls, config0, "adapter0", seed=seed0)
            torch.manual_seed(seed1)
            peft_model_01.add_adapter("adapter1", config1)
            peft_model_01.set_adapter(["adapter0", "adapter1"])
            output_mixed_01 = peft_model_01(input)

            # adapter 1 + 0
            peft_model_10 = self._get_model(model_cls, config1, "adapter1", seed=seed1)
            torch.manual_seed(seed0)
            peft_model_10.add_adapter("adapter0", config0)
            peft_model_10.set_adapter(["adapter1", "adapter0"])
            output_mixed_10 = peft_model_10(input)

            # LOADING
            # adapter 0
            base_model = self._get_model(model_cls)
            # Notes:
            # Path is tmp_dirname/adapter0/adapter0 because non-default adapters are saved in a subfolder.
            # As a sanity check, we should set a completely different seed here. That way, we ensure that the the
            # weights are not just randomly initialized exactly to the same values as before.
            torch.manual_seed(123456)
            peft_model_loaded0 = PeftMixedModel.from_pretrained(
                base_model, os.path.join(tmp_dirname, "adapter0", "adapter0"), "adapter0"
            )
            output_loaded0 = peft_model_loaded0(input)
            assert torch.allclose(output_config0, output_loaded0, atol=atol, rtol=rtol)

            # adapter 1
            base_model = self._get_model(model_cls)
            torch.manual_seed(654321)  # setting a completely different seed here should not affect the result
            peft_model_loaded1 = PeftMixedModel.from_pretrained(
                base_model, os.path.join(tmp_dirname, "adapter1", "adapter1"), "adapter1"
            )
            output_loaded1 = peft_model_loaded1(input)
            assert torch.allclose(output_config1, output_loaded1, atol=atol, rtol=rtol)

            # adapter 0 + 1
            base_model = self._get_model(model_cls)
            torch.manual_seed(97531)  # setting a completely different seed here should not affect the result
            peft_model_loaded_01 = PeftMixedModel.from_pretrained(
                base_model, os.path.join(tmp_dirname, "adapter0", "adapter0"), "adapter0"
            )
            peft_model_loaded_01.load_adapter(os.path.join(tmp_dirname, "adapter1", "adapter1"), "adapter1")
            # at this point, "adapter0" should still be active
            assert peft_model_loaded_01.active_adapters == ["adapter0"]
            output_loaded01_0 = peft_model_loaded_01(input)
            assert torch.allclose(output_config0, output_loaded01_0, atol=atol, rtol=rtol)
            # activate adapter1
            peft_model_loaded_01.set_adapter(["adapter1"])
            assert peft_model_loaded_01.active_adapters == ["adapter1"]
            output_loaded01_1 = peft_model_loaded_01(input)
            assert torch.allclose(output_config1, output_loaded01_1, atol=atol, rtol=rtol)
            # activate both adapters
            peft_model_loaded_01.set_adapter(["adapter0", "adapter1"])
            output_loaded01 = peft_model_loaded_01(input)
            assert torch.allclose(output_mixed_01, output_loaded01, atol=atol, rtol=rtol)

            # adapter 1 + 0
            base_model = self._get_model(model_cls)
            torch.manual_seed(445566)  # setting a completely different seed here should not affect the result
            peft_model_loaded_10 = PeftMixedModel.from_pretrained(
                base_model, os.path.join(tmp_dirname, "adapter1", "adapter1"), "adapter1"
            )
            peft_model_loaded_10.load_adapter(os.path.join(tmp_dirname, "adapter0", "adapter0"), "adapter0")
            # at this point, "adapter1" should still be active
            assert peft_model_loaded_10.active_adapters == ["adapter1"]
            output_loaded10_1 = peft_model_loaded_10(input)
            assert torch.allclose(output_config1, output_loaded10_1, atol=atol, rtol=rtol)
            # activate adapter1
            peft_model_loaded_10.set_adapter(["adapter0"])
            assert peft_model_loaded_10.active_adapters == ["adapter0"]
            output_loaded10_0 = peft_model_loaded_10(input)
            assert torch.allclose(output_config0, output_loaded10_0, atol=atol, rtol=rtol)
            # activate both adapters
            peft_model_loaded_10.set_adapter(["adapter1", "adapter0"])
            output_loaded10 = peft_model_loaded_10(input)
            assert torch.allclose(output_mixed_10, output_loaded10, atol=atol, rtol=rtol)

            if is_commutative:
                assert torch.allclose(output_loaded01, output_loaded10, atol=atol, rtol=rtol)
                assert torch.allclose(output_loaded10, output_mixed_01, atol=atol, rtol=rtol)

    @parameterized.expand(
        itertools.combinations(
            [
                LoraConfig(target_modules=["lin0"], init_lora_weights=False),
                LoHaConfig(target_modules=["lin0"], init_weights=False),
                LoKrConfig(target_modules=["lin0"], init_weights=False),
                AdaLoraConfig(target_modules=["lin0"], init_lora_weights=False),
                OFTConfig(target_modules=["lin0"], init_weights=False),
            ],
            r=2,
        ),
        name_func=_param_name_func,
    )
    def test_target_first_layer(self, config0, config1):
        input = torch.arange(90).reshape(9, 10).to(self.torch_device)
        self._check_mixed_outputs(SimpleNet, config0, config1, input, is_commutative=False)
        self._check_merging(SimpleNet, config0, config1, input)
        self._check_unload(SimpleNet, config0, config1, input)
        self._check_disable(SimpleNet, config1, config0, input)
        self._check_loading(SimpleNet, config0, config1, input, is_commutative=False)

    @parameterized.expand(
        itertools.combinations(
            [
                LoraConfig(target_modules=["lin1"], init_lora_weights=False),
                LoHaConfig(target_modules=["lin1"], init_weights=False),
                LoKrConfig(target_modules=["lin1"], init_weights=False),
                AdaLoraConfig(target_modules=["lin1"], init_lora_weights=False),
                OFTConfig(target_modules=["lin1"], init_weights=False),
            ],
            r=2,
        ),
        name_func=_param_name_func,
    )
    def test_target_last_layer(self, config0, config1):
        # We are targeting the last layer of the SimpleNet. Therefore, since the adapters only add their activations
        # to the output, the results should be commutative. This would *not* work if the adapters do something more
        # complex or if we target an earlier layer, because of the non-linearity would destroy the commutativity.
        input = torch.arange(90).reshape(9, 10).to(self.torch_device)
        # OFT is not commutative, as it's not a linear operation on the inputs
        is_commutative = not any(isinstance(config, OFTConfig) for config in [config0, config1])

        self._check_mixed_outputs(SimpleNet, config0, config1, input, is_commutative=is_commutative)
        self._check_merging(SimpleNet, config0, config1, input)
        self._check_unload(SimpleNet, config0, config1, input)
        self._check_disable(SimpleNet, config1, config0, input)
        self._check_loading(SimpleNet, config0, config1, input, is_commutative=is_commutative)

    @parameterized.expand(
        itertools.combinations(
            [
                LoraConfig(init_lora_weights=False),
                LoHaConfig(init_weights=False),
                LoKrConfig(init_weights=False),
                AdaLoraConfig(init_lora_weights=False),
                OFTConfig(init_weights=False),
            ],
            r=2,
        ),
        name_func=_param_name_func,
    )
    def test_target_different_layers(self, config0, config1):
        input = torch.arange(90).reshape(9, 10).to(self.torch_device)

        config0.target_modules = ["lin0"]
        config1.target_modules = ["lin1"]
        self._check_mixed_outputs(SimpleNet, config0, config1, input, is_commutative=False)
        self._check_merging(SimpleNet, config0, config1, input)
        self._check_unload(SimpleNet, config0, config1, input)
        self._check_disable(SimpleNet, config0, config1, input)
        self._check_loading(SimpleNet, config0, config1, input, is_commutative=False)

        # same, but switch target_modules around
        config0.target_modules = ["lin1"]
        config1.target_modules = ["lin0"]
        self._check_mixed_outputs(SimpleNet, config1, config0, input, is_commutative=False)
        self._check_merging(SimpleNet, config1, config0, input)
        self._check_unload(SimpleNet, config1, config0, input)
        self._check_disable(SimpleNet, config1, config0, input)
        self._check_loading(SimpleNet, config1, config0, input, is_commutative=False)

    @parameterized.expand(
        [
            (
                LoraConfig(target_modules=["lin1"], init_lora_weights=False),
                LoraConfig(target_modules=["lin1"], init_lora_weights=False),
            ),
            (
                LoHaConfig(target_modules=["lin1"], init_weights=False),
                LoHaConfig(target_modules=["lin1"], init_weights=False),
            ),
            (
                LoKrConfig(target_modules=["lin1"], init_weights=False),
                LoKrConfig(target_modules=["lin1"], init_weights=False),
            ),
            (
                AdaLoraConfig(target_modules=["lin1"], init_lora_weights=False),
                AdaLoraConfig(target_modules=["lin1"], init_lora_weights=False),
            ),
            (
                OFTConfig(target_modules=["lin1"], init_weights=False),
                OFTConfig(target_modules=["lin1"], init_weights=False),
            ),
        ],
        name_func=_param_name_func,
    )
    def test_target_last_layer_same_type(self, config0, config1):
        input = torch.arange(90).reshape(9, 10).to(self.torch_device)
        # OFT is not commutative, as it's not a linear operation on the inputs
        is_commutative = not any(isinstance(config, OFTConfig) for config in [config0, config1])

        self._check_mixed_outputs(SimpleNet, config0, config1, input, is_commutative=is_commutative)
        self._check_merging(SimpleNet, config0, config1, input)
        self._check_unload(SimpleNet, config0, config1, input)
        self._check_disable(SimpleNet, config1, config0, input)

    @parameterized.expand(
        [
            (
                LoraConfig(target_modules=["lin0"], init_lora_weights=False),
                LoraConfig(target_modules=["lin0"], init_lora_weights=False),
            ),
            (
                LoHaConfig(target_modules=["lin0"], init_weights=False),
                LoHaConfig(target_modules=["lin0"], init_weights=False),
            ),
            (
                LoKrConfig(target_modules=["lin0"], init_weights=False),
                LoKrConfig(target_modules=["lin0"], init_weights=False),
            ),
            (
                AdaLoraConfig(target_modules=["lin0"], init_lora_weights=False),
                AdaLoraConfig(target_modules=["lin0"], init_lora_weights=False),
            ),
            (
                OFTConfig(target_modules=["lin0"], init_weights=False),
                OFTConfig(target_modules=["lin0"], init_weights=False),
            ),
        ],
        name_func=_param_name_func,
    )
    def test_target_first_layer_same_type(self, config0, config1):
        input = torch.arange(90).reshape(9, 10).to(self.torch_device)
        self._check_mixed_outputs(SimpleNet, config0, config1, input, is_commutative=False)
        self._check_merging(SimpleNet, config0, config1, input)
        self._check_unload(SimpleNet, config0, config1, input)
        self._check_disable(SimpleNet, config1, config0, input)
        self._check_loading(SimpleNet, config0, config1, input, is_commutative=False)

    def test_deeply_nested(self):
        # a somewhat absurdly nested model using different adapter types
        atol = 1e-5
        rtol = 1e-5
        torch.manual_seed(0)

        model = SimpleNet().eval().to(self.torch_device)
        input = torch.arange(90).reshape(9, 10).to(self.torch_device)
        output_base = model(input)

        config0 = LoraConfig(r=4, lora_alpha=4, target_modules=["lin0", "lin1"], init_lora_weights=False)
        peft_model = get_peft_model(model, config0, "adapter0", mixed=True)

        config1 = LoHaConfig(r=4, alpha=4, target_modules=["lin0"], init_weights=False)
        peft_model.add_adapter("adapter1", config1)

        config2 = AdaLoraConfig(r=4, lora_alpha=4, target_modules=["lin1"], init_lora_weights=False)
        peft_model.add_adapter("adapter2", config2)

        config3 = LoKrConfig(r=4, alpha=4, target_modules=["lin0", "lin1"], init_weights=False)
        peft_model.add_adapter("adapter3", config3)

        config4 = OFTConfig(r=8, target_modules=["lin0", "lin1"], init_weights=False)
        peft_model.add_adapter("adapter4", config4)

        peft_model.set_adapter(["adapter0", "adapter1", "adapter2", "adapter3", "adapter4"])
        output_mixed = peft_model(input)
        assert torch.isfinite(output_base).all()
        assert not torch.allclose(output_base, output_mixed, atol=atol, rtol=rtol)

        # test disabling all adapters
        with peft_model.disable_adapter():
            output_disabled = peft_model(input)
        assert torch.isfinite(output_disabled).all()
        assert torch.allclose(output_base, output_disabled, atol=atol, rtol=rtol)
        assert not torch.allclose(output_mixed, output_disabled, atol=atol, rtol=rtol)

        # merge and unload all adapters
        model_copy = copy.deepcopy(peft_model)
        model = model_copy.merge_and_unload()
        output_merged = model(input)
        assert torch.isfinite(output_merged).all()
        assert torch.allclose(output_mixed, output_merged, atol=atol, rtol=rtol)

        # merge and unload only adapter1 and adapter3
        model_copy = copy.deepcopy(peft_model)
        model_copy.set_adapter(["adapter1", "adapter3"])
        output_13 = model_copy(input)
        assert torch.isfinite(output_13).all()
        assert not torch.allclose(output_mixed, output_13, atol=atol, rtol=rtol)

        model_copy.set_adapter(["adapter0", "adapter1", "adapter2", "adapter3", "adapter4"])
        model_merged_unloaded = model_copy.merge_and_unload(adapter_names=["adapter1", "adapter3"])
        output_merged_13 = model_merged_unloaded(input)
        assert torch.isfinite(output_merged_13).all()
        assert torch.allclose(output_13, output_merged_13, atol=atol, rtol=rtol)

        # test unloading
        model_copy = copy.deepcopy(peft_model)
        model_unloaded = model_copy.unload()
        output_unloaded = model_unloaded(input)
        assert torch.isfinite(output_unloaded).all()
        assert torch.allclose(output_base, output_unloaded, atol=atol, rtol=rtol)

    def test_delete_adapter(self):
        atol = 1e-5
        rtol = 1e-5
        torch.manual_seed(0)

        model = SimpleNet().eval().to(self.torch_device)
        input = torch.arange(90).reshape(9, 10).to(self.torch_device)
        output_base = model(input)

        # create adapter0
        torch.manual_seed(0)
        config0 = LoraConfig(r=4, lora_alpha=4, target_modules=["lin0", "lin1"], init_lora_weights=False)
        peft_model = get_peft_model(model, config0, "adapter0", mixed=True)
        output_0 = peft_model(input)
        assert not torch.allclose(output_base, output_0, atol=atol, rtol=rtol)

        # add adapter1
        torch.manual_seed(1)
        config1 = LoHaConfig(r=4, alpha=4, target_modules=["lin0"], init_weights=False)
        peft_model.add_adapter("adapter1", config1)
        peft_model.set_adapter(["adapter0", "adapter1"])
        output_01 = peft_model(input)
        assert not torch.allclose(output_base, output_01, atol=atol, rtol=rtol)
        assert not torch.allclose(output_0, output_01, atol=atol, rtol=rtol)

        # delete adapter1
        peft_model.delete_adapter("adapter1")
        assert peft_model.active_adapters == ["adapter0"]
        output_deleted_1 = peft_model(input)
        assert torch.allclose(output_0, output_deleted_1, atol=atol, rtol=rtol)

        msg = re.escape("Adapter(s) ['adapter1'] not found, available adapters: ['adapter0']")
        with pytest.raises(ValueError, match=msg):
            peft_model.set_adapter(["adapter0", "adapter1"])

        # re-add adapter1
        torch.manual_seed(1)
        peft_model.add_adapter("adapter1", config1)
        peft_model.set_adapter(["adapter0", "adapter1"])
        output_01_readded = peft_model(input)
        assert not torch.allclose(output_base, output_01_readded, atol=atol, rtol=rtol)

        # same as above, but this time delete adapter0 first
        torch.manual_seed(0)
        model = SimpleNet().eval().to(self.torch_device)
        torch.manual_seed(0)
        peft_model = get_peft_model(model, config0, "adapter0", mixed=True)
        torch.manual_seed(1)
        peft_model.add_adapter("adapter1", config1)
        peft_model.delete_adapter("adapter0")
        assert peft_model.active_adapters == ["adapter1"]
        output_deleted_0 = peft_model(input)
        assert not torch.allclose(output_deleted_0, output_base, atol=atol, rtol=rtol)
        assert not torch.allclose(output_deleted_0, output_01, atol=atol, rtol=rtol)

        msg = re.escape("Adapter(s) ['adapter0'] not found, available adapters: ['adapter1']")
        with pytest.raises(ValueError, match=msg):
            peft_model.set_adapter(["adapter0", "adapter1"])

        peft_model.delete_adapter("adapter1")
        assert peft_model.active_adapters == []
        output_deleted_01 = peft_model(input)
        assert torch.allclose(output_deleted_01, output_base, atol=atol, rtol=rtol)

    def test_modules_to_save(self):
        model = SimpleNet().eval().to(self.torch_device)
        config0 = LoraConfig(target_modules=["lin0"], modules_to_save=["lin1"])
        peft_model = get_peft_model(model, config0, "adapter0", mixed=True)

        # adding a second adapter with same modules_to_save is not allowed
        # TODO: theoretically, we could allow this if it's the same target layer
        config1 = LoHaConfig(target_modules=["lin0"], modules_to_save=["lin1"])
        peft_model.add_adapter("adapter1", config1)
        with pytest.raises(ValueError, match="Only one adapter can be set at a time for modules_to_save"):
            peft_model.set_adapter(["adapter0", "adapter1"])

    def test_get_nb_trainable_parameters(self):
        model = SimpleNet().eval().to(self.torch_device)
        params_base = sum(p.numel() for p in model.parameters())

        config0 = LoraConfig(target_modules=["lin0"])
        peft_model = get_peft_model(model, config0, "adapter0", mixed=True)
        trainable_params0, all_param0 = peft_model.get_nb_trainable_parameters()

        params_lora = sum(p.numel() for n, p in model.named_parameters() if "adapter0" in n)
        assert trainable_params0 == params_lora
        assert all_param0 == (params_base + params_lora)

        config1 = LoHaConfig(target_modules=["lin1"])
        peft_model.add_adapter("adapter1", config1)
        peft_model.set_adapter(["adapter0", "adapter1"])
        params_loha = sum(p.numel() for n, p in model.named_parameters() if "adapter1" in n)
        trainable_params1, all_param1 = peft_model.get_nb_trainable_parameters()
        assert trainable_params1 == (params_lora + params_loha)
        assert all_param1 == ((params_base + params_lora) + params_loha)

        config2 = AdaLoraConfig(target_modules=["lin0", "lin1"])
        peft_model.add_adapter("adapter2", config2)
        peft_model.set_adapter(["adapter0", "adapter1", "adapter2"])
        params_adalora = sum(p.numel() for n, p in model.named_parameters() if "adapter2" in n)
        trainable_params2, all_param2 = peft_model.get_nb_trainable_parameters()
        # remove 2 params because we need to exclude "ranknum" for AdaLora trainable params
        assert trainable_params2 == (((params_lora + params_loha) + params_adalora) - 2)
        assert all_param2 == (((params_base + params_lora) + params_loha) + params_adalora)

    def test_incompatible_config_raises(self):
        model = SimpleNet().eval().to(self.torch_device)
        config0 = LoraConfig(target_modules=["lin0"])
        peft_model = get_peft_model(model, config0, "adapter0", mixed=True)

        config1 = PrefixTuningConfig()
        msg = "The provided `peft_type` 'PREFIX_TUNING' is not compatible with the `PeftMixedModel`."
        with pytest.raises(ValueError, match=msg):
            peft_model.add_adapter("adapter1", config1)

    def test_decoder_model(self):
        # test a somewhat realistic model instead of a toy model
        torch.manual_seed(0)

        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).eval().to(self.torch_device)
        input_ids = torch.tensor([[1, 1, 1], [1, 2, 1]]).to(self.torch_device)
        attention_mask = torch.tensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)
        input_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        output_base = model.generate(**input_dict)

        torch.manual_seed(0)
        config0 = LoraConfig(task_type="CAUSAL_LM", init_lora_weights=False)
        peft_model = get_peft_model(model, config0, "adapter0", mixed=True)
        output0 = peft_model.generate(**input_dict)
        assert torch.isfinite(output0).all()
        assert not torch.allclose(output_base, output0)

        torch.manual_seed(1)
        config1 = LoHaConfig(task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"], init_weights=False)
        peft_model.add_adapter("adapter1", config1)
        peft_model.set_adapter(["adapter0", "adapter1"])
        output1 = peft_model.generate(**input_dict)
        assert torch.isfinite(output1).all()
        assert not torch.allclose(output0, output1)

        torch.manual_seed(2)
        config2 = AdaLoraConfig(task_type="CAUSAL_LM", init_lora_weights=False)
        peft_model.add_adapter("adapter2", config2)
        peft_model.set_adapter(["adapter0", "adapter1", "adapter2"])
        output2 = peft_model.generate(**input_dict)
        assert torch.isfinite(output2).all()
        assert not torch.allclose(output1, output2)

        torch.manual_seed(3)
        config3 = LoKrConfig(task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"], init_weights=False)
        peft_model.add_adapter("adapter3", config3)
        peft_model.set_adapter(["adapter0", "adapter1", "adapter2", "adapter3"])
        output3 = peft_model.generate(**input_dict)
        assert torch.isfinite(output3).all()
        assert not torch.allclose(output2, output3)

        torch.manual_seed(4)
        config4 = OFTConfig(task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"], init_weights=False)
        peft_model.add_adapter("adapter4", config4)
        peft_model.set_adapter(["adapter0", "adapter1", "adapter2", "adapter3", "adapter4"])
        output4 = peft_model.generate(**input_dict)
        assert torch.isfinite(output4).all()
        assert not torch.allclose(output3, output4)

        with peft_model.disable_adapter():
            output_disabled = peft_model.generate(**input_dict)
        assert torch.isfinite(output_disabled).all()
        assert torch.allclose(output_base, output_disabled)

        model_unloaded = peft_model.merge_and_unload()
        output_unloaded = model_unloaded.generate(**input_dict)
        assert torch.isfinite(output_unloaded).all()
        assert torch.allclose(output4, output_unloaded)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # save adapter0 (use normal PeftModel, because PeftMixedModel does not support saving)
            torch.manual_seed(0)
            model = AutoModelForCausalLM.from_pretrained(model_id).eval().to(self.torch_device)
            torch.manual_seed(0)
            peft_model = get_peft_model(model, config0, "adapter0")
            output0_save = peft_model(**input_dict).logits
            assert torch.isfinite(output0_save).all()
            peft_model.save_pretrained(tmp_dir)

            # save adapter1
            torch.manual_seed(0)
            model = AutoModelForCausalLM.from_pretrained(model_id).eval().to(self.torch_device)
            torch.manual_seed(1)
            peft_model = get_peft_model(model, config1, "adapter1")
            output1_save = peft_model(**input_dict).logits
            assert torch.isfinite(output1_save).all()
            peft_model.save_pretrained(tmp_dir)

            # load adapter0 and adapter1
            model = AutoModelForCausalLM.from_pretrained(model_id).eval().to(self.torch_device)
            peft_model = PeftMixedModel.from_pretrained(model, os.path.join(tmp_dir, "adapter0"), "adapter0")
            peft_model.load_adapter(os.path.join(tmp_dir, "adapter1"), "adapter1")
            peft_model.set_adapter(["adapter0", "adapter1"])
            output01_loaded = peft_model(**input_dict).logits

            atol, rtol = 1e-3, 1e-3
            assert torch.isfinite(output01_loaded).all()
            assert not torch.allclose(output0_save, output01_loaded, atol=atol, rtol=rtol)
            assert not torch.allclose(output1_save, output01_loaded, atol=atol, rtol=rtol)
