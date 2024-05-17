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

import re
from copy import deepcopy

import pytest
import torch
from scipy import stats
from torch import nn

from peft import AdaLoraConfig, LoraConfig, PeftModel, PromptTuningConfig, VeraConfig, get_peft_model
from peft.utils import infer_device


class TestLoraInitialization:
    """Test class to check the initialization of adapters."""

    torch_device = infer_device()

    def get_uniform(self, amin, amax, size=(10000,)):
        unif = torch.distributions.uniform.Uniform(amin, amax)
        samples = unif.sample(size)
        return samples

    def get_normal(self, mean, std, size=(10000,)):
        normal = torch.distributions.normal.Normal(mean, std)
        samples = normal.sample(size)
        return samples

    def get_model(self):
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                # choose a large weight so that averages are close to expected values
                self.linear = nn.Linear(1000, 1000)
                self.embed = nn.Embedding(1000, 1000)
                self.conv2d = nn.Conv2d(100, 100, 3)

            def forward(self, x):
                x_int = (100 * x).int()
                x_4d = x.flatten().reshape(1, 100, 10, 10)
                return self.linear(x), self.embed(x_int), self.conv2d(x_4d)

        return MyModule().eval().to(self.torch_device)

    @pytest.fixture
    def data(self):
        return torch.rand(10, 1000).to(self.torch_device)

    def test_lora_linear_init_default(self):
        # default is True
        torch.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["linear"])
        model = get_peft_model(model, config)
        weight_A = model.linear.lora_A["default"].weight
        weight_B = model.linear.lora_B["default"].weight

        # use statistical test to check if weight A is from a uniform distribution
        unif = self.get_uniform(weight_A.min().item(), weight_A.max().item())
        _, p_value = stats.kstest(weight_A.detach().flatten().cpu().numpy(), unif.flatten().cpu().numpy())
        assert p_value > 0.5

        # check that weight A is *not* from a normal distribution
        normal = self.get_normal(weight_A.mean().item(), weight_A.std().item())
        _, p_value = stats.kstest(weight_A.detach().flatten().cpu().numpy(), normal.flatten().cpu().numpy())
        assert p_value < 0.05

        # check that weight B is zero
        assert (weight_B == 0.0).all()

    def test_lora_linear_init_gaussian(self):
        # use gaussian init
        torch.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["linear"], init_lora_weights="gaussian")
        model = get_peft_model(model, config)
        weight_A = model.linear.lora_A["default"].weight
        weight_B = model.linear.lora_B["default"].weight

        # use statistical test to check if weight A is from a normal distribution
        normal = self.get_normal(0.0, 1 / config.r)
        _, p_value = stats.kstest(weight_A.detach().flatten().cpu().numpy(), normal.flatten().cpu().numpy())

        # import matplotlib.pyplot as plt
        # x = weight_A.detach().flatten().cpu().numpy()
        # breakpoint()

        assert p_value > 0.5

        # check that weight A is *not* from a uniform distribution
        unif = self.get_uniform(weight_A.min().item(), weight_A.max().item())
        _, p_value = stats.kstest(weight_A.detach().flatten().cpu().numpy(), unif.flatten().cpu().numpy())
        assert p_value < 0.05

        # check that weight B is zero
        assert (weight_B == 0.0).all()

    def test_lora_linear_false(self):
        torch.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["linear"], init_lora_weights=False)
        model = get_peft_model(model, config)
        weight_B = model.linear.lora_B["default"].weight

        # with init_lora_weights=False, weight B should *not* be zero. We don't care so much about the actual values
        # as long as they are not zero, in order to avoid identity transformation.
        assert not torch.allclose(weight_B, torch.zeros_like(weight_B))

    def test_lora_embedding_default(self):
        # embedding is initialized as a normal distribution, not kaiming uniform
        torch.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["embed"])
        model = get_peft_model(model, config)
        weight_A = model.embed.lora_embedding_A["default"]
        weight_B = model.embed.lora_embedding_B["default"]

        # use statistical test to check if weight B is from a normal distribution
        normal = self.get_normal(0.0, 1.0)
        _, p_value = stats.kstest(weight_B.detach().flatten().cpu().numpy(), normal.flatten().cpu().numpy())
        assert p_value > 0.5

        # check that weight B is *not* from a uniform distribution
        unif = self.get_uniform(weight_B.min().item(), weight_B.max().item())
        _, p_value = stats.kstest(weight_B.detach().flatten().cpu().numpy(), unif.flatten().cpu().numpy())
        assert p_value < 0.05

        # check that weight A is zero
        assert (weight_A == 0.0).all()

    def test_lora_embedding_gaussian(self):
        # embedding does not change with init_lora_weights="gaussian" vs True
        torch.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["embed"], init_lora_weights="gaussian")
        model = get_peft_model(model, config)
        weight_A = model.embed.lora_embedding_A["default"]
        weight_B = model.embed.lora_embedding_B["default"]

        # use statistical test to check if weight B is from a normal distribution
        normal = self.get_normal(0.0, 1.0)
        _, p_value = stats.kstest(weight_B.detach().flatten().cpu().numpy(), normal.flatten().cpu().numpy())
        assert p_value > 0.5

        # check that weight B is *not* from a uniform distribution
        unif = self.get_uniform(weight_B.min().item(), weight_B.max().item())
        _, p_value = stats.kstest(weight_B.detach().flatten().cpu().numpy(), unif.flatten().cpu().numpy())
        assert p_value < 0.05

        # check that weight A is zero
        assert (weight_A == 0.0).all()

    def test_lora_embedding_false(self):
        torch.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["embed"], init_lora_weights=False)
        model = get_peft_model(model, config)
        weight_A = model.embed.lora_embedding_B["default"]

        # with init_lora_weights=False, weight A should *not* be zero. We don't care so much about the actual values
        # as long as they are not zero, in order to avoid identity transformation.
        assert not torch.allclose(weight_A, torch.zeros_like(weight_A))

    def test_lora_conv2d_default(self):
        # default is True
        torch.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["conv2d"])
        model = get_peft_model(model, config)
        weight_A = model.conv2d.lora_A["default"].weight
        weight_B = model.conv2d.lora_B["default"].weight

        # use statistical test to check if weight A is from a uniform distribution
        unif = self.get_uniform(weight_A.min().item(), weight_A.max().item())
        _, p_value = stats.kstest(weight_A.detach().flatten().cpu().numpy(), unif.flatten().cpu().numpy())
        assert p_value > 0.5

        # check that weight A is *not* from a normal distribution
        normal = self.get_normal(weight_A.mean().item(), weight_A.std().item())
        _, p_value = stats.kstest(weight_A.detach().flatten().cpu().numpy(), normal.flatten().cpu().numpy())
        assert p_value < 0.05

        # check that weight B is zero
        assert (weight_B == 0.0).all()

    def test_lora_conv2d_init_gaussian(self):
        # use gaussian init
        torch.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["conv2d"], init_lora_weights="gaussian")
        model = get_peft_model(model, config)
        weight_A = model.conv2d.lora_A["default"].weight
        weight_B = model.conv2d.lora_B["default"].weight

        # use statistical test to check if weight A is from a normal distribution
        normal = self.get_normal(0.0, 1 / config.r)
        _, p_value = stats.kstest(weight_A.detach().flatten().cpu().numpy(), normal.flatten().cpu().numpy())
        assert p_value > 0.5

        # check that weight A is *not* from a uniform distribution
        unif = self.get_uniform(weight_A.min().item(), weight_A.max().item())
        _, p_value = stats.kstest(weight_A.detach().flatten().cpu().numpy(), unif.flatten().cpu().numpy())
        assert p_value < 0.05

        # check that weight B is zero
        assert (weight_B == 0.0).all()

    def test_lora_conv2d_false(self):
        torch.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["conv2d"], init_lora_weights=False)
        model = get_peft_model(model, config)
        weight_B = model.conv2d.lora_B["default"].weight

        # with init_lora_weights=False, weight B should *not* be zero. We don't care so much about the actual values
        # as long as they are not zero, in order to avoid identity transformation.
        assert not torch.allclose(weight_B, torch.zeros_like(weight_B))

    def test_lora_scaling_default(self):
        # default is True
        torch.manual_seed(0)

        model = self.get_model()

        # check scaling factor use_rslora=False
        config = LoraConfig(target_modules=["linear", "embed", "conv2d"], lora_alpha=3, r=16, use_rslora=False)
        model = get_peft_model(model, config)

        expected_scaling = config.lora_alpha / config.r

        assert model.linear.scaling["default"] == expected_scaling
        assert model.embed.scaling["default"] == expected_scaling
        assert model.conv2d.scaling["default"] == expected_scaling

    def test_lora_pissa_linear_init_default(self, data):
        model = self.get_model()
        output = model(data)[0]

        config = LoraConfig(init_lora_weights="pissa", target_modules=["linear"])
        peft_model = get_peft_model(deepcopy(model), config)
        assert torch.allclose(output, peft_model(data)[0], atol=1e-06)

        config = LoraConfig(init_lora_weights="pissa_niter_16", target_modules=["linear"])
        peft_model = get_peft_model(deepcopy(model), config)
        assert torch.allclose(output, peft_model(data)[0], atol=1e-06)

    def test_lora_pissa_conversion_same_output_after_loading(self, data, tmp_path):
        model = self.get_model()
        output_base = model(data)[0]

        config = LoraConfig(init_lora_weights="pissa", target_modules=["linear"], r=8)
        peft_model = get_peft_model(deepcopy(model), config)
        # save the initial model
        peft_model.peft_config["default"].init_lora_weights = True
        peft_model.save_pretrained(tmp_path / "init-model")
        peft_model.peft_config["default"].init_lora_weights = "pissa"

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight.data *= 2.0
        output_pissa = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not torch.allclose(output_base, output_pissa, atol=tol, rtol=tol)

        # save the model normally
        peft_model.save_pretrained(tmp_path / "pissa-model")
        model_loaded = PeftModel.from_pretrained(deepcopy(model), tmp_path / "pissa-model")
        output_loaded = model_loaded(data)[0]

        assert torch.allclose(output_pissa, output_loaded, atol=tol, rtol=tol)
        # sanity check: ranks should still be 8 as initially
        assert model_loaded.peft_config["default"].r == 8
        assert model_loaded.base_model.model.linear.lora_A["default"].weight.shape[0] == 8
        # sanity check: the base model weights were indeed changed
        assert not torch.allclose(
            model.linear.weight, model_loaded.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

        # save the model with conversion
        peft_model.save_pretrained(tmp_path / "pissa-model-converted", convert_pissa_to_lora=tmp_path / "init-model")
        model_converted = PeftModel.from_pretrained(deepcopy(model), tmp_path / "pissa-model-converted")
        output_converted = model_converted(data)[0]

        assert torch.allclose(output_pissa, output_converted, atol=tol, rtol=tol)
        # rank should be double of what it was initially
        assert model_converted.peft_config["default"].r == 16
        assert model_converted.base_model.model.linear.lora_A["default"].weight.shape[0] == 16
        # base model weights should be the same as the initial model
        assert torch.allclose(
            model.linear.weight, model_converted.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

    def test_lora_rslora_scaling(self):
        # default is True
        torch.manual_seed(0)

        model = self.get_model()

        # check scaling factor use_rslora=True
        config = LoraConfig(target_modules=["linear", "embed", "conv2d"], lora_alpha=3, r=16, use_rslora=True)
        model = get_peft_model(model, config)

        expected_scaling = config.lora_alpha / (config.r**0.5)

        assert model.linear.scaling["default"] == expected_scaling
        assert model.embed.scaling["default"] == expected_scaling
        assert model.conv2d.scaling["default"] == expected_scaling

    def test_lora_default_scaling_pattern(self):
        # default is True
        torch.manual_seed(0)

        model = self.get_model()

        # check scaling factor use_rslora=False with rank and alpha pattern
        config = LoraConfig(
            target_modules=["linear", "embed", "conv2d"],
            rank_pattern={"embed": 9, "conv2d": 16},
            alpha_pattern={"linear": 11, "conv2d": 13},
            lora_alpha=17,
            r=25,
            use_rslora=False,
        )
        model = get_peft_model(model, config)

        expected_scaling = {
            "linear": config.alpha_pattern["linear"] / config.r,
            "embed": config.lora_alpha / config.rank_pattern["embed"],
            "conv2d": config.alpha_pattern["conv2d"] / config.rank_pattern["conv2d"],
        }

        assert model.linear.scaling["default"] == expected_scaling["linear"]
        assert model.embed.scaling["default"] == expected_scaling["embed"]
        assert model.conv2d.scaling["default"] == expected_scaling["conv2d"]

    def test_lora_rslora_scaling_pattern(self):
        # default is True
        torch.manual_seed(0)

        model = self.get_model()

        # check scaling factor use_rslora=True with rank and alpha pattern
        config = LoraConfig(
            target_modules=["linear", "embed", "conv2d"],
            rank_pattern={"embed": 9, "conv2d": 16},
            alpha_pattern={"linear": 11, "conv2d": 13},
            lora_alpha=17,
            r=25,
            use_rslora=True,
        )
        model = get_peft_model(model, config)

        expected_scaling = {
            "linear": config.alpha_pattern["linear"] / (config.r**0.5),
            "embed": config.lora_alpha / (config.rank_pattern["embed"] ** 0.5),
            "conv2d": config.alpha_pattern["conv2d"] / (config.rank_pattern["conv2d"] ** 0.5),
        }

        assert model.linear.scaling["default"] == expected_scaling["linear"]
        assert model.embed.scaling["default"] == expected_scaling["embed"]
        assert model.conv2d.scaling["default"] == expected_scaling["conv2d"]

    def test_lora_use_dora_linear(self, data):
        # check that dora is a no-op when initialized
        torch.manual_seed(0)
        model = self.get_model()
        output_base, _, _ = model(data)

        # check scaling factor use_rslora=True
        config = LoraConfig(target_modules=["linear"], use_dora=True)
        model = get_peft_model(model, config)

        with model.disable_adapter():
            output_disabled, _, _ = model(data)
        output_dora, _, _ = model(data)

        assert torch.allclose(output_base, output_disabled)
        assert torch.allclose(output_base, output_dora)

    def test_lora_use_dora_linear_init_false(self, data):
        # with init_lora_weights=False, dora should not be a no-op
        torch.manual_seed(0)
        model = self.get_model()
        output_base, _, _ = model(data)

        # check scaling factor use_rslora=True
        config = LoraConfig(target_modules=["linear"], use_dora=True, init_lora_weights=False)
        model = get_peft_model(model, config)

        with model.disable_adapter():
            output_disabled, _, _ = model(data)
        output_dora, _, _ = model(data)

        assert torch.allclose(output_base, output_disabled)
        assert not torch.allclose(output_base, output_dora)

    def test_lora_use_dora_with_megatron_core_raises(self):
        megatron_config = {"does-not": "matter-here"}
        with pytest.raises(ValueError, match="DoRA does not support megatron_core"):
            LoraConfig(target_modules=["linear"], use_dora=True, megatron_config=megatron_config)


class TestAdaLoraInitialization:
    def test_adalora_target_modules_set(self):
        config = AdaLoraConfig(target_modules=["linear", "embed", "conv2d"])
        assert config.target_modules == {"linear", "embed", "conv2d"}

    def test_adalora_use_dora_raises(self):
        with pytest.raises(ValueError, match="ADALORA does not support DoRA"):
            AdaLoraConfig(use_dora=True)

    def test_adalora_loftq_config_raises(self):
        with pytest.raises(ValueError, match="ADALORA does not support LOFTQ"):
            AdaLoraConfig(loftq_config={"loftq": "config"})


class TestPromptTuningInitialization:
    torch_device = infer_device()

    def get_model(self):
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                # choose a large weight so that averages are close to expected values
                self.linear = nn.Linear(1000, 1000)
                self.embed = nn.Embedding(1000, 1000)
                self.conv2d = nn.Conv2d(100, 100, 3)

            def forward(self, x):
                x_int = (100 * x).int()
                x_4d = x.flatten().reshape(1, 100, 10, 10)
                return self.linear(x), self.embed(x_int), self.conv2d(x_4d)

        return MyModule().eval().to(self.torch_device)

    def test_use_prompt_tuning_init_text_raises(self):
        with pytest.raises(ValueError, match="When prompt_tuning_init='TEXT', tokenizer_name_or_path can't be None"):
            PromptTuningConfig(prompt_tuning_init="TEXT", prompt_tuning_init_text="prompt tuning init text")
        with pytest.raises(ValueError, match="When prompt_tuning_init='TEXT', prompt_tuning_init_text can't be None"):
            PromptTuningConfig(prompt_tuning_init="TEXT", tokenizer_name_or_path="t5-base")

    def test_vera_mixing_save_projection_raises(self):
        # it is unclear what the right thing to do would be if some adapters save the projection weights and some don't
        # so we better raise an error

        config0 = VeraConfig(target_modules="linear", init_weights=False, save_projection=True)
        model = self.get_model()
        model = get_peft_model(model, config0)
        config1 = VeraConfig(target_modules="linear", init_weights=False, save_projection=False)
        msg = re.escape(
            "VeRA projection weights must be saved for all adapters or none, but got multiple different values: "
            "[False, True]"
        )
        with pytest.raises(ValueError, match=msg):
            model.add_adapter("other", config1)
