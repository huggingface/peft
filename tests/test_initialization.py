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

import pytest
import torch
from scipy import stats
from torch import nn

from peft import LoraConfig, get_peft_model
from peft.utils import infer_device


class TestInitialization:
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

    def test_rslora_scaling(self):
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

    def test_rslora_scaling_pattern(self):
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

    def test_use_dora_linear(self, data):
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

    def test_use_dora_linear_init_false(self, data):
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

    def test_use_dora_with_loftq_raises(self):
        with pytest.raises(ValueError, match="DoRA does not support megatron_core or LoftQ"):
            LoraConfig(target_modules=["linear"], use_dora=True, init_lora_weights="loftq")

    def test_use_dora_with_megatron_core_raises(self):
        megatron_config = {"does-not": "matter-here"}
        with pytest.raises(ValueError, match="DoRA does not support megatron_core or LoftQ"):
            LoraConfig(target_modules=["linear"], use_dora=True, megatron_config=megatron_config)
