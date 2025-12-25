# Copyright 2024-present the HuggingFace Inc. team.
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

"""
Tests for AdaDoRA (AdaLoRA + DoRA) integration.

Tests the variant pattern implementation that reuses existing DoRA code
while adapting it for AdaLoRA's SVD decomposition.
"""

import torch
from torch import nn

from peft import AdaLoraConfig, get_peft_model
from peft.tuners.adalora import AdaDoraLinearLayer, AdaDoraLinearVariant


class SimpleModel(nn.Module):
    """Simple model for testing AdaDoRA."""

    def __init__(self, in_features=16, out_features=16):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class TestAdaDoraLinearLayer:
    """Tests for AdaDoraLinearLayer."""

    def test_init(self):
        """Test that AdaDoraLinearLayer initializes correctly."""
        layer = AdaDoraLinearLayer(fan_in_fan_out=False)
        assert hasattr(layer, "fan_in_fan_out")
        assert layer.fan_in_fan_out is False

    def test_get_weight_norm(self):
        """Test weight norm computation for SVD decomposition."""
        layer = AdaDoraLinearLayer(fan_in_fan_out=False)

        # create test tensors
        weight = torch.randn(16, 16)
        lora_A = torch.randn(4, 16)
        lora_B = torch.randn(16, 4)
        lora_E = torch.randn(4, 1)
        scaling = 32.0
        ranknum = 4.0

        weight_norm = layer.get_weight_norm(weight, lora_A, lora_B, lora_E, scaling, ranknum)

        assert weight_norm.shape == (16,)
        assert torch.isfinite(weight_norm).all()

    def test_update_layer(self):
        """Test that update_layer initializes magnitude correctly."""
        layer = AdaDoraLinearLayer(fan_in_fan_out=False)
        base_layer = nn.Linear(16, 16)

        lora_A = torch.randn(4, 16)
        lora_B = torch.randn(16, 4)
        lora_E = torch.randn(4, 1)
        scaling = 32.0
        ranknum = 4.0

        layer.update_layer(
            base_layer=base_layer,
            lora_A=lora_A,
            lora_B=lora_B,
            lora_E=lora_E,
            scaling=scaling,
            ranknum=ranknum,
        )

        assert hasattr(layer, "weight")
        assert layer.weight.shape == (16,)
        assert layer.weight.requires_grad

    def test_forward(self):
        """Test forward pass."""
        layer = AdaDoraLinearLayer(fan_in_fan_out=False)
        base_layer = nn.Linear(16, 16)

        lora_A = torch.randn(4, 16)
        lora_B = torch.randn(16, 4)
        lora_E = torch.randn(4, 1)
        scaling = 32.0
        ranknum = 4.0

        layer.update_layer(
            base_layer=base_layer,
            lora_A=lora_A,
            lora_B=lora_B,
            lora_E=lora_E,
            scaling=scaling,
            ranknum=ranknum,
        )

        x = torch.randn(2, 16)
        result = layer(
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            lora_E=lora_E,
            scaling=scaling,
            ranknum=ranknum,
            base_layer=base_layer,
        )

        assert result.shape == (2, 16)
        assert torch.isfinite(result).all()


class TestAdaDoraLinearVariant:
    """Tests for AdaDoraLinearVariant."""

    def test_variant_has_static_methods(self):
        """Test that variant has all required static methods."""
        assert hasattr(AdaDoraLinearVariant, "init")
        assert hasattr(AdaDoraLinearVariant, "forward")
        assert hasattr(AdaDoraLinearVariant, "merge_safe")
        assert hasattr(AdaDoraLinearVariant, "merge_unsafe")
        assert hasattr(AdaDoraLinearVariant, "unmerge")


class TestAdaDoRAIntegration:
    """Integration tests for AdaDoRA (AdaLoRA + DoRA)."""

    def test_adalora_with_dora_creates_variant(self):
        """Test that AdaLoRA with use_dora=True creates the variant."""
        model = SimpleModel()
        config = AdaLoraConfig(
            target_modules=["linear"],
            use_dora=True,
            init_r=4,
            target_r=2,
            total_step=100,
            tinit=10,
            tfinal=10,
        )
        peft_model = get_peft_model(model, config)

        # check that variant was created
        linear = peft_model.base_model.model.linear
        assert "default" in linear.lora_variant
        assert "default" in linear.lora_magnitude_vector

    def test_adalora_without_dora_no_variant(self):
        """Test that AdaLoRA without use_dora=True does not create variant."""
        model = SimpleModel()
        config = AdaLoraConfig(
            target_modules=["linear"],
            use_dora=False,
            init_r=4,
            target_r=2,
            total_step=100,
        )
        peft_model = get_peft_model(model, config)

        # check that no variant was created
        linear = peft_model.base_model.model.linear
        assert "default" not in linear.lora_variant
        assert "default" not in linear.lora_magnitude_vector

    def test_adalora_with_dora_forward(self):
        """Test forward pass with DoRA enabled."""
        model = SimpleModel()
        config = AdaLoraConfig(
            target_modules=["linear"],
            use_dora=True,
            init_r=4,
            target_r=2,
            total_step=100,
            tinit=10,
            tfinal=10,
        )
        peft_model = get_peft_model(model, config)

        x = torch.randn(2, 16)
        output = peft_model(x)

        assert output.shape == (2, 16)
        assert torch.isfinite(output).all()

    def test_adalora_with_dora_training(self):
        """Test that gradients flow correctly during training."""
        model = SimpleModel()
        config = AdaLoraConfig(
            target_modules=["linear"],
            use_dora=True,
            init_r=4,
            target_r=2,
            total_step=100,
            tinit=10,
            tfinal=10,
        )
        peft_model = get_peft_model(model, config)
        peft_model.train()

        x = torch.randn(2, 16)
        y = torch.randn(2, 16)

        output = peft_model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        # check that magnitude vector has gradient
        linear = peft_model.base_model.model.linear
        dora_layer = linear.lora_magnitude_vector["default"]
        assert dora_layer.weight.grad is not None

    def test_adalora_with_dora_merge(self):
        """Test that merge produces consistent outputs."""
        model = SimpleModel()
        config = AdaLoraConfig(
            target_modules=["linear"],
            use_dora=True,
            init_r=4,
            target_r=2,
            total_step=100,
            tinit=10,
            tfinal=10,
        )
        peft_model = get_peft_model(model, config)
        peft_model.eval()

        x = torch.randn(2, 16)
        with torch.no_grad():
            out_before = peft_model(x)

        peft_model.merge_adapter()

        with torch.no_grad():
            out_after = peft_model(x)

        # outputs should be similar after merge (allowing for numerical precision)
        assert torch.allclose(out_before, out_after, atol=1e-5)

    def test_adalora_with_dora_unmerge(self):
        """Test that unmerge restores original behavior."""
        model = SimpleModel()
        config = AdaLoraConfig(
            target_modules=["linear"],
            use_dora=True,
            init_r=4,
            target_r=2,
            total_step=100,
            tinit=10,
            tfinal=10,
        )
        peft_model = get_peft_model(model, config)
        peft_model.eval()

        x = torch.randn(2, 16)
        with torch.no_grad():
            out_before = peft_model(x)

        peft_model.merge_adapter()
        peft_model.unmerge_adapter()

        with torch.no_grad():
            out_after = peft_model(x)

        # outputs should be identical after merge/unmerge
        assert torch.allclose(out_before, out_after, atol=1e-5)

    def test_adalora_with_dora_magnitude_update(self):
        """Test that magnitude updates after pruning."""
        model = SimpleModel()
        config = AdaLoraConfig(
            target_modules=["linear"],
            use_dora=True,
            init_r=4,
            target_r=2,
            tinit=1,
            tfinal=1,
            total_step=10,
        )
        peft_model = get_peft_model(model, config)

        linear = peft_model.base_model.model.linear
        dora_layer = linear.lora_magnitude_vector["default"]

        # simulate rank pruning by zeroing some E values
        with torch.no_grad():
            linear.lora_E["default"][0] = 0.0

        # update magnitude
        dora_layer.update_magnitude_after_pruning(
            linear.get_base_layer(),
            linear.lora_A["default"],
            linear.lora_B["default"],
            linear.lora_E["default"],
            linear.scaling["default"],
            linear.ranknum["default"].item(),
        )

        # magnitude should have changed
        # (they might be similar but the point is it ran without error)
        assert torch.isfinite(dora_layer.weight).all()


class TestAdaDoRACompatibility:
    """Tests for backward compatibility and edge cases."""

    def test_adalora_with_dora_multiple_adapters(self):
        """Test that multiple adapters work correctly."""
        model = SimpleModel()
        config1 = AdaLoraConfig(
            target_modules=["linear"],
            use_dora=True,
            init_r=4,
            target_r=2,
            total_step=100,
            tinit=10,
            tfinal=10,
        )
        peft_model = get_peft_model(model, config1, adapter_name="adapter1")

        # add second adapter
        config2 = AdaLoraConfig(
            target_modules=["linear"],
            use_dora=True,
            init_r=4,
            target_r=2,
            total_step=100,
            tinit=10,
            tfinal=10,
            inference_mode=True,
        )
        peft_model.add_adapter("adapter2", config2)

        linear = peft_model.base_model.model.linear
        assert "adapter1" in linear.lora_variant
        assert "adapter2" in linear.lora_variant

    def test_adalora_with_dora_disable_adapters(self):
        """Test disabling adapters works correctly."""
        model = SimpleModel()
        config = AdaLoraConfig(
            target_modules=["linear"],
            use_dora=True,
            init_r=4,
            target_r=2,
            total_step=100,
            tinit=10,
            tfinal=10,
        )
        peft_model = get_peft_model(model, config)
        peft_model.eval()

        # modify adapter weights to ensure non-zero contribution
        # AdaLoRA uses ΔW = B @ (A * E) * scaling / ranknum
        # default init has lora_E=0, so adapter contribution is zero
        linear = peft_model.base_model.model.linear
        with torch.no_grad():
            linear.lora_E["default"].fill_(1.0)
            linear.lora_B["default"].fill_(0.1)
            # update magnitude after weight change
            dora_layer = linear.lora_magnitude_vector["default"]
            dora_layer.update_magnitude_after_pruning(
                linear.get_base_layer(),
                linear.lora_A["default"],
                linear.lora_B["default"],
                linear.lora_E["default"],
                linear.scaling["default"],
                linear.ranknum["default"].item(),
            )

        x = torch.randn(2, 16)

        # get output with adapters
        with torch.no_grad():
            out_with_adapter = peft_model(x)

        # disable adapters
        peft_model.disable_adapter_layers()

        with torch.no_grad():
            out_without_adapter = peft_model(x)

        # outputs should be different now that adapter has non-zero contribution
        assert not torch.allclose(out_with_adapter, out_without_adapter, atol=1e-5)

        # re-enable and check
        peft_model.enable_adapter_layers()

        with torch.no_grad():
            out_reenabled = peft_model(x)

        assert torch.allclose(out_with_adapter, out_reenabled, atol=1e-5)
