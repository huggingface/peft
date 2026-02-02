# Copyright 2026-present the HuggingFace Inc. team.
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
Tests for MonteCLoRA variant implementation.

This test suite verifies that MonteCLoRA works correctly as a LoRA variant,
following the same pattern as DoRA, aLoRA, etc.
"""

import tempfile
import unittest

import torch
from torch import nn

from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import Linear as LoraLinear
from peft.tuners.monteclora import MonteCLoraConfig


class SimpleNet(nn.Module):
    """
    A simple dummy model for testing Linear layer replacement.
    """

    def __init__(self, in_features=10, out_features=10):
        super().__init__()
        self.lin0 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(out_features, out_features)

    def forward(self, x):
        x = self.lin0(x)
        x = self.relu(x)
        x = self.lin1(x)
        return x


class TestMonteCLoraVariant(unittest.TestCase):
    """Test suite for MonteCLoRA as a LoRA variant."""

    def setUp(self):
        self.input_dim = 10
        self.output_dim = 10
        self.model = SimpleNet(self.input_dim, self.output_dim)

        # MonteCLoRA configuration
        self.monteclora_config = MonteCLoraConfig(
            monteclora_n=4,
            sample_scaler=1,
            kl_loss_weight=1e-5,
            use_entropy=False,
            mc_training=True,
        )

        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["lin0","lin1"],
            lora_dropout=0.1,
            task_type=None, 
            use_monteclora=True,
            monteclora_config=self.monteclora_config,
        )


    def test_variant_in_lora_variant_dict(self):
        """
        Test that the MonteCLoRA variant is registered in the lora_variant dict.
        """
        model = get_peft_model(self.model, self.lora_config)

        lin0 = model.base_model.model.lin0
        assert hasattr(lin0, "lora_variant"), "LoRA layer should have lora_variant attribute"

        assert "default" in lin0.lora_variant, "Default adapter should have a variant registered"

        variant = lin0.lora_variant["default"]
        assert variant.__class__.__name__ == "MonteCLoraLinearVariant", "Variant should be MonteCLoraLinearVariant"


    def test_trainable_parameters(self):
        """
        Test that base weights are frozen and MonteCLoRA parameters are trainable.
        """
        model = get_peft_model(self.model, self.lora_config)

        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]

        assert not model.base_model.model.lin0.weight.requires_grad, "Base weights should be frozen"

        assert any("lora_A" in n for n in trainable_names), "LoRA A weights should be trainable"
        assert any("lora_B" in n for n in trainable_names), "LoRA B weights should be trainable"

        assert any("lora_monteclora_sampler" in n for n in trainable_names), "MonteCLoRA sampler should be trainable"

        assert any("std_prior" in n for n in trainable_names), "std_prior should be trainable in MonteCLoRA sampler"
        assert any("expert_weights_prior" in n for n in trainable_names), (
            "expert_weights_prior should be trainable in MonteCLoRA sampler"
        )


    def test_forward_pass_training(self):
        """
        Test that the forward pass runs in training mode with Monte Carlo sampling.
        """
        model = get_peft_model(self.model, self.lora_config)
        model.train()  

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        input_data = torch.randn(2, self.input_dim).to(device)
        output = model(input_data)

        assert output.shape == (2, self.output_dim), f"Output shape mismatch: {output.shape}"

        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

        loss = output.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient not computed for {name}"



    def test_eval_mode_consistency(self):
        """
        Test that eval mode turns off sampling and produces deterministic results.
        """
        model = get_peft_model(self.model, self.lora_config)
        model.eval() 

        input_data = torch.randn(1, self.input_dim)

        with torch.no_grad():
            out1 = model(input_data)
            out2 = model(input_data)

        assert torch.allclose(out1, out2, atol=1e-08), "Outputs should be deterministic in eval mode (no MC sampling)"


    def test_variational_loss_computation(self):
        """
        Test that variational loss can be computed from MonteCLoRA samplers.
        """
        model = get_peft_model(self.model, self.lora_config)
        model.train()

        input_data = torch.randn(2, self.input_dim)
        _ = model(input_data)

        var_loss_sum = 0.0
        num_samplers = 0

        for name, module in model.named_modules():
            if hasattr(module, "get_variational_loss") and module.__class__.__name__ == "MonteCLoRASampler":
                kl_loss, entropy_loss = module.get_variational_loss()
                var_loss_sum += kl_loss + entropy_loss
                num_samplers += 1

        assert num_samplers > 0, "No MonteCLoRA samplers found for loss computation"
        assert var_loss_sum > 0, "Variational loss should be positive"
        assert not torch.isnan(var_loss_sum), "Variational loss contains NaN"


    def test_save_and_load(self):
        """
        Test saving the adapter and loading it back.
        """
        model = get_peft_model(self.model, self.lora_config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)

            model_loaded = PeftModel.from_pretrained(self.model, tmp_dir)

            loaded_config = model_loaded.peft_config["default"]
            assert loaded_config.use_monteclora, "use_monteclora should be True"
            assert loaded_config.monteclora_config is not None, "monteclora_config should not be None"
            assert loaded_config.monteclora_config.monteclora_n == 4, "monteclora_n should match"

            state_dict = model_loaded.state_dict()
            sampler_keys = [k for k in state_dict.keys() if "lora_monteclora_sampler" in k]
            assert len(sampler_keys) > 0, "No MonteCLoRA sampler parameters in loaded state dict"


    def test_merging_and_unmerging(self):
        """
        Test that merging and unmerging MonteCLoRA adapters works correctly.

        For merge/unmerge operations, we use the base LoRA weights (ignoring MC sampling).
        """
        model = get_peft_model(self.model, self.lora_config)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        input_data = torch.randn(2, self.input_dim).to(device)

        with torch.no_grad():
            output_before_merge = model(input_data)

        model.merge_adapter()
        assert model.base_model.model.lin0.merged, "Adapter should be marked as merged"

        with torch.no_grad():
            output_after_merge = model(input_data)

        assert torch.allclose(output_before_merge, output_after_merge, atol=1e-05), (
            "Output should be similar before and after merge"
        )

        model.unmerge_adapter()
        assert not model.base_model.model.lin0.merged, "Adapter should be marked as unmerged"

        with torch.no_grad():
            output_after_unmerge = model(input_data)

        assert torch.allclose(output_before_merge, output_after_unmerge, atol=1e-05), (
            "Output should match original after unmerge"
        )


    def test_config_validation(self):
        """
        Test that MonteCLoraConfig validates parameters correctly.
        """
        valid_config = MonteCLoraConfig(monteclora_n=8)
        assert valid_config.monteclora_n == 8

        with self.assertRaises(ValueError):
            MonteCLoraConfig(monteclora_n=-1)

        with self.assertRaises(ValueError):
            MonteCLoraConfig(dirichlet_prior=-0.1)

        with self.assertRaises(ValueError):
            MonteCLoraConfig(buffer_size=-10)


    def test_different_monteclora_configs(self):
        """
        Test that different MonteCLoRA configurations work correctly.
        """
        configs = [
            {"monteclora_n": 2, "sample_scaler": 1e-3},
            {"monteclora_n": 8, "use_entropy": True},
            {"monteclora_n": 4, "kl_loss_weight": 1e-4},
        ]

        for config_kwargs in configs:
            monteclora_config = MonteCLoraConfig(**config_kwargs)
            lora_config = LoraConfig(
                r=8,
                target_modules=["lin0"],
                use_monteclora=True,
                monteclora_config=monteclora_config,
            )

            model = SimpleNet(self.input_dim, self.output_dim)
            model = get_peft_model(model, lora_config)

            input_data = torch.randn(2, self.input_dim)
            output = model(input_data)

            assert output.shape == (2, self.output_dim)


    def test_compatibility_with_standard_lora(self):
        """
        Test that models with MonteCLoRA can coexist with standard LoRA adapters.
        """
        model = get_peft_model(self.model, self.lora_config)

        standard_lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["lin0"],
            use_monteclora=False,
        )

        model.add_adapter("standard_lora", standard_lora_config)

        assert "default" in model.peft_config
        assert "standard_lora" in model.peft_config

        model.set_adapter("default")
        input_data = torch.randn(2, self.input_dim)
        output_monteclora = model(input_data)
        assert output_monteclora.shape == (2, self.output_dim)

        model.set_adapter("standard_lora")
        output_standard = model(input_data)
        assert output_standard.shape == (2, self.output_dim)


if __name__ == "__main__":
    unittest.main()
