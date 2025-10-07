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

"""
Tests for negative weights support in adapter merging.

This test suite validates the implementation of negative weights for LoRA adapter merging,
as described in issue #2796 and the LoRA Hub paper.
"""

import pytest
import torch
from torch import nn

from peft import LoraConfig, get_peft_model


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def lora_config():
    """Create a basic LoRA config."""
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["linear1", "linear2"],
        lora_dropout=0.0,
        bias="none",
        init_lora_weights=False,  # Don't randomize for reproducibility
    )


class TestNegativeWeightsLinear:
    """Test negative weights with linear combination."""

    def test_single_negative_weight_linear(self, simple_model, lora_config):
        """Test that a single adapter with negative weight works."""
        model = get_peft_model(simple_model, lora_config, adapter_name="adapter1")

        # This should not raise ValueError
        model.add_weighted_adapter(
            adapters=["adapter1"],
            weights=[-0.5],
            adapter_name="merged_negative",
            combination_type="linear",
        )

        assert "merged_negative" in model.peft_config

    def test_mixed_positive_negative_weights_linear(self, simple_model, lora_config):
        """Test merging with both positive and negative weights."""
        model = get_peft_model(simple_model, lora_config, adapter_name="adapter1")
        model.add_adapter("adapter2", lora_config)

        # Mix of positive and negative weights
        model.add_weighted_adapter(
            adapters=["adapter1", "adapter2"],
            weights=[0.7, -0.3],
            adapter_name="merged_mixed",
            combination_type="linear",
        )

        assert "merged_mixed" in model.peft_config

    def test_all_negative_weights_linear(self, simple_model, lora_config):
        """Test that all negative weights work."""
        model = get_peft_model(simple_model, lora_config, adapter_name="adapter1")
        model.add_adapter("adapter2", lora_config)

        model.add_weighted_adapter(
            adapters=["adapter1", "adapter2"],
            weights=[-0.5, -0.3],
            adapter_name="merged_all_negative",
            combination_type="linear",
        )

        assert "merged_all_negative" in model.peft_config

    def test_zero_and_negative_weights_linear(self, simple_model, lora_config):
        """Test mixing zero and negative weights."""
        model = get_peft_model(simple_model, lora_config, adapter_name="adapter1")
        model.add_adapter("adapter2", lora_config)

        model.add_weighted_adapter(
            adapters=["adapter1", "adapter2"],
            weights=[0.0, -0.5],
            adapter_name="merged_zero_negative",
            combination_type="linear",
        )

        assert "merged_zero_negative" in model.peft_config


class TestNegativeWeightsAllCombinations:
    """Test negative weights with all combination types."""

    @pytest.mark.parametrize(
        "combination_type",
        ["linear", "ties", "dare_linear", "dare_ties", "magnitude_prune"],
    )
    def test_negative_weights_all_non_svd_methods(self, simple_model, lora_config, combination_type):
        """Test negative weights work with all non-SVD combination methods."""
        model = get_peft_model(simple_model, lora_config, adapter_name="adapter1")
        model.add_adapter("adapter2", lora_config)

        density = 0.5 if combination_type != "linear" else None
        kwargs = {"density": density} if density is not None else {}

        model.add_weighted_adapter(
            adapters=["adapter1", "adapter2"],
            weights=[0.6, -0.4],
            adapter_name=f"merged_{combination_type}",
            combination_type=combination_type,
            **kwargs,
        )

        assert f"merged_{combination_type}" in model.peft_config

    @pytest.mark.parametrize(
        "combination_type",
        ["svd", "ties_svd", "dare_linear_svd", "dare_ties_svd", "magnitude_prune_svd"],
    )
    def test_negative_weights_all_svd_methods(self, simple_model, lora_config, combination_type):
        """Test negative weights work with all SVD combination methods."""
        model = get_peft_model(simple_model, lora_config, adapter_name="adapter1")
        model.add_adapter("adapter2", lora_config)

        density = 0.5 if "ties" in combination_type or "dare" in combination_type or "prune" in combination_type else None
        kwargs = {"density": density} if density is not None else {}

        model.add_weighted_adapter(
            adapters=["adapter1", "adapter2"],
            weights=[0.6, -0.4],
            adapter_name=f"merged_{combination_type}",
            combination_type=combination_type,
            **kwargs,
        )

        assert f"merged_{combination_type}" in model.peft_config

    def test_negative_weights_cat_combination(self, simple_model, lora_config):
        """Test negative weights with cat combination."""
        model = get_peft_model(simple_model, lora_config, adapter_name="adapter1")
        model.add_adapter("adapter2", lora_config)

        model.add_weighted_adapter(
            adapters=["adapter1", "adapter2"],
            weights=[0.8, -0.2],
            adapter_name="merged_cat",
            combination_type="cat",
        )

        assert "merged_cat" in model.peft_config


class TestNegativeWeightsMathematicalCorrectness:
    """Test mathematical correctness of negative weight merging."""

    def test_negative_weight_negates_adapter(self, simple_model, lora_config):
        """Test that weight=-1.0 properly negates an adapter."""
        torch.manual_seed(42)
        model = get_peft_model(simple_model, lora_config, adapter_name="adapter1")

        # Create merged adapter with weight=1.0
        model.add_weighted_adapter(
            adapters=["adapter1"],
            weights=[1.0],
            adapter_name="merged_positive",
            combination_type="linear",
        )

        # Create merged adapter with weight=-1.0
        model.add_weighted_adapter(
            adapters=["adapter1"],
            weights=[-1.0],
            adapter_name="merged_negative",
            combination_type="linear",
        )

        # Get the LoRA weights for comparison
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and "merged_positive" in module.lora_A:
                pos_A = module.lora_A["merged_positive"].weight.data
                neg_A = module.lora_A["merged_negative"].weight.data
                pos_B = module.lora_B["merged_positive"].weight.data
                neg_B = module.lora_B["merged_negative"].weight.data

                # Check that negative adapter is negation of positive
                # Since we apply sign to both A and B: sign * sqrt(|w|)
                # For w=1: sqrt(1) = 1, for w=-1: -sqrt(1) = -1
                assert torch.allclose(neg_A, -pos_A, atol=1e-6), f"A matrices should be negated"
                assert torch.allclose(neg_B, -pos_B, atol=1e-6), f"B matrices should be negated"

    def test_subtraction_with_negative_weights(self, simple_model, lora_config):
        """Test that positive + negative weights properly subtract in merged result."""
        torch.manual_seed(42)
        model = get_peft_model(simple_model, lora_config, adapter_name="adapter1")
        model.add_adapter("adapter2", lora_config)

        # Merge: 1.0 * adapter1 + (-1.0) * adapter2
        # This should give us the difference between adapters
        model.add_weighted_adapter(
            adapters=["adapter1", "adapter2"],
            weights=[1.0, -1.0],
            adapter_name="difference",
            combination_type="linear",
        )

        # Verify the merged adapter was created
        assert "difference" in model.peft_config

        # The math should work: merged_A = A1 - A2, merged_B = B1 - B2
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and "adapter1" in module.lora_A and "adapter2" in module.lora_A:
                A1 = module.lora_A["adapter1"].weight.data
                A2 = module.lora_A["adapter2"].weight.data
                B1 = module.lora_B["adapter1"].weight.data
                B2 = module.lora_B["adapter2"].weight.data

                merged_A = module.lora_A["difference"].weight.data
                merged_B = module.lora_B["difference"].weight.data

                # Since we use sqrt and apply to both, the expected result is:
                # Get scaling factors
                s1 = module.scaling["adapter1"]
                s2 = module.scaling["adapter2"]

                # merged_A = 1*sqrt(1*s1)*A1 + (-1)*sqrt(1*s2)*A2
                # merged_B = 1*sqrt(1*s1)*B1 + (-1)*sqrt(1*s2)*B2
                import math
                expected_A = math.sqrt(1.0 * s1) * A1 + (-1) * math.sqrt(1.0 * s2) * A2
                expected_B = math.sqrt(1.0 * s1) * B1 + (-1) * math.sqrt(1.0 * s2) * B2

                assert torch.allclose(
                    merged_A, expected_A, atol=1e-5
                ), f"Merged A should equal sqrt(s1)*A1 - sqrt(s2)*A2, got diff {(merged_A - expected_A).abs().max()}"
                assert torch.allclose(
                    merged_B, expected_B, atol=1e-5
                ), f"Merged B should equal sqrt(s1)*B1 - sqrt(s2)*B2, got diff {(merged_B - expected_B).abs().max()}"


class TestBackwardCompatibility:
    """Test that positive weights still work correctly (backward compatibility)."""

    def test_positive_weights_unchanged(self, simple_model, lora_config):
        """Test that positive weights produce the same results as before."""
        torch.manual_seed(42)
        model = get_peft_model(simple_model, lora_config, adapter_name="adapter1")
        model.add_adapter("adapter2", lora_config)

        # Standard positive weight merging should work as before
        model.add_weighted_adapter(
            adapters=["adapter1", "adapter2"],
            weights=[0.6, 0.4],
            adapter_name="merged_positive",
            combination_type="linear",
        )

        assert "merged_positive" in model.peft_config

        # Verify the math is correct for positive weights
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and "adapter1" in module.lora_A and "adapter2" in module.lora_A:
                A1 = module.lora_A["adapter1"].weight.data
                A2 = module.lora_A["adapter2"].weight.data
                B1 = module.lora_B["adapter1"].weight.data
                B2 = module.lora_B["adapter2"].weight.data

                merged_A = module.lora_A["merged_positive"].weight.data
                merged_B = module.lora_B["merged_positive"].weight.data

                # Get scaling factors
                s1 = module.scaling["adapter1"]
                s2 = module.scaling["adapter2"]

                # Expected: sqrt(0.6 * s1) * A1 + sqrt(0.4 * s2) * A2
                import math

                expected_A = math.sqrt(0.6 * s1) * A1 + math.sqrt(0.4 * s2) * A2
                expected_B = math.sqrt(0.6 * s1) * B1 + math.sqrt(0.4 * s2) * B2

                assert torch.allclose(merged_A, expected_A, atol=1e-5), "Positive weight merging changed behavior"
                assert torch.allclose(merged_B, expected_B, atol=1e-5), "Positive weight merging changed behavior"


class TestEdgeCases:
    """Test edge cases with negative weights."""

    def test_negative_weight_with_different_scaling(self, simple_model):
        """Test negative weights with different scaling factors."""
        config1 = LoraConfig(
            r=8,
            lora_alpha=16,  # scaling = 2
            target_modules=["linear1", "linear2"],
            lora_dropout=0.0,
            bias="none",
            init_lora_weights=False,
        )
        config2 = LoraConfig(
            r=8,
            lora_alpha=32,  # scaling = 4
            target_modules=["linear1", "linear2"],
            lora_dropout=0.0,
            bias="none",
            init_lora_weights=False,
        )

        model = get_peft_model(simple_model, config1, adapter_name="adapter1")
        model.add_adapter("adapter2", config2)

        # Should handle different scalings correctly with negative weights
        model.add_weighted_adapter(
            adapters=["adapter1", "adapter2"],
            weights=[0.5, -0.3],
            adapter_name="merged_diff_scaling",
            combination_type="linear",
        )

        assert "merged_diff_scaling" in model.peft_config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
