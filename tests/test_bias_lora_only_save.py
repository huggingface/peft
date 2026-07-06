"""Test for issue #3306: bias='lora_only' should be saved in state_dict"""
import pytest
import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class TestBiasLoraOnlySave:
    """Test that bias is correctly saved when bias='lora_only'."""

    def test_bias_lora_only_saved_in_state_dict(self):
        """Test that bias parameters are saved in state_dict when bias='lora_only'."""
        model = SimpleModel()

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["linear1", "linear2"],
            bias="lora_only",
            lora_dropout=0.0,
        )

        peft_model = get_peft_model(model, config)

        # Get state dict
        state_dict = get_peft_model_state_dict(peft_model)

        # Check that bias is in state dict
        bias_keys = [k for k in state_dict if "bias" in k]
        assert len(bias_keys) > 0, "bias='lora_only' but no bias keys found in state_dict"

        # Verify specific bias keys exist
        assert "base_model.model.linear1.base_layer.bias" in state_dict
        assert "base_model.model.linear2.base_layer.bias" in state_dict

    def test_bias_lora_only_trainable(self):
        """Test that bias parameters are trainable when bias='lora_only'."""
        model = SimpleModel()

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["linear1", "linear2"],
            bias="lora_only",
            lora_dropout=0.0,
        )

        peft_model = get_peft_model(model, config)

        # Check that bias parameters are trainable
        trainable_params = {n: p for n, p in peft_model.named_parameters() if p.requires_grad}
        bias_params = [n for n in trainable_params if "bias" in n]

        assert len(bias_params) > 0, "No trainable bias parameters found"
        assert "base_model.model.linear1.base_layer.bias" in trainable_params
        assert "base_model.model.linear2.base_layer.bias" in trainable_params

    def test_bias_none_not_saved(self):
        """Test that bias is not saved when bias='none'."""
        model = SimpleModel()

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["linear1", "linear2"],
            bias="none",
            lora_dropout=0.0,
        )

        peft_model = get_peft_model(model, config)
        state_dict = get_peft_model_state_dict(peft_model)

        # Check that bias is not in state dict
        bias_keys = [k for k in state_dict if "bias" in k]
        assert len(bias_keys) == 0, "bias='none' but bias keys found in state_dict"

    def test_bias_all_saved(self):
        """Test that all bias parameters are saved when bias='all'."""
        model = SimpleModel()

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["linear1", "linear2"],
            bias="all",
            lora_dropout=0.0,
        )

        peft_model = get_peft_model(model, config)
        state_dict = get_peft_model_state_dict(peft_model)

        # Check that bias is in state dict
        bias_keys = [k for k in state_dict if "bias" in k]
        assert len(bias_keys) > 0, "bias='all' but no bias keys found in state_dict"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
