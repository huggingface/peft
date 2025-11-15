import pytest
import torch

from peft import LoraConfig, TaskType, get_peft_model


class TestIntruderMitigation:
    """Test intruder dimension mitigation functionality."""

    @pytest.fixture
    def tiny_lora_model(self):
        """Create a tiny model with LoRA for testing."""
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(vocab_size=1000, n_positions=128, n_embd=64, n_layer=2, n_head=2)
        base_model = GPT2LMHeadModel(config)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"],
        )

        model = get_peft_model(base_model, lora_config)

        # Add some noise to LoRA weights (simulate fine-tuning)
        for name, param in model.named_parameters():
            if "lora_" in name and param.requires_grad:
                param.data += torch.randn_like(param) * 0.1

        return model

    def test_reduce_intruder_dimensions_nondestructive(self, tiny_lora_model):
        """Test non-destructive path (creates new adapter)."""
        model = tiny_lora_model

        # This should create a new adapter
        model = model.reduce_intruder_dimensions(
            old_adapter_name="default",
            new_adapter_name="default_mitigated",
            mitigation_lambda=0.75,
            progressbar=False,
        )

        assert model is not None
        assert "default" in model.peft_config  # Original preserved
        assert "default_mitigated" in model.peft_config  # New adapter created
        assert model.active_adapter == "default_mitigated"  # New adapter is active

    def test_merge_and_unload_destructive(self, tiny_lora_model):
        """Test destructive path (merge and unload)."""
        model = tiny_lora_model

        base_model = model.merge_and_unload_with_reduced_intruder_dimensions(
            adapter_name="default",
            mitigation_lambda=0.75,
            progressbar=False,
        )

        assert base_model is not None
        assert not hasattr(base_model, "peft_config")  # No adapter after unload

    def test_invalid_adapter_name(self, tiny_lora_model):
        """Test that invalid adapter name raises error."""
        model = tiny_lora_model

        with pytest.raises(ValueError, match="Adapter 'nonexistent' not found"):
            model.reduce_intruder_dimensions(old_adapter_name="nonexistent")

    def test_invalid_lambda(self, tiny_lora_model):
        """Test that invalid lambda raises error."""
        model = tiny_lora_model

        with pytest.raises(ValueError, match="mitigation_lambda must be in"):
            model.reduce_intruder_dimensions(mitigation_lambda=1.5)

    def test_invalid_epsilon(self, tiny_lora_model):
        """Test that invalid epsilon raises error."""
        model = tiny_lora_model

        with pytest.raises(ValueError, match="threshold_epsilon must be in"):
            model.reduce_intruder_dimensions(threshold_epsilon=2.0)

    def test_duplicate_adapter_name(self, tiny_lora_model):
        """Test that duplicate adapter name raises error."""
        model = tiny_lora_model

        # Create first mitigated adapter
        model = model.reduce_intruder_dimensions(
            old_adapter_name="default",
            new_adapter_name="mitigated",
        )

        # Try to create another with same name
        with pytest.raises(ValueError, match="already exists"):
            model.reduce_intruder_dimensions(
                old_adapter_name="default",
                new_adapter_name="mitigated",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
