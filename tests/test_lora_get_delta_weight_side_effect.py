"""
Regression tests: Verify LoRA get_delta_weight() has no side effects and merge dtype consistency
"""
import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


class TestGetDeltaWeightNoSideEffect:
    """Test that get_delta_weight() does not modify original weights"""

    def test_linear_get_delta_weight_no_side_effect(self):
        """Test that Linear.get_delta_weight() does not modify original weights"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel().to(torch.bfloat16)
        config = LoraConfig(r=8, lora_alpha=16, target_modules=["linear"], lora_dropout=0.0)
        peft_model = get_peft_model(model, config).to(torch.bfloat16)

        lora_layer = peft_model.model.linear

        with torch.no_grad():
            lora_layer.lora_B["default"].weight.data.fill_(0.5)

        initial_A = lora_layer.lora_A["default"].weight.data.clone()
        initial_B = lora_layer.lora_B["default"].weight.data.clone()

        delta_weight = lora_layer.get_delta_weight("default")

        assert torch.equal(initial_A, lora_layer.lora_A["default"].weight.data), \
            "Linear.get_delta_weight() modified lora_A weights"
        assert torch.equal(initial_B, lora_layer.lora_B["default"].weight.data), \
            "Linear.get_delta_weight() modified lora_B weights"

        assert delta_weight.dtype == torch.bfloat16, \
            f"Returned delta_weight dtype should be bfloat16, got {delta_weight.dtype}"

    def test_embedding_get_delta_weight_no_side_effect(self):
        """Test that Embedding.get_delta_weight() does not modify original weights"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 10)

            def forward(self, x):
                return self.embed(x)

        model = SimpleModel().to(torch.bfloat16)
        config = LoraConfig(r=8, lora_alpha=16, target_modules=["embed"], lora_dropout=0.0)
        peft_model = get_peft_model(model, config).to(torch.bfloat16)

        lora_layer = peft_model.model.embed

        with torch.no_grad():
            lora_layer.lora_embedding_B["default"].data.fill_(0.5)

        initial_A = lora_layer.lora_embedding_A["default"].data.clone()
        initial_B = lora_layer.lora_embedding_B["default"].data.clone()

        delta_weight = lora_layer.get_delta_weight("default")

        assert torch.equal(initial_A, lora_layer.lora_embedding_A["default"].data), \
            "Embedding.get_delta_weight() modified lora_embedding_A weights"
        assert torch.equal(initial_B, lora_layer.lora_embedding_B["default"].data), \
            "Embedding.get_delta_weight() modified lora_embedding_B weights"

    def test_conv_get_delta_weight_no_side_effect(self):
        """Test that _ConvNd.get_delta_weight() does not modify original weights"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 3)

            def forward(self, x):
                return self.conv(x)

        model = SimpleModel().to(torch.bfloat16)
        config = LoraConfig(r=8, lora_alpha=16, target_modules=["conv"], lora_dropout=0.0)
        peft_model = get_peft_model(model, config).to(torch.bfloat16)

        lora_layer = peft_model.model.conv

        with torch.no_grad():
            lora_layer.lora_B["default"].weight.data.fill_(0.5)

        initial_A = lora_layer.lora_A["default"].weight.data.clone()
        initial_B = lora_layer.lora_B["default"].weight.data.clone()

        delta_weight = lora_layer.get_delta_weight("default")

        assert torch.equal(initial_A, lora_layer.lora_A["default"].weight.data), \
            "_ConvNd.get_delta_weight() modified lora_A weights"
        assert torch.equal(initial_B, lora_layer.lora_B["default"].weight.data), \
            "_ConvNd.get_delta_weight() modified lora_B weights"

    def test_get_delta_weight_multiple_calls_consistent(self):
        """Test that multiple calls to get_delta_weight() return identical results"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel().to(torch.bfloat16)
        config = LoraConfig(r=8, lora_alpha=16, target_modules=["linear"], lora_dropout=0.0)
        peft_model = get_peft_model(model, config).to(torch.bfloat16)

        lora_layer = peft_model.model.linear

        with torch.no_grad():
            lora_layer.lora_B["default"].weight.data.fill_(0.5)

        delta_weight_1 = lora_layer.get_delta_weight("default")
        delta_weight_2 = lora_layer.get_delta_weight("default")
        delta_weight_3 = lora_layer.get_delta_weight("default")

        assert torch.equal(delta_weight_1, delta_weight_2), \
            "Multiple calls to get_delta_weight() returned different results"
        assert torch.equal(delta_weight_2, delta_weight_3), \
            "Multiple calls to get_delta_weight() returned different results"


class TestMergeDtypeConsistency:
    """Test dtype consistency of merge operations"""

    def test_linear_merge_unsafe_dtype_consistency(self):
        """Test that Linear.merge() unsafe path preserves dtype"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel().to(torch.bfloat16)
        config = LoraConfig(r=8, lora_alpha=16, target_modules=["linear"], lora_dropout=0.0)
        peft_model = get_peft_model(model, config).to(torch.bfloat16)

        orig_dtype = peft_model.model.linear.base_layer.weight.dtype

        peft_model.merge_adapter(safe_merge=False)

        merged_dtype = peft_model.model.linear.base_layer.weight.dtype
        assert orig_dtype == merged_dtype, \
            f"unsafe merge changed dtype: {orig_dtype} -> {merged_dtype}"

    def test_linear_merge_safe_vs_unsafe_consistency(self):
        """Test that safe merge and unsafe merge produce identical numerical results"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        config = LoraConfig(r=8, lora_alpha=16, target_modules=["linear"], lora_dropout=0.0)

        torch.manual_seed(42)
        model1 = SimpleModel().to(torch.bfloat16)
        peft_model1 = get_peft_model(model1, config).to(torch.bfloat16)

        base_weight = peft_model1.model.linear.base_layer.weight.data.clone()
        lora_A_weight = peft_model1.model.linear.lora_A["default"].weight.data.clone()

        with torch.no_grad():
            peft_model1.model.linear.lora_B["default"].weight.data.fill_(0.5)

        peft_model1.merge_adapter(safe_merge=True)
        safe_merged_weight = peft_model1.model.linear.base_layer.weight.data.clone()

        model2 = SimpleModel().to(torch.bfloat16)
        peft_model2 = get_peft_model(model2, config).to(torch.bfloat16)

        with torch.no_grad():
            peft_model2.model.linear.base_layer.weight.data.copy_(base_weight)
            peft_model2.model.linear.lora_A["default"].weight.data.copy_(lora_A_weight)
            peft_model2.model.linear.lora_B["default"].weight.data.fill_(0.5)

        peft_model2.merge_adapter(safe_merge=False)
        unsafe_merged_weight = peft_model2.model.linear.base_layer.weight.data.clone()

        assert torch.allclose(safe_merged_weight, unsafe_merged_weight, rtol=1e-3, atol=1e-3), \
            "safe merge and unsafe merge produced different numerical results"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
