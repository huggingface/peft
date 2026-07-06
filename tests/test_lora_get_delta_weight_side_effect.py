"""
回归测试：验证 LoRA get_delta_weight() 无副作用和 merge dtype 一致性
"""
import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


class TestGetDeltaWeightNoSideEffect:
    """测试 get_delta_weight() 不修改原始权重"""

    def test_linear_get_delta_weight_no_side_effect(self):
        """测试 Linear.get_delta_weight() 不修改原始权重"""
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

        # 给 lora_B 一些非零值
        with torch.no_grad():
            lora_layer.lora_B["default"].weight.data.fill_(0.5)

        # 记录初始权重
        initial_A = lora_layer.lora_A["default"].weight.data.clone()
        initial_B = lora_layer.lora_B["default"].weight.data.clone()

        # 调用 get_delta_weight
        delta_weight = lora_layer.get_delta_weight("default")

        # 验证权重未被修改
        assert torch.equal(initial_A, lora_layer.lora_A["default"].weight.data), \
            "Linear.get_delta_weight() 修改了 lora_A 权重"
        assert torch.equal(initial_B, lora_layer.lora_B["default"].weight.data), \
            "Linear.get_delta_weight() 修改了 lora_B 权重"

        # 验证返回值的 dtype
        assert delta_weight.dtype == torch.bfloat16, \
            f"返回的 delta_weight dtype 应该是 bfloat16，实际是 {delta_weight.dtype}"

    def test_embedding_get_delta_weight_no_side_effect(self):
        """测试 Embedding.get_delta_weight() 不修改原始权重"""
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

        # 给 lora_embedding_B 一些非零值
        with torch.no_grad():
            lora_layer.lora_embedding_B["default"].data.fill_(0.5)

        # 记录初始权重
        initial_A = lora_layer.lora_embedding_A["default"].data.clone()
        initial_B = lora_layer.lora_embedding_B["default"].data.clone()

        # 调用 get_delta_weight
        delta_weight = lora_layer.get_delta_weight("default")

        # 验证权重未被修改
        assert torch.equal(initial_A, lora_layer.lora_embedding_A["default"].data), \
            "Embedding.get_delta_weight() 修改了 lora_embedding_A 权重"
        assert torch.equal(initial_B, lora_layer.lora_embedding_B["default"].data), \
            "Embedding.get_delta_weight() 修改了 lora_embedding_B 权重"

    def test_conv_get_delta_weight_no_side_effect(self):
        """测试 _ConvNd.get_delta_weight() 不修改原始权重"""
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

        # 给 lora_B 一些非零值
        with torch.no_grad():
            lora_layer.lora_B["default"].weight.data.fill_(0.5)

        # 记录初始权重
        initial_A = lora_layer.lora_A["default"].weight.data.clone()
        initial_B = lora_layer.lora_B["default"].weight.data.clone()

        # 调用 get_delta_weight
        delta_weight = lora_layer.get_delta_weight("default")

        # 验证权重未被修改
        assert torch.equal(initial_A, lora_layer.lora_A["default"].weight.data), \
            "_ConvNd.get_delta_weight() 修改了 lora_A 权重"
        assert torch.equal(initial_B, lora_layer.lora_B["default"].weight.data), \
            "_ConvNd.get_delta_weight() 修改了 lora_B 权重"

    def test_get_delta_weight_multiple_calls_consistent(self):
        """测试多次调用 get_delta_weight() 结果一致"""
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

        # 给 lora_B 一些非零值
        with torch.no_grad():
            lora_layer.lora_B["default"].weight.data.fill_(0.5)

        # 多次调用 get_delta_weight
        delta_weight_1 = lora_layer.get_delta_weight("default")
        delta_weight_2 = lora_layer.get_delta_weight("default")
        delta_weight_3 = lora_layer.get_delta_weight("default")

        # 验证结果一致
        assert torch.equal(delta_weight_1, delta_weight_2), \
            "多次调用 get_delta_weight() 结果不一致"
        assert torch.equal(delta_weight_2, delta_weight_3), \
            "多次调用 get_delta_weight() 结果不一致"


class TestMergeDtypeConsistency:
    """测试 merge 操作的 dtype 一致性"""

    def test_linear_merge_unsafe_dtype_consistency(self):
        """测试 Linear.merge() unsafe path 保持 dtype 一致"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel().to(torch.bfloat16)
        config = LoraConfig(r=8, lora_alpha=16, target_modules=["linear"], lora_dropout=0.0)
        peft_model = get_peft_model(model, config).to(torch.bfloat16)

        # 记录原始 dtype
        orig_dtype = peft_model.model.linear.base_layer.weight.dtype

        # 执行 unsafe merge
        peft_model.merge_adapter(safe_merge=False)

        # 验证 dtype 保持一致
        merged_dtype = peft_model.model.linear.base_layer.weight.dtype
        assert orig_dtype == merged_dtype, \
            f"unsafe merge 改变了 dtype: {orig_dtype} -> {merged_dtype}"

    def test_linear_merge_safe_vs_unsafe_consistency(self):
        """测试 safe merge 和 unsafe merge 结果数值一致"""
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
            "safe merge 和 unsafe merge 结果数值不一致"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
