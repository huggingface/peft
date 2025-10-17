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

# This test file is for tests specific to GraLoRA, since GraLoRA has some specific features
# like block-diagonal structure, hybrid mode, and tensor permutation for information exchange.

import pytest
import torch
from safetensors import safe_open
from torch import nn

from peft import PeftModel, get_peft_model
from peft.tuners.gralora import GraloraConfig


class MLP(nn.Module):
    """Simple MLP for testing"""

    def __init__(self, bias=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.lin1 = nn.Linear(20, 20, bias=bias)  # lin1 and lin2 have same shape
        self.lin2 = nn.Linear(20, 20, bias=bias)
        self.lin3 = nn.Linear(20, 2, bias=bias)
        self.sm = nn.LogSoftmax(dim=-1)

    def forward(self, X):
        X = self.lin0(X)
        X = self.relu(X)
        X = self.lin1(X)
        X = self.relu(X)
        X = self.lin2(X)
        X = self.relu(X)
        X = self.lin3(X)
        X = self.sm(X)
        return X


class TestGralora:
    @pytest.fixture
    def mlp(self):
        torch.manual_seed(0)
        model = MLP()
        return model

    @pytest.fixture
    def mlp_gralora_pure(self, mlp):
        """Pure GraLoRA without hybrid component"""
        torch.manual_seed(0)
        config = GraloraConfig(
            target_modules=["lin1", "lin2"],
            r=16,
            gralora_k=4,
            hybrid_r=0,
            gralora_alpha=32,
            gralora_dropout=0.1,
        )
        peft_model = get_peft_model(mlp, config)
        return peft_model

    @pytest.fixture
    def mlp_gralora_hybrid(self):
        """Hybrid GraLoRA with vanilla LoRA component"""
        torch.manual_seed(0)
        mlp = MLP()
        config = GraloraConfig(
            target_modules=["lin1", "lin2"],
            r=16,
            gralora_k=4,
            hybrid_r=4,
            gralora_alpha=32,
            gralora_dropout=0.1,
        )
        peft_model = get_peft_model(mlp, config)
        return peft_model

    def test_gralora_config_validation(self):
        """Test that config validation works correctly"""
        # Valid config
        config = GraloraConfig(r=16, gralora_k=4, hybrid_r=0)
        assert config.r == 16
        assert config.gralora_k == 4
        assert config.hybrid_r == 0

        # Hybrid config
        config = GraloraConfig(r=16, gralora_k=4, hybrid_r=4)
        assert config.r == 16
        assert config.hybrid_r == 4

    def test_gralora_parameter_shapes(self, mlp_gralora_hybrid):
        """Test that GraLoRA parameters have correct shapes"""
        for name, module in mlp_gralora_hybrid.named_modules():
            if hasattr(module, "gralora_A"):
                adapter_name = "default"
                gralora_A = module.gralora_A[adapter_name]
                gralora_B = module.gralora_B[adapter_name]
                gralora_A_general = module.gralora_A_general[adapter_name]
                gralora_B_general = module.gralora_B_general[adapter_name]

                in_features = module.in_features
                out_features = module.out_features
                k = 4
                gralora_rank = 16 - 4  # r - hybrid_r

                # Check GraLoRA block shapes
                # Each block has full gralora_rank, not gralora_rank // k
                assert gralora_A.shape == (k, in_features // k, gralora_rank)
                assert gralora_B.shape == (k, gralora_rank, out_features // k)

                # Check hybrid component shapes
                assert gralora_A_general.weight.shape == (4, in_features)
                assert gralora_B_general.weight.shape == (out_features, 4)

    def test_gralora_block_diagonal_structure(self):
        """Test that pure GraLoRA produces block-diagonal delta weights"""
        # Use init_weights=False to have non-zero B matrices
        torch.manual_seed(0)
        mlp = MLP()
        config = GraloraConfig(
            target_modules=["lin1", "lin2"],
            r=16,
            gralora_k=4,
            hybrid_r=0,
            init_weights=False,  # Both A and B initialized randomly
        )
        model = get_peft_model(mlp, config)

        for name, module in model.named_modules():
            if hasattr(module, "get_delta_weight"):
                adapter_name = "default"
                delta_weight = module.get_delta_weight(adapter_name)

                k = 4
                in_features = module.in_features
                out_features = module.out_features
                block_size_in = in_features // k
                block_size_out = out_features // k

                # Check diagonal blocks have non-zero values
                for i in range(k):
                    row_start = i * block_size_out
                    row_end = (i + 1) * block_size_out
                    col_start = i * block_size_in
                    col_end = (i + 1) * block_size_in

                    block = delta_weight[row_start:row_end, col_start:col_end]
                    block_norm = torch.norm(block).item()
                    # Diagonal blocks should have some values (initialized with kaiming)
                    assert block_norm > 0, f"Diagonal block [{i},{i}] is zero"

    def test_gralora_forward_pass(self, mlp_gralora_hybrid):
        """Test that forward pass works without errors"""
        mlp_gralora_hybrid.eval()
        x = torch.randn(5, 10)

        with torch.no_grad():
            output = mlp_gralora_hybrid(x)

        assert output.shape == (5, 2)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gralora_backward_pass(self, mlp_gralora_hybrid):
        """Test that backward pass computes gradients correctly"""
        mlp_gralora_hybrid.train()
        x = torch.randn(5, 10)

        output = mlp_gralora_hybrid(x)
        loss = output.sum()
        loss.backward()

        # Check that GraLoRA parameters have gradients
        for name, param in mlp_gralora_hybrid.named_parameters():
            if "gralora" in name and param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"
                assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"

    def test_gralora_pure_vs_hybrid_params(self):
        """Test that pure and hybrid modes have same total parameters but different distribution"""
        torch.manual_seed(0)
        mlp_pure = MLP()
        config_pure = GraloraConfig(
            target_modules=["lin1", "lin2"],
            r=16,
            gralora_k=4,
            hybrid_r=0,
        )
        model_pure = get_peft_model(mlp_pure, config_pure)

        torch.manual_seed(0)
        mlp_hybrid = MLP()
        config_hybrid = GraloraConfig(
            target_modules=["lin1", "lin2"],
            r=16,
            gralora_k=4,
            hybrid_r=4,
        )
        model_hybrid = get_peft_model(mlp_hybrid, config_hybrid)

        def count_trainable_params(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        params_pure = count_trainable_params(model_pure)
        params_hybrid = count_trainable_params(model_hybrid)

        # Pure and hybrid should have same total parameters (r is constant)
        # but distributed differently between block-diagonal and full-rank components
        assert params_pure == params_hybrid, (
            f"Pure ({params_pure}) and Hybrid ({params_hybrid}) should have same parameter count"
        )

        # Check that hybrid has general components
        has_general = False
        for name, _ in model_hybrid.named_modules():
            if "gralora_A_general" in name or "gralora_B_general" in name:
                has_general = True
                break
        assert has_general, "Hybrid mode should have general components"

    def test_gralora_save_load_roundtrip(self, mlp_gralora_hybrid, tmp_path):
        """Test that save/load preserves model behavior"""
        mlp_gralora_hybrid.eval()
        x = torch.randn(5, 10)

        # Get output before save
        with torch.no_grad():
            output_before = mlp_gralora_hybrid(x)

        # Save adapter
        mlp_gralora_hybrid.save_pretrained(tmp_path)

        # Load adapter
        torch.manual_seed(0)
        new_mlp = MLP()
        loaded_model = PeftModel.from_pretrained(new_mlp, tmp_path)
        loaded_model.eval()

        # Get output after load
        with torch.no_grad():
            output_after = loaded_model(x)

        # Outputs should be very close
        assert torch.allclose(output_before, output_after, atol=1e-5, rtol=1e-5)

    def test_gralora_state_dict_structure(self, mlp_gralora_hybrid, tmp_path):
        """Test that state dict contains only necessary parameters"""
        mlp_gralora_hybrid.save_pretrained(tmp_path)

        # Load state dict
        sd = {}
        with safe_open(tmp_path / "adapter_model.safetensors", framework="pt", device="cpu") as f:
            for key in f.keys():
                sd[key] = f.get_tensor(key)

        # Check that gralora parameters are present
        assert any("gralora_A" in key for key in sd), "gralora_A not found in state dict"
        assert any("gralora_B" in key for key in sd), "gralora_B not found in state dict"

        # For hybrid mode, check hybrid components
        assert any("gralora_A_general" in key for key in sd), "gralora_A_general not found"
        assert any("gralora_B_general" in key for key in sd), "gralora_B_general not found"

    def test_gralora_merge_and_unload(self, mlp_gralora_hybrid):
        """Test merge_and_unload functionality"""
        mlp_gralora_hybrid.eval()
        x = torch.randn(5, 10)

        # Get output before merge
        with torch.no_grad():
            output_before = mlp_gralora_hybrid(x)

        # Merge and unload
        merged_model = mlp_gralora_hybrid.merge_and_unload()
        merged_model.eval()

        # Get output after merge
        with torch.no_grad():
            output_after = merged_model(x)

        # Outputs should be very close
        assert torch.allclose(output_before, output_after, atol=1e-4, rtol=1e-4)

        # Check that merged model has no GraLoRA layers
        has_gralora = any("gralora" in name for name, _ in merged_model.named_parameters())
        assert not has_gralora, "Merged model still has GraLoRA parameters"

    def test_gralora_merge_unmerge(self):
        """Test merge/unmerge functionality"""
        torch.manual_seed(0)
        mlp = MLP()
        config = GraloraConfig(
            target_modules=["lin1"],
            r=8,
            gralora_k=2,
            hybrid_r=0,
        )
        model = get_peft_model(mlp, config)
        model.eval()

        x = torch.randn(5, 10)

        # Output before merge
        with torch.no_grad():
            output_before = model(x)

        # Merge adapter using PEFT API
        model.merge_adapter()

        with torch.no_grad():
            output_merged = model(x)

        # Outputs should be the same after merge
        assert torch.allclose(output_before, output_merged, atol=1e-4, rtol=1e-4)

        # Unmerge adapter using PEFT API
        model.unmerge_adapter()

        with torch.no_grad():
            output_unmerged = model(x)

        # Outputs should be the same after unmerge
        assert torch.allclose(output_before, output_unmerged, atol=1e-4, rtol=1e-4)

    def test_gralora_multiple_adapters(self):
        """Test adding and switching between multiple adapters"""
        torch.manual_seed(0)
        mlp = MLP()

        # Use init_weights=False to have non-zero outputs
        config1 = GraloraConfig(target_modules=["lin1"], r=8, gralora_k=2, hybrid_r=0, init_weights=False)
        model = get_peft_model(mlp, config1, adapter_name="adapter1")

        torch.manual_seed(42)  # Different seed for second adapter
        config2 = GraloraConfig(target_modules=["lin1"], r=8, gralora_k=2, hybrid_r=0, init_weights=False)
        model.add_adapter("adapter2", config2)

        x = torch.randn(5, 10)

        # Test adapter1
        model.set_adapter("adapter1")
        with torch.no_grad():
            output1 = model(x)

        # Test adapter2
        model.set_adapter("adapter2")
        with torch.no_grad():
            output2 = model(x)

        # Different adapters should give different outputs
        assert not torch.allclose(output1, output2, atol=1e-3, rtol=1e-3)

    def test_gralora_dtype_compatibility(self):
        """Test that GraLoRA works with different dtypes"""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            if dtype == torch.bfloat16 and not torch.cuda.is_available():
                # Skip bfloat16 on CPU if not supported
                continue

            torch.manual_seed(0)
            mlp = MLP().to(dtype)
            config = GraloraConfig(
                target_modules=["lin1"],
                r=8,
                gralora_k=2,
                hybrid_r=0,
            )
            model = get_peft_model(mlp, config)

            x = torch.randn(5, 10).to(dtype)
            output = model(x)

            assert output.dtype == dtype, f"Output dtype mismatch for {dtype}"

    def test_gralora_disable_adapters(self):
        """Test disabling adapters"""
        torch.manual_seed(0)
        mlp = MLP()
        # Use init_weights=False to have non-zero effect
        config = GraloraConfig(
            target_modules=["lin1", "lin2"],
            r=16,
            gralora_k=4,
            hybrid_r=4,
            init_weights=False,
        )
        model = get_peft_model(mlp, config)
        model.eval()
        x = torch.randn(5, 10)

        # Output with adapter enabled
        with torch.no_grad():
            output_enabled = model(x)

        # Output with adapter disabled
        with model.disable_adapter():
            with torch.no_grad():
                output_disabled = model(x)

        # Outputs should be different
        assert not torch.allclose(output_enabled, output_disabled, atol=1e-6, rtol=1e-6)

    def test_gralora_different_k_values(self):
        """Test GraLoRA with different k values"""
        for k in [2, 4]:
            torch.manual_seed(0)
            mlp = MLP()
            config = GraloraConfig(
                target_modules=["lin1", "lin2"],
                r=k * 4,  # Make sure r is divisible by k
                gralora_k=k,
                hybrid_r=0,
            )
            model = get_peft_model(mlp, config)

            x = torch.randn(5, 10)
            output = model(x)

            assert output.shape == (5, 2)
            assert not torch.isnan(output).any()

    def test_gralora_rank_divisibility_check(self):
        """Test that invalid rank/k combinations raise errors"""
        torch.manual_seed(0)
        mlp = MLP()

        # This should raise an error because (r - hybrid_r) is not divisible by k
        # r=15, hybrid_r=0, k=4 -> gralora_rank=15, 15 % 4 != 0
        config = GraloraConfig(
            target_modules=["lin1"],
            r=15,
            gralora_k=4,
            hybrid_r=0,
        )

        with pytest.raises(AssertionError, match="r should be divisible by gralora_k"):
            get_peft_model(mlp, config)

    def test_gralora_trainable_parameters_only(self, mlp_gralora_hybrid):
        """Test that only GraLoRA parameters are trainable"""
        for name, param in mlp_gralora_hybrid.named_parameters():
            if "gralora" in name or "modules_to_save" in name:
                assert param.requires_grad, f"GraLoRA parameter {name} should be trainable"
            else:
                assert not param.requires_grad, f"Base parameter {name} should be frozen"

    def test_gralora_save_pretrained_files(self, mlp_gralora_hybrid, tmp_path):
        """Test that save_pretrained creates expected files"""
        mlp_gralora_hybrid.save_pretrained(tmp_path)

        # Check for config file
        assert (tmp_path / "adapter_config.json").exists()

        # Check for weights file (either .bin or .safetensors)
        assert (tmp_path / "adapter_model.safetensors").exists() or (tmp_path / "adapter_model.bin").exists()

    def test_gralora_information_exchange_via_permutation(self, mlp_gralora_pure):
        """
        Test that information exchange happens through tensor permutation. Even though delta weights are
        block-diagonal, the forward pass should allow information flow between blocks via the permutation operation.
        """
        mlp_gralora_pure.eval()

        # Create two inputs that differ only in specific blocks
        x1 = torch.randn(1, 10)
        x2 = x1.clone()

        # Modify only the first block (assuming k=4, block size = 10//4 = 2.5, rounded to 2-3 features)
        x2[0, :5] += 1.0  # Modify first block

        with torch.no_grad():
            out1 = mlp_gralora_pure(x1)
            out2 = mlp_gralora_pure(x2)

        # Due to information exchange, changing one block should affect all outputs
        # (not just outputs corresponding to that block)
        diff = (out1 - out2).abs()

        # All output dimensions should be affected (not just the first block's outputs)
        assert (diff > 1e-6).all(), "Information exchange not happening correctly"

    def test_gralora_scaling_factor(self):
        """Test that scaling factor is correctly applied"""
        torch.manual_seed(0)
        mlp = MLP()

        # Create two configs with different alpha values
        config_alpha16 = GraloraConfig(
            target_modules=["lin1"],
            r=8,
            gralora_alpha=16,
            gralora_k=2,
            hybrid_r=0,
        )

        config_alpha32 = GraloraConfig(
            target_modules=["lin1"],
            r=8,
            gralora_alpha=32,
            gralora_k=2,
            hybrid_r=0,
        )

        model_alpha16 = get_peft_model(MLP(), config_alpha16)
        model_alpha32 = get_peft_model(MLP(), config_alpha32)

        # Copy weights to make them identical except for scaling
        for (n1, p1), (n2, p2) in zip(model_alpha16.named_parameters(), model_alpha32.named_parameters()):
            if "gralora" in n1:
                p2.data = p1.data.clone()

        x = torch.randn(5, 10)

        model_alpha16.eval()
        model_alpha32.eval()

        with torch.no_grad():
            out1 = model_alpha16(x)
            out2 = model_alpha32(x)

        # Outputs should be different due to different scaling
        assert not torch.allclose(out1, out2, atol=1e-6, rtol=1e-6)

    def test_gralora_safe_merge_success(self):
        """Test safe_merge with valid weights"""
        torch.manual_seed(0)
        mlp = MLP()
        config = GraloraConfig(
            target_modules=["lin1"],
            r=8,
            gralora_k=2,
            hybrid_r=0,
            init_weights=False,
        )
        model = get_peft_model(mlp, config)

        x = torch.randn(5, 10)
        with torch.no_grad():
            output_before = model(x)

        # Test safe merge
        model.base_model.model.lin1.merge(safe_merge=True)

        with torch.no_grad():
            output_after = model(x)

        assert torch.allclose(output_before, output_after, atol=1e-4, rtol=1e-4)

    def test_gralora_safe_merge_detects_nan(self):
        """Test that safe_merge detects NaN values"""
        torch.manual_seed(0)
        mlp = MLP()
        config = GraloraConfig(
            target_modules=["lin1"],
            r=8,
            gralora_k=2,
            hybrid_r=0,
        )
        model = get_peft_model(mlp, config)

        # Inject NaN into adapter weights (use .data to avoid requires_grad error)
        model.base_model.model.lin1.gralora_A["default"].data[0, 0, 0] = float("nan")

        # safe_merge should raise ValueError
        with pytest.raises(ValueError, match="NaNs detected"):
            model.base_model.model.lin1.merge(safe_merge=True)

    def test_gralora_unmerge_warning_when_not_merged(self):
        """Test that unmerge warns when already unmerged"""
        torch.manual_seed(0)
        mlp = MLP()
        config = GraloraConfig(target_modules=["lin1"], r=8, gralora_k=2)
        model = get_peft_model(mlp, config)

        # Try to unmerge without merging first
        with pytest.warns(UserWarning, match="Already unmerged"):
            model.base_model.model.lin1.unmerge()

    def test_gralora_hybrid_forward_computation(self):
        """Test that hybrid LoRA component is used in forward pass"""
        torch.manual_seed(0)
        mlp_hybrid = MLP()
        mlp_pure = MLP()

        config_hybrid = GraloraConfig(
            target_modules=["lin1"],
            r=16,
            gralora_k=4,
            hybrid_r=4,
            init_weights=False,
        )
        model_hybrid = get_peft_model(mlp_hybrid, config_hybrid)

        config_pure = GraloraConfig(
            target_modules=["lin1"],
            r=16,
            gralora_k=4,
            hybrid_r=0,
            init_weights=False,
        )
        model_pure = get_peft_model(mlp_pure, config_pure)

        x = torch.randn(5, 10)

        with torch.no_grad():
            output_hybrid = model_hybrid(x)
            output_pure = model_pure(x)

        # Outputs should be different due to hybrid component
        assert not torch.allclose(output_hybrid, output_pure, atol=1e-3)

    def test_gralora_invalid_rank_zero(self):
        """Test that r=0 raises error"""
        mlp = MLP()
        config = GraloraConfig(target_modules=["lin1"], r=0, gralora_k=2)

        with pytest.raises(ValueError, match="`r` should be a positive integer"):
            get_peft_model(mlp, config)

    def test_gralora_invalid_rank_negative(self):
        """Test that negative r raises error"""
        mlp = MLP()
        config = GraloraConfig(target_modules=["lin1"], r=-1, gralora_k=2)

        with pytest.raises(ValueError, match="`r` should be a positive integer"):
            get_peft_model(mlp, config)

    def test_gralora_bias_all(self):
        """Test bias='all' configuration"""
        torch.manual_seed(0)
        mlp = MLP(bias=True)
        config = GraloraConfig(
            target_modules=["lin1"],
            r=8,
            gralora_k=2,
            bias="all",
        )
        model = get_peft_model(mlp, config)

        # Check that all bias parameters are trainable
        bias_params = [name for name, param in model.named_parameters() if "bias" in name and param.requires_grad]
        assert len(bias_params) > 0, "At least some bias parameters should be trainable"

    def test_gralora_bias_gralora_only(self):
        """Test bias='gralora_only' configuration"""
        torch.manual_seed(0)
        mlp = MLP(bias=True)
        config = GraloraConfig(
            target_modules=["lin1"],
            r=8,
            gralora_k=2,
            bias="gralora_only",
        )
        model = get_peft_model(mlp, config)

        # Only GraLoRA layer biases should be trainable
        assert model.base_model.model.lin1.bias.requires_grad
        assert not model.base_model.model.lin0.bias.requires_grad

    def test_gralora_multiple_adapters_with_bias_raises(self):
        """Test that multiple adapters with bias raises error"""
        torch.manual_seed(0)
        mlp = MLP()
        config1 = GraloraConfig(target_modules=["lin1"], r=8, gralora_k=2, bias="all")
        model = get_peft_model(mlp, config1)

        config2 = GraloraConfig(target_modules=["lin1"], r=8, gralora_k=2, bias="all")

        with pytest.raises(ValueError, match="supports only 1 adapter with bias"):
            model.add_adapter("adapter2", config2)

    def test_gralora_cpu_fp16_merge(self):
        """Test merge with fp16 on CPU"""
        torch.manual_seed(0)
        mlp = MLP().to(torch.float16)
        config = GraloraConfig(
            target_modules=["lin1"],
            r=8,
            gralora_k=2,
            hybrid_r=0,
            init_weights=False,
        )
        model = get_peft_model(mlp, config)

        x = torch.randn(5, 10).to(torch.float16)

        with torch.no_grad():
            output_before = model(x)

        # Merge (should handle CPU fp16 correctly)
        model.merge_adapter()

        with torch.no_grad():
            output_after = model(x)

        assert torch.allclose(output_before, output_after, atol=1e-2, rtol=1e-2)

    def test_gralora_cpu_bf16_merge(self):
        """Test merge with bf16 on CPU (if supported)"""
        # Check if bfloat16 is supported
        try:
            _ = torch.randn(2, 2).to(torch.bfloat16)
        except RuntimeError:
            pytest.skip("bfloat16 not supported on this system")

        torch.manual_seed(0)
        mlp = MLP().to(torch.bfloat16)
        config = GraloraConfig(
            target_modules=["lin1"],
            r=8,
            gralora_k=2,
            hybrid_r=2,
            init_weights=False,
        )
        model = get_peft_model(mlp, config)

        x = torch.randn(5, 10).to(torch.bfloat16)

        with torch.no_grad():
            output_before = model(x)

        # Merge with hybrid component
        model.merge_adapter()

        with torch.no_grad():
            output_after = model(x)

        assert torch.allclose(output_before, output_after, atol=1e-2, rtol=1e-2)

    def test_gralora_disable_adapter_layers_warns_with_bias(self):
        """Test that disable_adapter_layers warns when bias is configured"""
        torch.manual_seed(0)
        mlp = MLP()
        config = GraloraConfig(
            target_modules=["lin1"],
            r=8,
            gralora_k=2,
            bias="all",
        )
        model = get_peft_model(mlp, config)

        with pytest.warns(UserWarning, match="disabling adapter layers with bias"):
            model.disable_adapter_layers()

    def test_gralora_set_adapter_warns_when_merged(self):
        """Test that set_adapter warns and unmerges when model is merged"""
        torch.manual_seed(0)
        mlp = MLP()
        config1 = GraloraConfig(target_modules=["lin1"], r=8, gralora_k=2)
        model = get_peft_model(mlp, config1, adapter_name="adapter1")

        config2 = GraloraConfig(target_modules=["lin1"], r=8, gralora_k=2)
        model.add_adapter("adapter2", config2)

        # Merge first adapter
        model.merge_adapter()

        # Setting adapter should warn and unmerge
        with pytest.warns(UserWarning, match="Adapter cannot be set when the model is merged"):
            model.set_adapter("adapter2")

        # Model should be unmerged now
        assert not model.base_model.model.lin1.merged

    def test_gralora_delete_adapter(self):
        """Test deleting an adapter"""
        torch.manual_seed(0)
        mlp = MLP()
        config = GraloraConfig(target_modules=["lin1"], r=8, gralora_k=2)
        model = get_peft_model(mlp, config, adapter_name="adapter1")

        config2 = GraloraConfig(target_modules=["lin1"], r=8, gralora_k=2)
        model.add_adapter("adapter2", config2)

        # Delete adapter1
        model.delete_adapter("adapter1")

        assert "adapter1" not in model.peft_config
        assert "adapter2" in model.peft_config

    def test_gralora_delete_nonexistent_adapter_raises(self):
        """Test that deleting nonexistent adapter raises error"""
        torch.manual_seed(0)
        mlp = MLP()
        config = GraloraConfig(target_modules=["lin1"], r=8, gralora_k=2)
        model = get_peft_model(mlp, config)

        with pytest.raises(ValueError, match="Adapter .* does not exist"):
            model.delete_adapter("nonexistent")

    def test_gralora_unload_without_merge(self):
        """Test unload without merging"""
        torch.manual_seed(0)
        mlp = MLP()
        config = GraloraConfig(
            target_modules=["lin1"],
            r=8,
            gralora_k=2,
            init_weights=False,
        )
        model = get_peft_model(mlp, config)

        x = torch.randn(5, 10)

        # Get base model output
        with model.disable_adapter():
            with torch.no_grad():
                base_output = model(x)

        # Unload without merge
        unloaded_model = model.unload()

        with torch.no_grad():
            unloaded_output = unloaded_model(x)

        # Should match base model output (no merge)
        assert torch.allclose(base_output, unloaded_output, atol=1e-5)

    def test_gralora_get_peft_config_as_dict(self):
        """Test get_peft_config_as_dict method"""
        torch.manual_seed(0)
        mlp = MLP()
        config = GraloraConfig(
            target_modules=["lin1"],
            r=8,
            gralora_k=2,
            hybrid_r=4,
            gralora_alpha=16,
        )
        model = get_peft_model(mlp, config)

        config_dict = model.get_peft_config_as_dict(inference=False)

        assert "default" in config_dict
        assert config_dict["default"]["r"] == 8
        assert config_dict["default"]["gralora_k"] == 2
        assert config_dict["default"]["hybrid_r"] == 4

    def test_gralora_get_peft_config_as_dict_inference_mode(self):
        """Test get_peft_config_as_dict with inference=True"""
        torch.manual_seed(0)
        mlp = MLP()
        config = GraloraConfig(target_modules=["lin1"], r=8, gralora_k=2)
        model = get_peft_model(mlp, config)

        config_dict = model.get_peft_config_as_dict(inference=True)

        assert config_dict["default"]["inference_mode"] is True

    def test_gralora_merge_with_hybrid_component(self):
        """Test that merge works correctly with hybrid component"""
        torch.manual_seed(0)
        mlp = MLP()
        config = GraloraConfig(
            target_modules=["lin1"],
            r=16,
            gralora_k=4,
            hybrid_r=4,
            init_weights=False,
        )
        model = get_peft_model(mlp, config)

        x = torch.randn(5, 10)

        with torch.no_grad():
            output_before = model(x)

        # Merge
        model.merge_adapter()

        with torch.no_grad():
            output_after = model(x)

        # Outputs should be very close
        assert torch.allclose(output_before, output_after, atol=1e-4, rtol=1e-4)

    def test_gralora_repr(self):
        """Test __repr__ method"""
        torch.manual_seed(0)
        mlp = MLP()
        config = GraloraConfig(target_modules=["lin1"], r=8, gralora_k=2)
        model = get_peft_model(mlp, config)

        repr_str = repr(model.base_model.model.lin1)
        assert "gralora" in repr_str.lower()

    def test_gralora_merge_with_adapter_names(self):
        """Test merge with specific adapter names"""
        torch.manual_seed(0)
        mlp = MLP()
        config1 = GraloraConfig(target_modules=["lin1"], r=8, gralora_k=2, init_weights=False)
        model = get_peft_model(mlp, config1, adapter_name="adapter1")

        torch.manual_seed(42)
        config2 = GraloraConfig(target_modules=["lin1"], r=8, gralora_k=2, init_weights=False)
        model.add_adapter("adapter2", config2)

        x = torch.randn(5, 10)

        # Set to adapter1 and get output
        model.set_adapter("adapter1")
        with torch.no_grad():
            output_before = model(x)

        # Merge only adapter1
        model.base_model.model.lin1.merge(adapter_names=["adapter1"])

        with torch.no_grad():
            output_after = model(x)

        # Outputs should be close
        assert torch.allclose(output_before, output_after, atol=1e-4, rtol=1e-4)

    def test_gralora_enable_disable_adapter_layers(self):
        """Test enable/disable adapter layers"""
        torch.manual_seed(0)
        mlp = MLP()
        config = GraloraConfig(
            target_modules=["lin1"],
            r=8,
            gralora_k=2,
            init_weights=False,
        )
        model = get_peft_model(mlp, config)

        x = torch.randn(5, 10)

        # Get output with adapter enabled
        with torch.no_grad():
            output_enabled = model(x)

        # Disable adapters
        model.disable_adapter_layers()

        with torch.no_grad():
            output_disabled = model(x)

        # Enable adapters
        model.enable_adapter_layers()

        with torch.no_grad():
            output_re_enabled = model(x)

        # Output with disabled adapter should be different
        assert not torch.allclose(output_enabled, output_disabled, atol=1e-6)
        # Output after re-enabling should match original
        assert torch.allclose(output_enabled, output_re_enabled, atol=1e-6)

    def test_gralora_forward_with_merged_adapter(self):
        """Test forward pass with merged adapter"""
        torch.manual_seed(0)
        mlp = MLP()
        config = GraloraConfig(
            target_modules=["lin1"],
            r=8,
            gralora_k=2,
            init_weights=False,
        )
        model = get_peft_model(mlp, config)

        x = torch.randn(5, 10)

        # Get output before merge
        with torch.no_grad():
            output_before = model(x)

        # Merge adapter
        model.merge_adapter()

        # Forward with merged adapter (should take merged path)
        with torch.no_grad():
            output_after = model(x)

        assert torch.allclose(output_before, output_after, atol=1e-4)

    def test_gralora_forward_with_disable_adapters_and_merged(self):
        """Test forward when disable_adapters=True and model is merged"""
        torch.manual_seed(0)
        mlp = MLP()
        config = GraloraConfig(
            target_modules=["lin1"],
            r=8,
            gralora_k=2,
            init_weights=False,
        )
        model = get_peft_model(mlp, config)

        x = torch.randn(5, 10)

        # Merge adapter
        model.merge_adapter()

        # Get output with merged adapter
        with torch.no_grad():
            output_merged = model(x)

        # Disable adapters (should unmerge)
        with model.disable_adapter():
            with torch.no_grad():
                output_disabled = model(x)

        # Outputs should be different
        assert not torch.allclose(output_merged, output_disabled, atol=1e-5)

    def test_gralora_bias_invalid_option_raises(self):
        """Test that invalid bias option raises NotImplementedError"""
        torch.manual_seed(0)
        mlp = MLP()

        # Create config with invalid bias (need to bypass validation)
        config = GraloraConfig(target_modules=["lin1"], r=8, gralora_k=2)
        model = get_peft_model(mlp, config)

        # Manually set invalid bias to trigger the error
        model.peft_config["default"].bias = "invalid_option"

        with pytest.raises(NotImplementedError, match="Requested bias"):
            model._mark_only_adapters_as_trainable(model.model)

    def test_gralora_merge_empty_adapter_names(self):
        """Test merge with empty adapter_names returns early"""
        torch.manual_seed(0)
        mlp = MLP()
        config = GraloraConfig(target_modules=["lin1"], r=8, gralora_k=2)
        model = get_peft_model(mlp, config)

        # Call merge with empty list (should return early)
        model.base_model.model.lin1.merge(adapter_names=[])

        # Model should not be merged
        assert not model.base_model.model.lin1.merged

    def test_gralora_add_non_active_adapter(self):
        """Test adding adapter that is not active (should not be trainable)"""
        torch.manual_seed(0)
        mlp = MLP()
        config1 = GraloraConfig(target_modules=["lin1"], r=8, gralora_k=2)
        model = get_peft_model(mlp, config1, adapter_name="adapter1")

        # Keep adapter1 active
        model.set_adapter("adapter1")

        # Add adapter2 (should not be active/trainable initially)
        torch.manual_seed(42)
        config2 = GraloraConfig(target_modules=["lin1"], r=8, gralora_k=2)
        model.add_adapter("adapter2", config2)

        # adapter2 parameters should exist but might not be in active_adapters initially
        assert "adapter2" in model.base_model.model.lin1.gralora_A

    def test_gralora_forward_with_no_adapter_in_active_list(self):
        """Test forward when active_adapter is not in gralora_A keys"""
        torch.manual_seed(0)
        mlp = MLP()
        config = GraloraConfig(target_modules=["lin1"], r=8, gralora_k=2)
        model = get_peft_model(mlp, config, adapter_name="adapter1")

        x = torch.randn(5, 10)

        # Manually set _active_adapter to include non-existent adapter
        original_adapter = model.base_model.model.lin1._active_adapter
        model.base_model.model.lin1._active_adapter = ["nonexistent", "adapter1"]

        # Should still work (skip nonexistent adapter)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (5, 2)

        # Restore
        model.base_model.model.lin1._active_adapter = original_adapter
