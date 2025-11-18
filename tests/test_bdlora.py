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
from torch import nn

from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.config import BdLoraConfig
from peft.tuners.lora.layer import Linear
from peft.tuners.lora.variants import BlockDiagonalLinear


class MLP(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.lin1 = nn.Linear(20, 20, bias=bias)
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


class TestBdLora:
    @pytest.fixture
    def mlp(self):
        torch.manual_seed(0)
        model = MLP()
        return model

    def test_bdlora_config_creation(self):
        """Test that BdLoraConfig can be created with proper parameters."""
        bdlora_config = BdLoraConfig(
            target_modules_bd_a=["lin1"],
            target_modules_bd_b=["lin2"],
            nblocks=4,
        )
        config = LoraConfig(
            r=16,
            target_modules=["lin1", "lin2"],
            use_bdlora=bdlora_config,
        )
        assert config.r == 16
        assert set(config.target_modules) == {"lin1", "lin2"}
        assert config.use_bdlora.target_modules_bd_a == ["lin1"]
        assert config.use_bdlora.target_modules_bd_b == ["lin2"]
        assert config.use_bdlora.nblocks == 4

    def test_bdlora_model_creation(self, mlp):
        """Test that BD-LoRA model can be created and has trainable parameters."""
        config = LoraConfig(
            r=8,
            target_modules=["lin1", "lin2"],
            use_bdlora=BdLoraConfig(
                target_modules_bd_a=["lin1"],
                target_modules_bd_b=["lin2"],
                nblocks=4,
            ),
        )
        peft_model = get_peft_model(mlp, config)

        # Check that model has trainable parameters
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        assert trainable_params > 0

        # Check that correct layer types are created
        assert isinstance(peft_model.base_model.model.lin1, Linear)
        assert isinstance(peft_model.base_model.model.lin2, Linear)

        # Check that block diagonal layers are used
        assert isinstance(peft_model.base_model.model.lin1.lora_A["default"], BlockDiagonalLinear)
        assert isinstance(peft_model.base_model.model.lin2.lora_B["default"], BlockDiagonalLinear)

    def test_block_diagonal_linear_forward(self):
        """Test BlockDiagonalLinear forward pass."""
        nblocks = 4
        in_features = 20
        out_features = 16

        layer = BlockDiagonalLinear(in_features, out_features, nblocks, bias=False)  # No bias
        x = torch.randn(2, in_features)

        output = layer(x)
        assert output.shape == (2, out_features)

    def test_block_diagonal_linear_shapes(self):
        """Test BlockDiagonalLinear weight shapes."""
        nblocks = 4
        in_features = 20
        out_features = 16

        layer = BlockDiagonalLinear(in_features, out_features, nblocks, bias=False)  # No bias

        expected_shape = (out_features, in_features // nblocks)
        assert layer.weight.shape == expected_shape

    def test_bdlora_forward_pass(self, mlp):
        """Test that BD-LoRA model can perform forward pass."""
        config = LoraConfig(
            r=8,
            target_modules=["lin1", "lin2"],
            use_bdlora=BdLoraConfig(
                target_modules_bd_a=["lin1"],
                target_modules_bd_b=["lin2"],
                nblocks=4,
            ),
        )
        peft_model = get_peft_model(mlp, config)

        x = torch.randn(3, 10)
        output = peft_model(x)

        assert output.shape == (3, 2)
        assert not torch.isnan(output).any()

    def test_bdlora_save_load(self, mlp, tmp_path):
        """Test BD-LoRA model save and load functionality."""
        torch.manual_seed(42)
        config = LoraConfig(
            r=8,
            target_modules=["lin1", "lin2"],
            use_bdlora=BdLoraConfig(
                target_modules_bd_a=["lin1"],
                target_modules_bd_b=["lin2"],
                nblocks=4,
            ),
        )
        peft_model = get_peft_model(mlp, config)

        torch.manual_seed(42)
        x = torch.randn(2, 10)
        output_before = peft_model(x)

        save_path = tmp_path / "bdlora_test"
        peft_model.save_pretrained(save_path)

        torch.manual_seed(0)
        base_model = MLP()
        loaded_model = PeftModel.from_pretrained(base_model, save_path)

        output_after = loaded_model(x)
        assert torch.allclose(output_before, output_after, atol=1e-5)

    def test_bdlora_different_nblocks(self):
        """Test BD-LoRA with different nblocks values."""
        for nblocks in [2, 4]:
            torch.manual_seed(0)
            mlp = MLP()
            config = LoraConfig(
                r=16,
                target_modules=["lin1"],
                use_bdlora=BdLoraConfig(target_modules_bd_a=["lin1"], nblocks=nblocks),
            )
            peft_model = get_peft_model(mlp, config)

            lora_a = peft_model.base_model.model.lin1.lora_A["default"]
            assert isinstance(lora_a, BlockDiagonalLinear)
            assert lora_a.nblocks == nblocks

            x = torch.randn(2, 10)
            output = peft_model(x)
            assert output.shape == (2, 2)

    def test_bdlora_row_vs_column_modules(self, mlp):
        """Test that row and column modules create different layer types."""
        config = LoraConfig(
            r=8,
            target_modules=["lin1", "lin2", "lin3"],
            use_bdlora=BdLoraConfig(
                target_modules_bd_a=["lin1", "lin2"],
                target_modules_bd_b=["lin3"],
                nblocks=4,
            ),
        )
        peft_model = get_peft_model(mlp, config)

        lin1_lora_a = peft_model.base_model.model.lin1.lora_A["default"]
        lin2_lora_a = peft_model.base_model.model.lin2.lora_A["default"]
        assert isinstance(lin1_lora_a, BlockDiagonalLinear)
        assert isinstance(lin2_lora_a, BlockDiagonalLinear)

        lin3_lora_b = peft_model.base_model.model.lin3.lora_B["default"]
        assert isinstance(lin3_lora_b, BlockDiagonalLinear)

    def test_bdlora_unspecified_module_defaults_to_linear(self, mlp):
        """Test that modules not in row/column lists default to vanilla linear lora."""
        config = LoraConfig(
            r=8,
            target_modules=["lin1", "lin2"],
            use_bdlora=BdLoraConfig(
                target_modules_bd_a=["lin1"],
                nblocks=4,
            ),
        )
        peft_model = get_peft_model(mlp, config)

        assert isinstance(peft_model.base_model.model.lin2, Linear)
        # lin2 should have regular LoRA layers, not block diagonal
        assert not isinstance(peft_model.base_model.model.lin2.lora_A["default"], BlockDiagonalLinear)
        assert not isinstance(peft_model.base_model.model.lin2.lora_B["default"], BlockDiagonalLinear)

    def test_bdlora_trainable_parameters_count(self, mlp):
        """Test that BD-LoRA creates expected number of trainable parameters."""
        config = LoraConfig(
            r=8,
            target_modules=["lin1", "lin2"],
            use_bdlora=BdLoraConfig(
                target_modules_bd_a=["lin1"],
                target_modules_bd_b=["lin2"],
                nblocks=4,
            ),
        )
        peft_model = get_peft_model(mlp, config)

        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)

        # For BD-LoRA:
        # lin1: A is block-diagonal, B is regular
        #   - A: out_features * (in_features // nblocks) = 8 * (20 // 4) = 8 * 5 = 40
        #   - B: r * out_features = 8 * 20 = 160
        # lin2: A is regular, B is block-diagonal
        #   - A: r * in_features = 8 * 20 = 160
        #   - B: out_features * (r // nblocks) = 20 * (8 // 4) = 20 * 2 = 40
        # Total: 40 + 160 + 160 + 40 = 400
        expected_params = 400
        assert trainable_params == expected_params

    def test_bdlora_adapter_merging(self, mlp):
        """Test BD-LoRA adapter merging functionality."""
        config = LoraConfig(
            r=8,
            target_modules=["lin1", "lin2"],
            use_bdlora=BdLoraConfig(
                target_modules_bd_a=["lin1"],
                target_modules_bd_b=["lin2"],
                nblocks=4,
            ),
        )
        peft_model = get_peft_model(mlp, config)

        x = torch.randn(2, 10)
        output_before_merge = peft_model(x)

        # Merge adapters
        peft_model.merge_and_unload()

        # Test that merged model produces same output
        output_after_merge = peft_model(x)
        assert torch.allclose(output_before_merge, output_after_merge, atol=1e-5)

    def test_bdlora_adapter_unmerging(self, mlp):
        """Test BD-LoRA adapter unmerging functionality."""
        config = LoraConfig(
            r=8,
            target_modules=["lin1", "lin2"],
            use_bdlora=BdLoraConfig(
                target_modules_bd_a=["lin1"],
                target_modules_bd_b=["lin2"],
                nblocks=4,
            ),
        )
        peft_model = get_peft_model(mlp, config)

        x = torch.randn(2, 10)
        output_original = peft_model(x)

        # Merge then unmerge adapters
        peft_model.merge_adapter()
        output_merged = peft_model(x)

        peft_model.unmerge_adapter()
        output_unmerged = peft_model(x)

        # Test that unmerged model produces same output as original
        assert torch.allclose(output_original, output_unmerged, atol=1e-5)
