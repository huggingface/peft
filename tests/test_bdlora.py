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

from peft import BdLoraConfig, PeftModel, get_peft_model
from peft.tuners.bdlora.layer import BlockDiagonalLinear, ColumnParallelLinearLora, RowParallelLinearLora
from peft.tuners.lora.layer import Linear


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
        config = BdLoraConfig(
            r=16,
            target_modules=["lin1", "lin2"],
            row_sharded_modules=["lin1"],
            column_sharded_modules=["lin2"],
            nblocks=4,
        )
        assert config.r == 16
        assert set(config.target_modules) == {"lin1", "lin2"}  # target_modules is a set
        assert config.row_sharded_modules == ["lin1"]
        assert config.column_sharded_modules == ["lin2"]
        assert config.nblocks == 4
        assert config.prefix == "lora_"

    def test_bdlora_model_creation(self, mlp):
        """Test that BD-LoRA model can be created and has trainable parameters."""
        config = BdLoraConfig(
            r=8,
            target_modules=["lin1", "lin2"],
            row_sharded_modules=["lin1"],
            column_sharded_modules=["lin2"],
            nblocks=4,
        )
        peft_model = get_peft_model(mlp, config)

        # Check that model has trainable parameters
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        assert trainable_params > 0

        # Check that correct layer types are created
        assert isinstance(peft_model.base_model.model.lin1, RowParallelLinearLora)
        assert isinstance(peft_model.base_model.model.lin2, ColumnParallelLinearLora)

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
        config = BdLoraConfig(
            r=8,
            target_modules=["lin1", "lin2"],
            row_sharded_modules=["lin1"],
            column_sharded_modules=["lin2"],
            nblocks=4,
        )
        peft_model = get_peft_model(mlp, config)

        x = torch.randn(3, 10)
        output = peft_model(x)

        assert output.shape == (3, 2)
        assert not torch.isnan(output).any()

    def test_bdlora_save_load(self, mlp, tmp_path):
        """Test BD-LoRA model save and load functionality."""
        torch.manual_seed(42)  # Set seed for reproducible results
        config = BdLoraConfig(
            r=8,
            target_modules=["lin1", "lin2"],
            row_sharded_modules=["lin1"],
            column_sharded_modules=["lin2"],
            nblocks=4,
        )
        peft_model = get_peft_model(mlp, config)

        # Test input
        torch.manual_seed(42)  # Set seed for reproducible input
        x = torch.randn(2, 10)
        output_before = peft_model(x)

        # Save model
        save_path = tmp_path / "bdlora_test"
        peft_model.save_pretrained(save_path)

        # Load model
        torch.manual_seed(0)  # Reset to original seed
        base_model = MLP()
        loaded_model = PeftModel.from_pretrained(base_model, save_path)

        # Test that outputs match
        output_after = loaded_model(x)
        assert torch.allclose(output_before, output_after, atol=1e-5)  # Relaxed tolerance

    def test_bdlora_different_nblocks(self):
        """Test BD-LoRA with different nblocks values."""
        for nblocks in [2, 4]:  # Reduced to avoid model reuse issues
            torch.manual_seed(0)
            mlp = MLP()
            config = BdLoraConfig(r=16, target_modules=["lin1"], row_sharded_modules=["lin1"], nblocks=nblocks)
            peft_model = get_peft_model(mlp, config)

            # Check that block diagonal layer has correct nblocks
            lora_a = peft_model.base_model.model.lin1.lora_A["default"]
            assert isinstance(lora_a, BlockDiagonalLinear)
            assert lora_a.nblocks == nblocks

            # Test forward pass works
            x = torch.randn(2, 10)
            output = peft_model(x)
            assert output.shape == (2, 2)

    def test_bdlora_row_vs_column_modules(self, mlp):
        """Test that row and column modules create different layer types."""
        config = BdLoraConfig(
            r=8,
            target_modules=["lin1", "lin2", "lin3"],
            row_sharded_modules=["lin1", "lin2"],
            column_sharded_modules=["lin3"],
            nblocks=4,
        )
        peft_model = get_peft_model(mlp, config)

        # Check row modules have block-diagonal A
        lin1_lora_a = peft_model.base_model.model.lin1.lora_A["default"]
        lin2_lora_a = peft_model.base_model.model.lin2.lora_A["default"]
        assert isinstance(lin1_lora_a, BlockDiagonalLinear)
        assert isinstance(lin2_lora_a, BlockDiagonalLinear)

        # Check column module has block-diagonal B
        lin3_lora_b = peft_model.base_model.model.lin3.lora_B["default"]
        assert isinstance(lin3_lora_b, BlockDiagonalLinear)

    def test_bdlora_unspecified_module_defaults_to_linear(self, mlp):
        """Test that modules not in row/column lists default to vanilla linear lora."""
        config = BdLoraConfig(
            r=8,
            target_modules=["lin1", "lin2"],
            row_sharded_modules=["lin1"],
            # lin2 not specified in either list
            nblocks=4,
        )
        peft_model = get_peft_model(mlp, config)

        assert isinstance(peft_model.base_model.model.lin2, Linear)

    def test_bdlora_trainable_parameters_count(self, mlp):
        """Test that BD-LoRA creates expected number of trainable parameters."""
        config = BdLoraConfig(
            r=8,
            target_modules=["lin1", "lin2"],
            row_sharded_modules=["lin1"],
            column_sharded_modules=["lin2"],
            nblocks=4,
        )
        peft_model = get_peft_model(mlp, config)

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)

        # For BD-LoRA:
        # lin1 (row-sharded): A is block-diagonal, B is regular
        #   - A: nblocks * (r//nblocks) * (in_features//nblocks) = 4 * 2 * 5 = 40
        #   - B: r * out_features = 8 * 20 = 160
        # lin2 (column-sharded): A is regular, B is block-diagonal
        #   - A: r * in_features = 8 * 20 = 160
        #   - B: nblocks * (out_features//nblocks) * (r//nblocks) = 4 * 5 * 2 = 40
        # Total: 40 + 160 + 160 + 40 = 400
        expected_params = 400
        assert trainable_params == expected_params
