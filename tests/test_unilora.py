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

import os
import pytest
import torch
from safetensors import safe_open
from torch import nn

from peft import PeftModel, UniLoRAConfig, get_peft_model


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


class TestUniLoRA:
    def get_mlp(self):
        model = MLP()
        return model

    def test_unilora_parameters(self):
        mlp = self.get_mlp()

        # In the current implementation, `theta_d_length` effectively acts as the
        # size of the shared parameter pool (codebook size).
        # The values in `indices` are in the range [0, theta_d_length).
        theta_d_length = 100
        r = 4

        config = UniLoRAConfig(
            target_modules=["lin0", "lin1", "lin3"],
            theta_d_length=theta_d_length,
            r=r,
        )
        mlp_unilora = get_peft_model(mlp, config)

        theta_d = mlp_unilora.unilora_theta_d["default"]

        # 1. Check Theta_d (shared parameter pool)
        # According to `_init_unilora_theta_d`: torch.zeros(config.theta_d_length)
        assert theta_d.shape == (theta_d_length,)

        # 2. Check Indices and Scales (formerly Logits and Norm)
        # Indices should directly match the shape of the LoRA low-rank matrices

        # lin0: (10 -> 20)
        # indices_B: (out_features, r) -> (20, 4)
        unilora_lin0_indices_B = mlp_unilora.lin0.unilora_indices_B["default"]
        assert unilora_lin0_indices_B.shape == (mlp.lin0.out_features, config.r)

        # scales_B should have the same shape as indices_B
        unilora_lin0_scales_B = mlp_unilora.lin0.unilora_scales_B["default"]
        assert unilora_lin0_scales_B.shape == (mlp.lin0.out_features, config.r)

        # lin1: (20 -> 20)
        # indices_A: (r, in_features) -> (4, 20)
        unilora_lin1_indices_A = mlp_unilora.lin1.unilora_indices_A["default"]
        assert unilora_lin1_indices_A.shape == (config.r, mlp.lin1.in_features)

        # lin3: (20 -> 2)
        unilora_lin3_indices_A = mlp_unilora.lin3.unilora_indices_A["default"]
        assert unilora_lin3_indices_A.shape == (config.r, mlp.lin3.in_features)

        # 3. Check parameter sharing
        # Ensure that all layers reference the same underlying theta_d tensor
        assert (
            mlp_unilora.lin0.unilora_theta_d["default"].data_ptr()
            == mlp_unilora.lin3.unilora_theta_d["default"].data_ptr()
        )
        assert (
            mlp_unilora.lin1.unilora_theta_d["default"].data_ptr()
            == theta_d.data_ptr()
        )

        # 4. Forward pass test
        input = torch.randn(5, 10)
        output = mlp_unilora(input)
        assert output.shape == (5, 2)

    def test_save_load(self, tmp_path):
        """Test save/load consistency (no Top-K logic involved)."""
        torch.manual_seed(0)
        mlp = self.get_mlp()
        config = UniLoRAConfig(
            target_modules=["lin0", "lin1", "lin3"],
            theta_d_length=50,
            r=4,
        )
        mlp_unilora = get_peft_model(mlp, config)

        # Run a forward pass to ensure buffers are initialized and moved
        # to the correct device
        input = torch.randn(5, 10)
        output_before = mlp_unilora(input)

        save_path = tmp_path / "unilora"
        mlp_unilora.save_pretrained(save_path)
        assert os.path.exists(save_path / "adapter_config.json")
        assert os.path.exists(save_path / "adapter_model.safetensors")

        # Inspect safetensors contents
