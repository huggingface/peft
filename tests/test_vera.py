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

# This test file is for tests specific to VeRA, since VeRA has some specific challenges due to the shared weights.

import os

import pytest
import torch
from torch import nn

from peft import PeftModel, VeraConfig, get_peft_model


class MLP(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.lin1 = nn.Linear(20, 20, bias=bias)  # lin1 and lin2 have same shape
        self.lin2 = nn.Linear(20, 20, bias=bias)
        self.lin3 = nn.Linear(20, 2, bias=bias)
        self.sm = nn.LogSoftmax(dim=-1)

    def forward(self, X):
        X = X.float()
        X = self.lin0(X)
        X = self.relu(X)
        X = self.lin1(X)
        X = self.relu(X)
        X = self.lin2(X)
        X = self.relu(X)
        X = self.lin3(X)
        X = self.sm(X)


class TestVera:
    @pytest.fixture
    def mlp(self):
        torch.manual_seed(0)
        model = MLP()
        return model

    @pytest.fixture
    def mlp_same_prng(self, mlp):
        torch.manual_seed(0)

        config = VeraConfig(target_modules=["lin1", "lin2"], init_weights=False, projection_prng_key=42)
        # creates a default VeRA adapter
        peft_model = get_peft_model(mlp, config)
        config2 = VeraConfig(target_modules=["lin1", "lin2"], init_weights=False, projection_prng_key=42)
        peft_model.add_adapter("other", config2)
        return peft_model

    def test_multiple_adapters_same_prng_weights(self, mlp_same_prng):
        # we can have multiple adapters with the same prng key, in which case the weights should be shared
        assert (
            mlp_same_prng.base_model.model.lin1.vera_A["default"]
            is mlp_same_prng.base_model.model.lin1.vera_A["other"]
        )
        assert (
            mlp_same_prng.base_model.model.lin1.vera_B["default"]
            is mlp_same_prng.base_model.model.lin1.vera_B["other"]
        )
        assert (
            mlp_same_prng.base_model.model.lin2.vera_A["default"]
            is mlp_same_prng.base_model.model.lin2.vera_A["other"]
        )
        assert (
            mlp_same_prng.base_model.model.lin2.vera_B["default"]
            is mlp_same_prng.base_model.model.lin2.vera_B["other"]
        )

        input = torch.randn(1, 10)
        mlp_same_prng.set_adapter("default")
        output_default = mlp_same_prng(input)
        mlp_same_prng.set_adapter("other")
        output_other = mlp_same_prng(input)
        assert not torch.allclose(output_default, output_other, atol=1e-3, rtol=1e-3)

    def test_multiple_adapters_different_prng_raises(self):
        # we cannot have multiple adapters with different prng keys
        model = MLP()
        config = VeraConfig(target_modules=["lin1", "lin2"], init_weights=False, projection_prng_key=42)
        # creates a default VeRA adapter
        peft_model = get_peft_model(model, config)
        config2 = VeraConfig(target_modules=["lin1", "lin2"], init_weights=False, projection_prng_key=123)

        msg = r"Vera PRNG initialisation key must be the same for all adapters. Got config.projection_prng_key=123 but previous config had 42"
        with pytest.raises(ValueError, match=msg):
            peft_model.add_adapter("other", config2)

    def test_multiple_adapters_save_load(self, mlp_same_prng, tmp_path):
        # check saving and loading works with multiple adapters
        torch.manual_seed(0)
        input = torch.randn(1, 10)
        mlp_same_prng.set_adapter("default")
        output_default = mlp_same_prng(input)
        mlp_same_prng.set_adapter("other")
        output_other = mlp_same_prng(input)

        save_path = tmp_path / "vera"
        mlp_same_prng.save_pretrained(save_path)
        assert os.path.exists(save_path / "adapter-config.json")
        assert os.path.exists(save_path / "other" / "adapter-config.json")

        torch.manual_seed(0)
        mlp = MLP()
        peft_model = PeftModel.from_pretrained(mlp, save_path)
        peft_model.set_adapter("default")
        output_default_loaded = peft_model(input)
        peft_model.set_adapter("other")
        output_other_loaded = peft_model(input)

        assert torch.allclose(output_default, output_default_loaded, atol=1e-3, rtol=1e-3)
        assert torch.allclose(output_other, output_other_loaded, atol=1e-3, rtol=1e-3)
