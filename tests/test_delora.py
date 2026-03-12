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

# Tests specific to DeLoRA that go beyond the basic initialization tests in test_initialization.py.

import os

import pytest
import torch
from torch import nn

from peft import DeloraConfig, PeftModel, get_peft_model


class MLP(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.lin1 = nn.Linear(20, 2, bias=bias)

    def forward(self, X):
        X = self.lin0(X)
        X = self.lin1(X)
        return X


class TestDelora:
    @pytest.fixture
    def mlp(self):
        torch.manual_seed(0)
        return MLP()

    @pytest.fixture
    def data(self):
        torch.manual_seed(0)
        return torch.randn(4, 10)

    def test_merge_unmerge_roundtrip(self, mlp, data):
        config = DeloraConfig(target_modules=["lin0"], r=4, delora_lambda=15, init_weights=False)
        model = get_peft_model(mlp, config)
        model.eval()

        out_before = model(data)
        model.merge_adapter()
        model.unmerge_adapter()
        out_after = model(data)

        assert torch.allclose(out_before, out_after, atol=1e-5)

    def test_merge_produces_same_output(self, mlp, data):
        config = DeloraConfig(target_modules=["lin0"], r=4, delora_lambda=15, init_weights=False)
        model = get_peft_model(mlp, config)
        model.eval()

        out_unmerged = model(data)
        model.merge_adapter()
        out_merged = model(data)

        assert torch.allclose(out_unmerged, out_merged, atol=1e-5)

    def test_safe_merge_detects_nans(self, mlp, data):
        config = DeloraConfig(target_modules=["lin0"], r=4, delora_lambda=15)
        model = get_peft_model(mlp, config)

        # Corrupt weights to produce NaN in the delta
        layer = model.base_model.model.lin0
        layer.delora_B["default"].data.fill_(float("inf"))

        with pytest.raises(ValueError, match="NaNs detected"):
            model.merge_adapter(safe_merge=True)

    def test_disable_adapters_matches_base(self, mlp, data):
        base_out = mlp(data)
        config = DeloraConfig(target_modules=["lin0"], r=4, delora_lambda=15, init_weights=False)
        model = get_peft_model(mlp, config)

        with model.disable_adapter():
            disabled_out = model(data)

        assert torch.allclose(base_out, disabled_out, atol=1e-5)

    def test_save_load_roundtrip(self, mlp, data, tmp_path):
        config = DeloraConfig(target_modules=["lin0"], r=4, delora_lambda=15, init_weights=False)
        model = get_peft_model(mlp, config)
        model.eval()
        out_original = model(data)

        save_path = tmp_path / "delora"
        model.save_pretrained(save_path)
        assert os.path.exists(save_path / "adapter_config.json")
        assert os.path.exists(save_path / "adapter_model.safetensors")

        torch.manual_seed(0)
        base = MLP()
        loaded = PeftModel.from_pretrained(base, save_path)
        loaded.eval()
        out_loaded = loaded(data)

        assert torch.allclose(out_original, out_loaded, atol=1e-5)

    def test_multiple_adapters(self, mlp, data):
        torch.manual_seed(1)
        config1 = DeloraConfig(target_modules=["lin0"], r=4, delora_lambda=15, init_weights=False)
        model = get_peft_model(mlp, config1, adapter_name="first")

        torch.manual_seed(2)
        config2 = DeloraConfig(target_modules=["lin0"], r=4, delora_lambda=15, init_weights=False)
        model.add_adapter("second", config2)

        model.set_adapter("first")
        out_first = model(data)

        model.set_adapter("second")
        out_second = model(data)

        # Different random init should produce different outputs
        assert not torch.allclose(out_first, out_second, atol=1e-3)

    def test_module_dropout_deterministic_in_eval(self, mlp, data):
        config = DeloraConfig(target_modules=["lin0"], r=4, delora_lambda=15, module_dropout=0.5, init_weights=False)
        model = get_peft_model(mlp, config)
        model.eval()

        out1 = model(data)
        out2 = model(data)
        assert torch.allclose(out1, out2)

    def test_rank_pattern(self, mlp):
        config = DeloraConfig(
            target_modules=["lin0", "lin1"],
            r=4,
            delora_lambda=15,
            rank_pattern={"lin1": 8},
        )
        model = get_peft_model(mlp, config)

        lin0 = model.base_model.model.lin0
        lin1 = model.base_model.model.lin1
        assert lin0.r["default"] == 4
        assert lin1.r["default"] == 8

    def test_lambda_pattern(self, mlp):
        config = DeloraConfig(
            target_modules=["lin0", "lin1"],
            r=4,
            delora_lambda=15,
            lambda_pattern={"lin1": 30},
        )
        model = get_peft_model(mlp, config)

        lin0 = model.base_model.model.lin0
        lin1 = model.base_model.model.lin1
        assert lin0.delora_lambda["default"].item() == pytest.approx(15.0)
        assert lin1.delora_lambda["default"].item() == pytest.approx(30.0)

    def test_config_r_zero_raises(self):
        with pytest.raises(ValueError, match="`r` must be a positive integer"):
            DeloraConfig(target_modules=["lin0"], r=0)

    def test_config_r_negative_raises(self):
        with pytest.raises(ValueError, match="`r` must be a positive integer"):
            DeloraConfig(target_modules=["lin0"], r=-1)

    def test_lora_conversion_supported(self, mlp):
        config = DeloraConfig(target_modules=["lin0"], r=4, delora_lambda=15)
        model = get_peft_model(mlp, config)
        layer = model.base_model.model.lin0
        assert layer.supports_lora_conversion()
