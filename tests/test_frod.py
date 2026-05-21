# Copyright 2026-present the HuggingFace Inc. team.
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

# This test file is for tests specific to FRoD, since FRoD has shared projection buffers.

import os

import pytest
import torch
from accelerate.utils.imports import is_bf16_available
from safetensors import safe_open
from torch import nn

from peft import FRODConfig, PeftModel, get_peft_model


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
        X = self.lin0(X)
        X = self.relu(X)
        X = self.lin1(X)
        X = self.relu(X)
        X = self.lin2(X)
        X = self.relu(X)
        X = self.lin3(X)
        X = self.sm(X)
        return X


class TestFROD:
    @pytest.fixture
    def mlp(self):
        torch.manual_seed(0)
        model = MLP()
        return model

    @pytest.fixture
    def mlp_same_prng(self, mlp):
        torch.manual_seed(0)

        config = FRODConfig(target_modules=["lin1", "lin2"], init_weights=False)
        peft_model = get_peft_model(mlp, config)
        config2 = FRODConfig(target_modules=["lin1", "lin2"], init_weights=False)
        peft_model.add_adapter("other", config2)
        return peft_model

    @staticmethod
    def _make_second_adapter_different(peft_model):
        with torch.no_grad():
            for module in peft_model.base_model.model.modules():
                if hasattr(module, "frod_lambda_l") and "second" in module.frod_lambda_l:
                    module.frod_lambda_l["second"].add_(0.1)

    def test_multiple_adapters_same_prng_projection_buffers(self, mlp_same_prng):
        # Multiple adapters with the same PRNG key share fixed projection buffers within each FRoD layer.
        assert (
            mlp_same_prng.base_model.model.lin1.frod_V["default"].data_ptr()
            == mlp_same_prng.base_model.model.lin1.frod_V["other"].data_ptr()
        )
        assert (
            mlp_same_prng.base_model.model.lin1.frod_s_indices["default"].data_ptr()
            == mlp_same_prng.base_model.model.lin1.frod_s_indices["other"].data_ptr()
        )
        assert (
            mlp_same_prng.base_model.model.lin2.frod_V["default"].data_ptr()
            == mlp_same_prng.base_model.model.lin2.frod_V["other"].data_ptr()
        )
        assert (
            mlp_same_prng.base_model.model.lin2.frod_s_indices["default"].data_ptr()
            == mlp_same_prng.base_model.model.lin2.frod_s_indices["other"].data_ptr()
        )

    def test_multiple_adapters_different_prng_raises(self):
        model = MLP()
        config = FRODConfig(target_modules=["lin1", "lin2"], init_weights=False)
        peft_model = get_peft_model(model, config)
        config2 = FRODConfig(target_modules=["lin1", "lin2"], init_weights=False, projection_prng_key=123)

        msg = (
            r"FRoD projection initialization key must be the same for all adapters. Got "
            r"config.projection_prng_key=123 but previous config had 0"
        )
        with pytest.raises(ValueError, match=msg):
            peft_model.add_adapter("other", config2)

    def test_multiple_adapters_save_load_save_projection_false(self, mlp, tmp_path):
        # Check saving and loading works with multiple adapters without saved projection tensors.
        torch.manual_seed(1)
        config = FRODConfig(target_modules=["lin1", "lin2"], init_weights=False, save_projection=False)
        peft_model = get_peft_model(mlp, config, adapter_name="first")
        config2 = FRODConfig(target_modules=["lin1", "lin2"], init_weights=False, save_projection=False)
        peft_model.add_adapter("second", config2)
        self._make_second_adapter_different(peft_model)
        peft_model.eval()

        input = torch.randn(5, 10)
        peft_model.set_adapter("first")
        output_first = peft_model(input)
        peft_model.set_adapter("second")
        output_second = peft_model(input)

        assert not torch.allclose(output_first, output_second, atol=1e-3, rtol=1e-3)

        save_path = tmp_path / "frod"
        peft_model.save_pretrained(save_path)
        assert os.path.exists(save_path / "first" / "adapter_config.json")
        assert os.path.exists(save_path / "second" / "adapter_config.json")

        torch.manual_seed(0)
        mlp = MLP()
        peft_model = PeftModel.from_pretrained(mlp, save_path / "first", adapter_name="first")
        peft_model.load_adapter(save_path / "second", "second")
        peft_model.eval()

        peft_model.set_adapter("first")
        output_first_loaded = peft_model(input)
        peft_model.set_adapter("second")
        output_second_loaded = peft_model(input)

        assert torch.allclose(output_first, output_first_loaded, atol=1e-3, rtol=1e-3)
        assert torch.allclose(output_second, output_second_loaded, atol=1e-3, rtol=1e-3)

    def test_save_projection_false_contains_no_frod_projection_tensors(self, mlp, tmp_path):
        config = FRODConfig(target_modules=["lin1", "lin2"], init_weights=False, save_projection=False)
        peft_model = get_peft_model(mlp, config)

        save_path = tmp_path / "frod"
        peft_model.save_pretrained(save_path)

        state_dict = {}
        with safe_open(save_path / "adapter_model.safetensors", framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

        assert not any("frod_V" in key for key in state_dict)
        assert not any("frod_s_indices" in key for key in state_dict)
        assert not any("frod_s_size" in key for key in state_dict)
        assert not any("frod_U" in key for key in state_dict)

    def test_save_projection_true_contains_top_level_projection_tensors_only(self, mlp, tmp_path):
        config = FRODConfig(target_modules=["lin1", "lin2"], init_weights=False)
        peft_model = get_peft_model(mlp, config)

        save_path = tmp_path / "frod"
        peft_model.save_pretrained(save_path)

        keys = []
        with safe_open(save_path / "adapter_model.safetensors", framework="pt", device="cpu") as f:
            keys = list(f.keys())

        assert "base_model.frod_V.lin1" in keys
        assert "base_model.frod_s_indices.lin1" in keys
        assert "base_model.frod_s_size.lin1" in keys
        assert "base_model.frod_V.lin2" in keys
        assert not any(".model.lin1.frod_V" in key for key in keys)
        assert not any("frod_U" in key for key in keys)

    def test_frod_projection_buffers_share_memory_with_layers(self, mlp_same_prng):
        frod_V_lin1 = mlp_same_prng.base_model.frod_V["lin1"]["default"]
        frod_s_indices_lin1 = mlp_same_prng.base_model.frod_s_indices["lin1"]["default"]

        assert frod_V_lin1.data_ptr() == mlp_same_prng.base_model.model.lin1.frod_V["default"].data_ptr()
        assert frod_V_lin1.data_ptr() == mlp_same_prng.base_model.model.lin1.frod_V["other"].data_ptr()
        assert (
            frod_s_indices_lin1.data_ptr() == mlp_same_prng.base_model.model.lin1.frod_s_indices["default"].data_ptr()
        )
        assert frod_s_indices_lin1.data_ptr() == mlp_same_prng.base_model.model.lin1.frod_s_indices["other"].data_ptr()

        # Different target categories have distinct projection buffers.
        assert frod_V_lin1.data_ptr() != mlp_same_prng.base_model.frod_V["lin2"]["default"].data_ptr()

    def test_frod_lambda_dont_share_memory(self, mlp_same_prng):
        assert (
            mlp_same_prng.base_model.model.lin1.frod_lambda_s_values["default"].data_ptr()
            != mlp_same_prng.base_model.model.lin1.frod_lambda_s_values["other"].data_ptr()
        )
        assert (
            mlp_same_prng.base_model.model.lin1.frod_lambda_s_values["default"].data_ptr()
            != mlp_same_prng.base_model.model.lin2.frod_lambda_s_values["default"].data_ptr()
        )
        assert (
            mlp_same_prng.base_model.model.lin1.frod_lambda_l["default"].data_ptr()
            != mlp_same_prng.base_model.model.lin1.frod_lambda_l["other"].data_ptr()
        )
        assert (
            mlp_same_prng.base_model.model.lin1.frod_lambda_l["default"].data_ptr()
            != mlp_same_prng.base_model.model.lin2.frod_lambda_l["default"].data_ptr()
        )

    def test_frod_different_shapes(self, mlp):
        config = FRODConfig(target_modules=["lin0", "lin3"], init_weights=False)
        mlp_different_shapes = get_peft_model(mlp, config)

        assert mlp.lin0.base_layer.weight.shape != mlp.lin3.base_layer.weight.shape
        assert mlp_different_shapes.base_model.frod_V["lin0"]["default"].shape == (
            mlp.lin0.in_features,
            mlp.lin0.in_features,
        )
        assert mlp_different_shapes.base_model.frod_V["lin3"]["default"].shape == (
            mlp.lin3.in_features,
            mlp.lin3.in_features,
        )

        input = torch.randn(5, 10)
        mlp_different_shapes(input)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_frod_dtypes(self, dtype):
        if dtype == torch.bfloat16:
            if not is_bf16_available():
                pytest.skip("bfloat16 not supported on this system, skipping the test")

        model = MLP().to(dtype)
        config = FRODConfig(target_modules=["lin1", "lin2"], init_weights=False)
        peft_model = get_peft_model(model, config)
        inputs = torch.randn(5, 10).to(dtype)
        output = peft_model(inputs)
        assert output.dtype == dtype
