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
from safetensors import safe_open
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
        X = self.lin0(X)
        X = self.relu(X)
        X = self.lin1(X)
        X = self.relu(X)
        X = self.lin2(X)
        X = self.relu(X)
        X = self.lin3(X)
        X = self.sm(X)
        return X


class TestVera:
    @pytest.fixture
    def mlp(self):
        torch.manual_seed(0)
        model = MLP()
        return model

    @pytest.fixture
    def mlp_same_prng(self, mlp):
        torch.manual_seed(0)

        config = VeraConfig(target_modules=["lin1", "lin2"], init_weights=False)
        # creates a default VeRA adapter
        peft_model = get_peft_model(mlp, config)
        config2 = VeraConfig(target_modules=["lin1", "lin2"], init_weights=False)
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

        input = torch.randn(5, 10)
        mlp_same_prng.set_adapter("default")
        output_default = mlp_same_prng(input)
        mlp_same_prng.set_adapter("other")
        output_other = mlp_same_prng(input)
        assert not torch.allclose(output_default, output_other, atol=1e-3, rtol=1e-3)

    def test_multiple_adapters_different_prng_raises(self):
        # we cannot have multiple adapters with different prng keys
        model = MLP()
        config = VeraConfig(target_modules=["lin1", "lin2"], init_weights=False)
        # creates a default VeRA adapter
        peft_model = get_peft_model(model, config)
        config2 = VeraConfig(target_modules=["lin1", "lin2"], init_weights=False, projection_prng_key=123)

        msg = (
            r"Vera PRNG initialisation key must be the same for all adapters. Got config.projection_prng_key=123 but "
            r"previous config had 0"
        )
        with pytest.raises(ValueError, match=msg):
            peft_model.add_adapter("other", config2)

    def test_multiple_adapters_save_load_save_projection_true(self, mlp_same_prng, tmp_path):
        # check saving and loading works with multiple adapters and saved projection weights
        torch.manual_seed(0)
        input = torch.randn(5, 10)
        mlp_same_prng.set_adapter("default")
        output_default = mlp_same_prng(input)
        mlp_same_prng.set_adapter("other")
        output_other = mlp_same_prng(input)

        # sanity check
        assert not torch.allclose(output_default, output_other, atol=1e-3, rtol=1e-3)

        save_path = tmp_path / "vera"
        mlp_same_prng.save_pretrained(save_path)
        assert os.path.exists(save_path / "adapter_config.json")
        assert os.path.exists(save_path / "other" / "adapter_config.json")

        torch.manual_seed(0)
        mlp = MLP()
        peft_model = PeftModel.from_pretrained(mlp, save_path)
        peft_model.load_adapter(save_path / "other", "other")

        peft_model.set_adapter("default")
        output_default_loaded = peft_model(input)
        peft_model.set_adapter("other")
        output_other_loaded = peft_model(input)

        assert torch.allclose(output_default, output_default_loaded, atol=1e-3, rtol=1e-3)
        assert torch.allclose(output_other, output_other_loaded, atol=1e-3, rtol=1e-3)

    def test_multiple_adapters_save_load_save_projection_false(self, mlp, tmp_path):
        # check saving and loading works with multiple adapters without saved projection weights
        torch.manual_seed(1)
        config = VeraConfig(target_modules=["lin1", "lin2"], init_weights=False, save_projection=False)
        # creates a default VeRA adapter
        peft_model = get_peft_model(mlp, config, adapter_name="first")
        config2 = VeraConfig(target_modules=["lin1", "lin2"], init_weights=False, save_projection=False)
        peft_model.add_adapter("second", config2)

        input = torch.randn(5, 10)
        peft_model.set_adapter("first")
        output_first = peft_model(input)
        peft_model.set_adapter("second")
        output_second = peft_model(input)

        # sanity check
        assert not torch.allclose(output_first, output_second, atol=1e-3, rtol=1e-3)

        save_path = tmp_path / "vera"
        peft_model.save_pretrained(save_path)
        assert os.path.exists(save_path / "first" / "adapter_config.json")
        assert os.path.exists(save_path / "second" / "adapter_config.json")

        torch.manual_seed(0)
        mlp = MLP()
        peft_model = PeftModel.from_pretrained(mlp, save_path / "first", adapter_name="first")
        peft_model.load_adapter(save_path / "second", "second")

        peft_model.set_adapter("first")
        output_first_loaded = peft_model(input)
        peft_model.set_adapter("second")
        output_second_loaded = peft_model(input)

        assert torch.allclose(output_first, output_first_loaded, atol=1e-3, rtol=1e-3)
        assert torch.allclose(output_second, output_second_loaded, atol=1e-3, rtol=1e-3)

    def test_multiple_adapters_save_projection_true_contains_vera_A_vera_B(self, mlp_same_prng, tmp_path):
        # check that the state_dicts don't contain the projection weights
        save_path = tmp_path / "vera"
        mlp_same_prng.save_pretrained(save_path)

        sd_default = {}
        with safe_open(save_path / "adapter_model.safetensors", framework="pt", device="cpu") as f:
            for key in f.keys():
                sd_default[key] = f.get_tensor(key)

        assert any("vera_A" in key for key in sd_default)
        assert any("vera_B" in key for key in sd_default)
        # default rank for VeRA is 256
        assert sd_default["base_model.vera_A"].shape == (256, 20)
        assert sd_default["base_model.vera_B"].shape == (20, 256)

        sd_other = {}
        with safe_open(save_path / "other" / "adapter_model.safetensors", framework="pt", device="cpu") as f:
            for key in f.keys():
                sd_other[key] = f.get_tensor(key)

        assert any("vera_A" in key for key in sd_other)
        assert any("vera_B" in key for key in sd_other)
        assert sd_other["base_model.vera_A"].shape == (256, 20)
        assert sd_other["base_model.vera_B"].shape == (20, 256)

    def test_multiple_adapters_save_projection_false_contains_no_vera_A_vera_B(self, mlp, tmp_path):
        torch.manual_seed(1)
        config = VeraConfig(target_modules=["lin1", "lin2"], init_weights=False, save_projection=False)
        # creates a default VeRA adapter
        peft_model = get_peft_model(mlp, config, adapter_name="first")
        config2 = VeraConfig(target_modules=["lin1", "lin2"], init_weights=False, save_projection=False)
        peft_model.add_adapter("second", config2)

        save_path = tmp_path / "vera"
        peft_model.save_pretrained(save_path)

        sd_default = {}
        with safe_open(save_path / "first" / "adapter_model.safetensors", framework="pt", device="cpu") as f:
            for key in f.keys():
                sd_default[key] = f.get_tensor(key)

        assert not any("vera_A" in key for key in sd_default)
        assert not any("vera_B" in key for key in sd_default)

        sd_other = {}
        with safe_open(save_path / "second" / "adapter_model.safetensors", framework="pt", device="cpu") as f:
            for key in f.keys():
                sd_other[key] = f.get_tensor(key)

        assert not any("vera_A" in key for key in sd_other)
        assert not any("vera_B" in key for key in sd_other)

    def test_vera_A_vera_B_share_memory(self, mlp_same_prng):
        vera_A = mlp_same_prng.vera_A["default"]
        vera_B = mlp_same_prng.vera_B["default"]

        # these tensors should share the same data
        assert vera_A.data_ptr() == mlp_same_prng.base_model.model.lin1.vera_A["default"].data_ptr()
        assert vera_B.data_ptr() == mlp_same_prng.base_model.model.lin1.vera_B["default"].data_ptr()
        assert vera_A.data_ptr() == mlp_same_prng.base_model.model.lin2.vera_A["default"].data_ptr()
        assert vera_B.data_ptr() == mlp_same_prng.base_model.model.lin2.vera_B["default"].data_ptr()
        # sanity check: these tensors shouldn't share the same data
        assert vera_A.data_ptr() != vera_B.data_ptr()

    def test_vera_lambda_dont_share_memory(self, mlp_same_prng):
        # sanity check: these tensors shouldn't share the same data
        assert (
            mlp_same_prng.base_model.model.lin1.vera_lambda_b["default"].data_ptr()
            != mlp_same_prng.base_model.model.lin1.vera_lambda_b["other"].data_ptr()
        )
        assert (
            mlp_same_prng.base_model.model.lin1.vera_lambda_b["default"].data_ptr()
            != mlp_same_prng.base_model.model.lin2.vera_lambda_b["default"].data_ptr()
        )
        assert (
            mlp_same_prng.base_model.model.lin1.vera_lambda_b["other"].data_ptr()
            != mlp_same_prng.base_model.model.lin2.vera_lambda_b["other"].data_ptr()
        )
        assert (
            mlp_same_prng.base_model.model.lin1.vera_lambda_d["default"].data_ptr()
            != mlp_same_prng.base_model.model.lin1.vera_lambda_d["other"].data_ptr()
        )
        assert (
            mlp_same_prng.base_model.model.lin1.vera_lambda_d["default"].data_ptr()
            != mlp_same_prng.base_model.model.lin2.vera_lambda_d["default"].data_ptr()
        )
        assert (
            mlp_same_prng.base_model.model.lin1.vera_lambda_d["other"].data_ptr()
            != mlp_same_prng.base_model.model.lin2.vera_lambda_d["other"].data_ptr()
        )

    def test_vera_different_shapes(self, mlp):
        config = VeraConfig(target_modules=["lin0", "lin3"], init_weights=False)
        mlp_different_shapes = get_peft_model(mlp, config)

        vera_A = mlp_different_shapes.vera_A["default"]
        vera_B = mlp_different_shapes.vera_B["default"]

        # sanity check
        assert mlp.lin0.base_layer.weight.shape != mlp.lin3.base_layer.weight.shape

        # lin0 has the largest output dimension, lin3 has the largest input dimension
        # vera_A should have the shape of (rank, largest_in), vera_B should have the shape of (largest_out, rank)
        assert vera_A.shape == (config.r, mlp.lin3.in_features)
        assert vera_B.shape == (mlp.lin0.out_features, config.r)

        # should not raise
        input = torch.randn(5, 10)
        mlp_different_shapes(input)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_vera_dtypes(self, dtype):
        # 1872
        if (dtype == torch.bfloat16) and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            pytest.skip("bfloat16 not supported on this system, skipping the test")

        model = MLP().to(dtype)
        config = VeraConfig(target_modules=["lin1", "lin2"], init_weights=False)
        peft_model = get_peft_model(model, config)
        inputs = torch.randn(5, 10).to(dtype)
        output = peft_model(inputs)  # should not raise
        assert output.dtype == dtype
