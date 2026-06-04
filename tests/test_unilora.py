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

import pytest
import torch
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from torch import nn

from peft import PeftModel, UniLoraConfig, get_peft_model


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


def _unilora_config_accepts(name):
    return name in UniLoraConfig.__dataclass_fields__


def _make_unilora_config(**kwargs):
    if "theta_d_length" not in kwargs:
        kwargs["theta_d_length"] = 101
    if "r" not in kwargs:
        kwargs["r"] = 4
    return UniLoraConfig(**kwargs)


def _get_unilora_index_state(model):
    return {
        name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items() if "unilora_indices" in name
    }


class TestUniLora:
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

        config = _make_unilora_config(
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
        assert mlp_unilora.lin1.unilora_theta_d["default"].data_ptr() == theta_d.data_ptr()

        # 4. Forward pass test
        input = torch.randn(5, 10)
        output = mlp_unilora(input)
        assert output.shape == (5, 2)

    def test_save_load(self, tmp_path):
        torch.manual_seed(0)
        mlp = self.get_mlp()
        config = _make_unilora_config(
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
        assert (save_path / "adapter_config.json").exists()
        assert (save_path / "adapter_model.safetensors").exists()

        saved_tensors = safe_load_file(save_path / "adapter_model.safetensors")
        assert "base_model.unilora_theta_d" in saved_tensors

        torch.manual_seed(0)
        loaded = PeftModel.from_pretrained(self.get_mlp(), save_path)
        output_after = loaded(input)

        torch.testing.assert_close(output_before, output_after)

    def test_proj_seed_deterministically_generates_indices(self):
        config_kwargs = {
            "target_modules": ["lin0", "lin1", "lin3"],
            "theta_d_length": 53,
            "r": 4,
        }
        model0 = get_peft_model(self.get_mlp(), _make_unilora_config(**config_kwargs, proj_seed=17))
        model1 = get_peft_model(self.get_mlp(), _make_unilora_config(**config_kwargs, proj_seed=17))
        model2 = get_peft_model(self.get_mlp(), _make_unilora_config(**config_kwargs, proj_seed=18))

        indices0 = _get_unilora_index_state(model0)
        indices1 = _get_unilora_index_state(model1)
        indices2 = _get_unilora_index_state(model2)

        assert indices0
        assert indices0.keys() == indices1.keys() == indices2.keys()
        for key in indices0:
            torch.testing.assert_close(indices0[key], indices1[key])

        assert any(not torch.equal(indices0[key], indices2[key]) for key in indices0)

    def test_init_weights_false_changes_theta_d_initialization(self):
        if not _unilora_config_accepts("init_weights"):
            pytest.skip("UniLoraConfig does not expose init_weights yet.")

        config_kwargs = {
            "target_modules": ["lin0", "lin1"],
            "theta_d_length": 47,
            "r": 4,
        }

        torch.manual_seed(0)
        default_model = get_peft_model(self.get_mlp(), _make_unilora_config(**config_kwargs))
        torch.manual_seed(0)
        random_model = get_peft_model(self.get_mlp(), _make_unilora_config(**config_kwargs, init_weights=False))

        assert not torch.allclose(
            default_model.unilora_theta_d["default"],
            random_model.unilora_theta_d["default"],
        )

    def test_saved_indices_round_trip_when_enabled(self, tmp_path):
        if not _unilora_config_accepts("save_indices"):
            pytest.skip("UniLoraConfig does not expose save_indices yet.")

        config = _make_unilora_config(
            target_modules=["lin0", "lin1", "lin3"],
            theta_d_length=59,
            r=4,
            save_indices=True,
        )
        model = get_peft_model(self.get_mlp(), config)
        original_indices = _get_unilora_index_state(model)

        save_path = tmp_path / "unilora-with-indices"
        model.save_pretrained(save_path)

        saved_tensors = safe_load_file(save_path / "adapter_model.safetensors")
        saved_index_tensors = {name: tensor for name, tensor in saved_tensors.items() if "unilora_indices" in name}
        assert saved_index_tensors

        loaded = PeftModel.from_pretrained(self.get_mlp(), save_path)
        loaded_indices = _get_unilora_index_state(loaded)
        assert original_indices.keys() == loaded_indices.keys()
        for key in original_indices:
            torch.testing.assert_close(original_indices[key], loaded_indices[key])

    def test_missing_theta_d_key_warns_on_load(self, tmp_path):
        model = get_peft_model(
            self.get_mlp(), _make_unilora_config(target_modules=["lin0", "lin1"], save_indices=True)
        )
        save_path = tmp_path / "unilora"
        model.save_pretrained(save_path)

        tensors = safe_load_file(save_path / "adapter_model.safetensors")
        tensors.pop("base_model.unilora_theta_d")
        safe_save_file(tensors, save_path / "adapter_model.safetensors")

        with pytest.warns(UserWarning, match="unilora_theta_d"):
            PeftModel.from_pretrained(self.get_mlp(), save_path)
