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
from transformers import LlamaConfig, LlamaForCausalLM

from peft import FrodConfig, PeftModel, get_peft_model


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


class TestFrod:
    @pytest.fixture
    def mlp(self):
        torch.manual_seed(0)
        model = MLP()
        return model

    @pytest.fixture
    def mlp_same_prng(self, mlp):
        torch.manual_seed(0)

        config = FrodConfig(target_modules=["lin1", "lin2"], init_weights=False)
        peft_model = get_peft_model(mlp, config)
        config2 = FrodConfig(target_modules=["lin1", "lin2"], init_weights=False)
        peft_model.add_adapter("other", config2)
        return peft_model

    def test_multiple_adapters_save_load_save_projection_false(self, mlp, tmp_path):
        # Check saving and loading works with multiple adapters without saved projection tensors.
        torch.manual_seed(1)
        config = FrodConfig(target_modules=["lin1", "lin2"], init_weights=False, save_projection=False)
        peft_model = get_peft_model(mlp, config, adapter_name="first")
        config2 = FrodConfig(target_modules=["lin1", "lin2"], init_weights=False, save_projection=False)
        peft_model.add_adapter("second", config2)
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
        config = FrodConfig(target_modules=["lin1", "lin2"], init_weights=False, save_projection=False)
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
        config = FrodConfig(target_modules=["lin1", "lin2"], init_weights=False)
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

    def test_frod_default_initialization_reconstructs_base_weight(self, mlp):
        torch.manual_seed(0)
        mlp.eval()
        inputs = torch.randn(5, 10)
        expected = mlp(inputs)

        config = FrodConfig(target_modules=["lin1", "lin2"])
        peft_model = get_peft_model(mlp, config)
        peft_model.eval()

        actual = peft_model(inputs)
        assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)

        for module in (peft_model.base_model.model.lin1, peft_model.base_model.model.lin2):
            delta_weight = module.get_delta_weight("default")

            assert module.frod_lambda_l["default"].norm() > 0
            assert torch.count_nonzero(module.frod_lambda_s_values["default"]) == 0
            assert torch.allclose(delta_weight, torch.zeros_like(delta_weight), atol=1e-4)

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

    def test_frod_sparse_activation_matches_dense_and_gradients(self, mlp):
        config = FrodConfig(target_modules=["lin1"], init_weights=False)
        peft_model = get_peft_model(mlp, config)
        layer = peft_model.base_model.model.lin1

        indices = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]])
        values = torch.tensor([0.5, -0.25, 1.5, 0.75, -1.0], dtype=torch.float16, requires_grad=True)
        sparse = torch.sparse_coo_tensor(indices, values, (4, 4)).coalesce()
        z = torch.randn(3, 4, dtype=torch.float16, requires_grad=True)

        actual = layer._sparse_activation_mm(z, sparse)
        actual.float().pow(2).sum().backward()

        z_expected = z.detach().clone().requires_grad_(True)
        values_expected = values.detach().clone().requires_grad_(True)
        dense = torch.zeros(4, 4, dtype=torch.float16)
        dense[indices[0], indices[1]] = values_expected
        expected = z_expected @ dense.t()
        expected.float().pow(2).sum().backward()

        assert values.grad is not None
        assert z.grad is not None
        assert torch.allclose(actual, expected, atol=1e-3, rtol=1e-3)
        assert torch.allclose(values.grad, values_expected.grad, atol=1e-3, rtol=1e-3)
        assert torch.allclose(z.grad, z_expected.grad, atol=1e-3, rtol=1e-3)

    def test_frod_autocast_keeps_frozen_u_in_base_dtype(self):
        model = MLP().to(torch.bfloat16)
        config = FrodConfig(target_modules=["lin1"], init_weights=False)
        peft_model = get_peft_model(model, config)
        lin1 = peft_model.base_model.model.lin1

        assert lin1.frod_U["default"].dtype == torch.bfloat16
        assert lin1.frod_lambda_l["default"].dtype == torch.float32
        assert lin1.frod_lambda_s_values["default"].dtype == torch.float32

    def test_frod_categories_with_common_llama_targets(self):
        model = LlamaForCausalLM(
            LlamaConfig(
                hidden_size=16,
                intermediate_size=32,
                num_attention_heads=4,
                num_hidden_layers=2,
                vocab_size=32,
            )
        )
        config = FrodConfig(target_modules=["q_proj", "v_proj"])

        peft_model = get_peft_model(model, config)

        assert sorted(peft_model.base_model.frod_V.keys()) == ["self_attn_q_proj", "self_attn_v_proj"]
        assert "default" in peft_model.base_model.frod_V["self_attn_q_proj"]
        assert "default" in peft_model.base_model.frod_V["self_attn_v_proj"]

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
        config = FrodConfig(target_modules=["lin0", "lin3"], init_weights=False)
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
        config = FrodConfig(target_modules=["lin1", "lin2"], init_weights=False)
        peft_model = get_peft_model(model, config)
        inputs = torch.randn(5, 10).to(dtype)
        output = peft_model(inputs)
        assert output.dtype == dtype

    def _build_two_adapter_model(self):
        # FRoD's delta is `frod_weight - base_weight`, so it is explicitly conditioned on the base weight.
        # Building this the exact same way every time (fresh model, same seed, same call order) means adapters
        # "A" and "B" always end up with identical random parameters across independently-built models, so the
        # different merge-call orderings below are directly comparable.
        torch.manual_seed(0)
        model = MLP()
        config_a = FrodConfig(target_modules=["lin1"], init_weights=False)
        peft_model = get_peft_model(model, config_a, adapter_name="A")
        config_b = FrodConfig(target_modules=["lin1"], init_weights=False)
        peft_model.add_adapter("B", config_b)
        return peft_model

    def test_sequential_merge_calls_match_combined_merge_regardless_of_order(self):
        # Regression test: separate, sequential merge() calls used to silently diverge from a single combined
        # merge() call, because each new call recomputed its delta against the *current* (possibly
        # already-merged) base weight instead of the layer's original, pristine one.
        combined_model = self._build_two_adapter_model()
        combined_layer = combined_model.base_model.model.lin1
        pristine_weight = combined_layer.base_layer.weight.data.clone()
        combined_layer.merge(adapter_names=["A", "B"])
        ground_truth = combined_layer.base_layer.weight.data.clone()

        # sanity check: merging must actually change the weight, else the comparisons below are vacuous
        assert not torch.allclose(ground_truth, pristine_weight, atol=1e-6)

        model_ab = self._build_two_adapter_model()
        layer_ab = model_ab.base_model.model.lin1
        layer_ab.merge(adapter_names=["A"])
        layer_ab.merge(adapter_names=["B"])
        weight_ab = layer_ab.base_layer.weight.data.clone()

        model_ba = self._build_two_adapter_model()
        layer_ba = model_ba.base_model.model.lin1
        layer_ba.merge(adapter_names=["B"])
        layer_ba.merge(adapter_names=["A"])
        weight_ba = layer_ba.base_layer.weight.data.clone()

        assert torch.allclose(weight_ab, ground_truth, atol=1e-4, rtol=1e-4)
        assert torch.allclose(weight_ba, ground_truth, atol=1e-4, rtol=1e-4)
        # the clearest proof this was a real bug and not just numerical noise: merging in the other order used
        # to produce a *different* wrong answer
        assert torch.allclose(weight_ab, weight_ba, atol=1e-4, rtol=1e-4)

    def test_unmerge_after_sequential_merge_calls_restores_pristine_weight(self):
        # The delta *values* cached at merge time and subtracted by unmerge() were never wrong -- only the
        # computation of a *new* delta during a later, separate merge() call was. So after fully unmerging,
        # the base weight must come back exactly to the original, regardless of how many separate merge()
        # calls were used to get there.
        model = self._build_two_adapter_model()
        layer = model.base_model.model.lin1
        pristine_weight = layer.base_layer.weight.data.clone()

        layer.merge(adapter_names=["A"])
        layer.merge(adapter_names=["B"])
        assert not torch.allclose(layer.base_layer.weight.data, pristine_weight, atol=1e-6)

        layer.unmerge()
        assert torch.allclose(layer.base_layer.weight.data, pristine_weight, atol=1e-5, rtol=1e-5)
