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

from peft import PeftModel, VBLoRAConfig, get_peft_model


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


class TestVBLoRA:
    def get_mlp(self):
        model = MLP()
        return model

    def test_vblora_parameters(self):
        mlp = self.get_mlp()
        vector_length = 2
        num_vectors = 10
        config = VBLoRAConfig(
            target_modules=["lin0", "lin1", "lin3"], vector_length=vector_length, num_vectors=num_vectors
        )
        mlp_vblora = get_peft_model(mlp, config)

        vector_bank = mlp_vblora.vblora_vector_bank["default"]

        vblora_lin0_logits_B = mlp_vblora.lin0.vblora_logits_B["default"]
        assert vblora_lin0_logits_B.shape == (mlp.lin0.out_features // vector_length, config.r, num_vectors)

        vblora_lin1_logits_A = mlp_vblora.lin1.vblora_logits_A["default"]
        assert vblora_lin1_logits_A.shape == (config.r, mlp.lin1.in_features // vector_length, num_vectors)

        vblora_lin3_logits_A = mlp_vblora.lin3.vblora_logits_A["default"]
        assert vblora_lin3_logits_A.shape == (config.r, mlp.lin3.in_features // vector_length, num_vectors)

        assert vector_bank.shape == (num_vectors, vector_length)

        # test if the vector bank is shared across the layers
        assert (
            mlp_vblora.lin0.vblora_vector_bank["default"].data_ptr()
            == mlp_vblora.lin3.vblora_vector_bank["default"].data_ptr()
        )
        assert mlp_vblora.lin1.vblora_vector_bank["default"].data_ptr() == vector_bank.data_ptr()

        # should not raise
        input = torch.randn(5, 10)
        mlp_vblora(input)

    def test_save_with_topk_weights(self, tmp_path):
        torch.manual_seed(0)
        mlp = self.get_mlp()
        vector_length = 2
        num_vectors = 10
        topk = 2
        config = VBLoRAConfig(
            target_modules=["lin0", "lin3"],
            topk=topk,
            vector_length=vector_length,
            num_vectors=num_vectors,
            save_only_topk_weights=True,
        )
        mlp_vblora = get_peft_model(mlp, config)
        save_path = tmp_path / "vblora"
        mlp_vblora.save_pretrained(save_path)
        assert os.path.exists(save_path / "adapter_model.safetensors")

        adapter_model_dict = {}
        with safe_open(save_path / "adapter_model.safetensors", framework="pt") as f:
            for k in f.keys():
                adapter_model_dict[k] = f.get_tensor(k)
        assert "base_model.model.lin0.vblora_logits_A_topk_indices" in adapter_model_dict
        assert "base_model.model.lin0.vblora_logits_A_topk_weights" in adapter_model_dict
        assert "base_model.model.lin3.vblora_logits_B_topk_indices" in adapter_model_dict
        assert "base_model.model.lin3.vblora_logits_B_topk_weights" in adapter_model_dict
        assert "base_model.model.lin0.vblora_logits_A" not in adapter_model_dict
        assert "base_model.model.lin3.vblora_logits_B" not in adapter_model_dict

        assert adapter_model_dict["base_model.model.lin0.vblora_logits_B_topk_indices"].shape == (
            mlp.lin0.out_features // vector_length,
            config.r,
            topk,
        )
        assert adapter_model_dict["base_model.model.lin0.vblora_logits_B_topk_weights"].shape == (
            mlp.lin0.out_features // vector_length,
            config.r,
            topk - 1,
        )
        assert adapter_model_dict["base_model.model.lin3.vblora_logits_A_topk_indices"].shape == (
            config.r,
            mlp.lin3.in_features // vector_length,
            topk,
        )
        assert adapter_model_dict["base_model.model.lin3.vblora_logits_A_topk_weights"].shape == (
            config.r,
            mlp.lin3.in_features // vector_length,
            topk - 1,
        )

    @pytest.mark.parametrize("save_only_topk_weights", [True, False])
    def test_save_load(self, save_only_topk_weights, tmp_path):
        torch.manual_seed(0)
        mlp = self.get_mlp()
        config = VBLoRAConfig(
            target_modules=["lin0", "lin1", "lin3"],
            topk=2,
            vector_length=2,
            num_vectors=10,
            save_only_topk_weights=save_only_topk_weights,
        )
        mlp_vblora = get_peft_model(mlp, config)
        save_path = tmp_path / "vblora"
        mlp_vblora.save_pretrained(save_path)
        assert os.path.exists(save_path / "adapter_config.json")

        del mlp
        torch.manual_seed(0)  # make sure the base model has the same weights
        mlp = self.get_mlp()
        mlp_vblora_loaded = PeftModel.from_pretrained(mlp, save_path)

        input = torch.randn(5, 10)
        output = mlp_vblora(input)
        output_loaded = mlp_vblora_loaded(input)
        assert torch.allclose(output, output_loaded, atol=1e-8, rtol=1e-5)

    def test_resume_training_model_with_topk_weights(self, tmp_path):
        torch.manual_seed(1)
        mlp = self.get_mlp()
        config = VBLoRAConfig(
            target_modules=["lin0", "lin1", "lin3"],
            topk=2,
            vector_length=2,
            num_vectors=10,
            save_only_topk_weights=True,
        )
        mlp_vblora = get_peft_model(mlp, config)
        save_path = tmp_path / "vblora"
        mlp_vblora.save_pretrained(save_path)

        input = torch.randn(5, 10)
        mlp_vblora.train()
        # should not raise
        mlp_vblora(input)

        del mlp
        torch.manual_seed(1)
        mlp = self.get_mlp()
        mlp_vblora_loaded = PeftModel.from_pretrained(mlp, save_path)
        mlp_vblora_loaded.train()
        msg = "Found infinity values in VB-LoRA logits. Ensure training was not resumed from a `save_only_topk_weights` model."
        with pytest.raises(RuntimeError, match=msg):
            mlp_vblora_loaded(input)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_vblora_dtypes(self, dtype):
        mlp = self.get_mlp()
        if (dtype == torch.bfloat16) and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            pytest.skip("bfloat16 not supported on this system, skipping the test")

        config = VBLoRAConfig(
            target_modules=["lin0", "lin1", "lin3"], vector_length=2, num_vectors=10, save_only_topk_weights=False
        )
        mlp_vblora = get_peft_model(mlp.to(dtype), config)
        inputs = torch.randn(5, 10).to(dtype)
        output = mlp_vblora(inputs)  # should not raise
        assert output.dtype == dtype

    def test_vblora_nb_savable_params_only_topk_weights(self):
        mlp = self.get_mlp()
        vector_length = 2
        num_vectors = 10
        topk = 2
        r = 4
        config = VBLoRAConfig(
            target_modules=["lin0", "lin1"],
            vector_length=vector_length,
            num_vectors=num_vectors,
            topk=topk,
            r=r,
            save_only_topk_weights=True,
        )
        mlp_vblora = get_peft_model(mlp, config)

        mlp_vblora.lin3.requires_grad_(True)  # set lin3 to trainable

        adapter_params, other_params = mlp_vblora.get_nb_savable_parameters()
        factor = 0.25  # dtype of index is uint8
        topk_indices_parameter = int(
            (mlp.lin0.out_features + mlp.lin0.in_features + mlp.lin1.out_features + mlp.lin1.in_features)
            / vector_length
            * r
            * topk
            * factor
        )
        topk_weights_parameter = int(
            (mlp.lin0.out_features + mlp.lin0.in_features + mlp.lin1.out_features + mlp.lin1.in_features)
            / vector_length
            * r
            * (topk - 1)
        )
        vector_bank_parameter = num_vectors * vector_length
        assert adapter_params == topk_indices_parameter + topk_weights_parameter + vector_bank_parameter
        assert other_params == (mlp.lin3.in_features + 1) * mlp.lin3.out_features

    def test_vblora_nb_savable_params_all_logits(self):
        mlp = self.get_mlp()
        vector_length = 2
        num_vectors = 10
        topk = 2
        r = 4
        config = VBLoRAConfig(
            target_modules=["lin0", "lin1"],
            vector_length=vector_length,
            num_vectors=num_vectors,
            topk=topk,
            r=r,
            save_only_topk_weights=False,
        )
        mlp_vblora = get_peft_model(mlp, config)

        mlp_vblora.lin3.requires_grad_(True)  # set lin3 to trainable

        adapter_params, other_params = mlp_vblora.get_nb_savable_parameters()
        logits_parameter = int(
            (mlp.lin0.out_features + mlp.lin0.in_features + mlp.lin1.out_features + mlp.lin1.in_features)
            / vector_length
            * r
            * num_vectors
        )
        vector_bank_parameter = num_vectors * vector_length
        assert adapter_params == logits_parameter + vector_bank_parameter
        assert other_params == (mlp.lin3.in_features + 1) * mlp.lin3.out_features
