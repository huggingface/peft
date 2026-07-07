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

# This test file is for tests specific to EWoRA.
#
# NOTE: EWoRA's forward pass unpacks ``bs, seq_len, _ = x.size()``, i.e. it requires
# 3-D inputs of shape (batch, seq_len, hidden). All tests below therefore drive the
# model with 3-D tensors. This is also why EWoRA is not wired into the MLP-based
# ``test_custom_models.py`` suite, whose activations are 2-D.

import os

import pytest
import torch
from accelerate.utils.imports import is_bf16_available
from torch import nn

from peft import EworaConfig, PeftModel, get_peft_model


class MLP(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(20, 20, bias=bias)
        self.sm = nn.LogSoftmax(dim=-1)

    def forward(self, X):
        # X is 3-D: (batch, seq_len, hidden)
        X = self.lin0(X)
        X = self.relu(X)
        X = self.lin1(X)
        X = self.sm(X)
        return X


class TestEwora:
    @pytest.fixture
    def mlp(self):
        torch.manual_seed(0)
        return MLP()

    def test_adapter_creates_expert_weighted_params(self, mlp):
        # EWoRA holds one (in_features, r) "A" and one (r, out_features) "B" per expert.
        r, num_experts = 2, 4
        config = EworaConfig(r=r, num_experts=num_experts, target_modules=["lin0", "lin1"])
        peft_model = get_peft_model(mlp, config)

        lin0 = peft_model.base_model.model.lin0
        assert lin0.ewora_As["default"].shape == (num_experts, 10, r)  # lin0 in_features=10
        assert lin0.ewora_Bs["default"].shape == (num_experts, r, 20)  # lin0 out_features=20
        assert lin0.num_experts["default"] == num_experts

    def test_only_ewora_params_are_trainable(self, mlp):
        config = EworaConfig(r=2, num_experts=4, target_modules=["lin0", "lin1"])
        peft_model = get_peft_model(mlp, config)
        for name, param in peft_model.named_parameters():
            if "ewora_" in name:
                assert param.requires_grad
            else:
                assert not param.requires_grad

    def test_forward_3d_output_shape(self, mlp):
        config = EworaConfig(r=2, num_experts=4, target_modules=["lin0", "lin1"])
        peft_model = get_peft_model(mlp, config)
        inputs = torch.randn(3, 5, 10)  # (batch, seq_len, hidden)
        output = peft_model(inputs)
        assert output.shape == (3, 5, 20)

    def test_default_init_is_noop_then_weights_change_output(self, mlp):
        config = EworaConfig(r=2, num_experts=4, target_modules=["lin0", "lin1"])
        peft_model = get_peft_model(mlp, config)
        inputs = torch.randn(3, 5, 10)

        # Default init sets the "B" matrices to zero, so a freshly-created adapter is a
        # no-op: its output must equal the frozen base model's output (adapters disabled).
        with peft_model.disable_adapter():
            base_output = peft_model(inputs)
        assert torch.allclose(base_output, peft_model(inputs), atol=1e-6)

        # Once the "B" matrices are non-zero, the adapter must change the output.
        with torch.no_grad():
            for name, param in peft_model.named_parameters():
                if "ewora_Bs" in name:
                    param.copy_(torch.randn_like(param))
        assert not torch.allclose(base_output, peft_model(inputs), atol=1e-4)

    def test_save_and_load_roundtrip(self, mlp, tmp_path):
        config = EworaConfig(r=2, num_experts=4, target_modules=["lin0", "lin1"])
        peft_model = get_peft_model(mlp, config)

        # Give the adapter non-trivial weights so the round-trip is meaningful.
        with torch.no_grad():
            for name, param in peft_model.named_parameters():
                if "ewora_Bs" in name or "ewora_weighting" in name:
                    param.copy_(torch.randn_like(param))

        inputs = torch.randn(3, 5, 10)
        output_before = peft_model(inputs)

        save_path = os.path.join(tmp_path, "ewora")
        peft_model.save_pretrained(save_path)
        assert os.path.exists(os.path.join(save_path, "adapter_config.json"))
        del peft_model

        torch.manual_seed(0)
        loaded = PeftModel.from_pretrained(MLP(), save_path)
        output_after = loaded(inputs)
        assert torch.allclose(output_before, output_after, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_ewora_dtypes(self, dtype):
        if dtype == torch.bfloat16 and not is_bf16_available():
            pytest.skip("bfloat16 not supported on this system, skipping the test")
        torch.manual_seed(0)
        model = MLP().to(dtype)
        config = EworaConfig(r=2, num_experts=4, target_modules=["lin0", "lin1"])
        peft_model = get_peft_model(model, config)
        inputs = torch.randn(3, 5, 10).to(dtype)
        output = peft_model(inputs)  # should not raise
        assert output.dtype == dtype

    @pytest.mark.parametrize("num_experts", [1, 2, 8])
    def test_num_experts_reflected_in_param_shape(self, mlp, num_experts):
        config = EworaConfig(r=3, num_experts=num_experts, target_modules=["lin0"])
        peft_model = get_peft_model(mlp, config)
        lin0 = peft_model.base_model.model.lin0
        assert lin0.ewora_As["default"].shape[0] == num_experts
        assert lin0.ewora_Bs["default"].shape[0] == num_experts

    def test_ewora_cannot_be_merged(self, mlp):
        # EWoRA weights its experts dynamically per input, so there is no static delta to fold
        # into the base weights. Merging must raise a clear, EWoRA-specific error.
        config = EworaConfig(r=2, num_experts=4, target_modules=["lin0", "lin1"])
        peft_model = get_peft_model(mlp, config)
        with pytest.raises(NotImplementedError):
            peft_model.merge_and_unload()
