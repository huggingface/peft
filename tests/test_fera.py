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

import os

import pytest
import torch
from accelerate.utils.imports import is_bf16_available
from safetensors import safe_open
from torch import nn

from peft import FeRAConfig, FeRALinear, PeftModel, get_peft_model


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


class TestFeRA:
    @pytest.fixture
    def mlp(self):
        torch.manual_seed(0)
        model = MLP()
        return model

    @pytest.fixture
    def fera_model(self, mlp):
        torch.manual_seed(0)
        config = FeRAConfig(target_modules=["lin1", "lin2"], init_lora_weights=False, num_bands=3, num_experts=3)
        peft_model = get_peft_model(mlp, config)
        return peft_model

    def test_attributes_and_structure(self, fera_model):
        assert isinstance(fera_model.base_model.model.lin1, FeRALinear)
        assert isinstance(fera_model.base_model.model.lin2, FeRALinear)
        assert hasattr(fera_model, "router")
        assert hasattr(fera_model, "fei_indicator")

    def test_forward_requires_prepare(self, fera_model):
        input_data = torch.randn(5, 10)
        dummy_latent = torch.randn(5, 4, 32, 32)

        with torch.no_grad():
            output_base = fera_model.get_base_model()(input_data)
            output_fera_no_route = fera_model(input_data)

        assert torch.allclose(output_base, output_fera_no_route)

        routing_weights = fera_model.prepare_forward(dummy_latent)
        assert routing_weights.shape == (5, 3)  # (Batch, Num_Experts)

        with torch.no_grad():
            for layer in [fera_model.base_model.model.lin1, fera_model.base_model.model.lin2]:
                for experts in layer.experts.values():
                    for expert in experts:
                        nn.init.normal_(expert.lora_up.weight)

            output_fera_routed = fera_model(input_data)

        assert not torch.allclose(output_base, output_fera_routed)

    def test_multiple_adapters_switching(self, mlp):
        torch.manual_seed(0)
        config1 = FeRAConfig(target_modules=["lin1"], num_experts=3, num_bands=3)
        peft_model = get_peft_model(mlp, config1, adapter_name="default")

        config2 = FeRAConfig(target_modules=["lin1"], num_experts=3, num_bands=3)
        peft_model.add_adapter("other", config2)

        input_data = torch.randn(2, 10)
        dummy_latent = torch.randn(2, 4, 16, 16)

        peft_model.set_adapter("default")
        peft_model.prepare_forward(dummy_latent)
        with torch.no_grad():
            nn.init.constant_(peft_model.base_model.model.lin1.experts["default"][0].lora_up.weight, 10.0)
        output_default = peft_model(input_data)

        peft_model.set_adapter("other")
        peft_model.prepare_forward(dummy_latent)
        with torch.no_grad():
            nn.init.constant_(peft_model.base_model.model.lin1.experts["other"][0].lora_up.weight, -10.0)
        output_other = peft_model(input_data)

        assert not torch.allclose(output_default, output_other)

    def test_save_load(self, fera_model, tmp_path):
        torch.manual_seed(0)
        input_data = torch.randn(2, 10)
        dummy_latent = torch.randn(2, 4, 16, 16)

        fera_model.prepare_forward(dummy_latent)
        output_before = fera_model(input_data)

        save_path = tmp_path / "fera_model"
        fera_model.save_pretrained(save_path)

        assert os.path.exists(save_path / "adapter_config.json")
        assert os.path.exists(save_path / "adapter_model.safetensors")

        mlp_new = MLP()
        peft_model_new = PeftModel.from_pretrained(mlp_new, save_path)

        state_dict_loaded = {}
        with safe_open(save_path / "adapter_model.safetensors", framework="pt", device="cpu") as f:
            for k in f.keys():
                state_dict_loaded[k] = f.get_tensor(k)

        has_router_keys = any("router" in k for k in state_dict_loaded.keys())
        assert has_router_keys, "Router weights were not saved in the checkpoint!"

        peft_model_new.prepare_forward(dummy_latent)
        output_after = peft_model_new(input_data)

        assert torch.allclose(output_before, output_after)

    def test_shared_router(self, fera_model):
        dummy_latent = torch.randn(2, 4, 16, 16)

        weights = fera_model.prepare_forward(dummy_latent)

        lin1 = fera_model.base_model.model.lin1
        lin2 = fera_model.base_model.model.lin2

        assert lin1.current_routing_weights is not None
        assert lin2.current_routing_weights is not None

        assert torch.allclose(lin1.current_routing_weights, weights)
        assert torch.allclose(lin2.current_routing_weights, weights)

        assert lin1.current_routing_weights is lin2.current_routing_weights

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_fera_dtypes(self, dtype):
        if dtype == torch.bfloat16 and not is_bf16_available():
            pytest.skip("bfloat16 not supported on this system")

        model = MLP().to(dtype)
        config = FeRAConfig(target_modules=["lin1"], num_experts=3, num_bands=3)
        peft_model = get_peft_model(model, config)

        input_data = torch.randn(5, 10).to(dtype)
        dummy_latent = torch.randn(5, 4, 16, 16).to(dtype)

        peft_model.prepare_forward(dummy_latent)
        output = peft_model(input_data)

        assert output.dtype == dtype
