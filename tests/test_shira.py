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

# This test file is for tests specific to SHiRA.

import os

import pytest
import torch
from accelerate.utils.imports import is_bf16_available
from torch import nn

from peft import PeftModel, ShiraConfig, get_peft_model


class MLP(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.lin1 = nn.Linear(20, 40, bias=bias)  # lin1 and lin2 have same shape
        self.lin2 = nn.Linear(40, 30, bias=bias)
        self.lin3 = nn.Linear(30, 10, bias=bias)
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


class TestShira:
    @pytest.fixture
    def mlp(self):
        torch.manual_seed(0)
        model = MLP()
        return model

    def test_mlp_single_adapter_shapes(self, mlp):
        # torch.manual_seed(0)

        r = 2
        config = ShiraConfig(r=r, target_modules=["lin1", "lin2"])
        # creates a default SHiRA adapter
        peft_model = get_peft_model(mlp, config)

        shira_weight1_size = peft_model.base_model.model.lin1.shira_weight["default"].shape[0]
        shira_weight2_size = peft_model.base_model.model.lin2.shira_weight["default"].shape[0]
        shira_indices1_size = peft_model.base_model.model.lin1.shira_indices["default"].shape[1]
        shira_indices2_size = peft_model.base_model.model.lin2.shira_indices["default"].shape[1]

        base_weight1_size = peft_model.base_model.model.lin1.base_layer.weight.shape
        base_weight2_size = peft_model.base_model.model.lin2.base_layer.weight.shape

        delta_weight1_shape = peft_model.base_model.model.lin1.get_delta_weight("default").shape
        delta_weight2_shape = peft_model.base_model.model.lin2.get_delta_weight("default").shape

        assert shira_weight1_size == r * (base_weight1_size[0] + base_weight1_size[1])
        assert shira_weight2_size == r * (base_weight2_size[0] + base_weight2_size[1])

        assert shira_weight1_size == shira_indices1_size
        assert shira_weight2_size == shira_indices2_size

        assert delta_weight1_shape == base_weight1_size
        assert delta_weight2_shape == base_weight2_size

        return peft_model

    def test_multiple_adapters_save_load(self, mlp, tmp_path):
        # check saving and loading works with multiple adapters
        # note, the random seeds in the below two configs are not the default values.
        # so it will lead to different random sparse masks between saving and loading.
        # our goal is to make sure that loaded indices are exactly the same as the saved indices regardless of what initial random mask gets generated.
        # we will also make sure that parameters are saved and loaded correctly, and the output remains the same.
        # torch.manual_seed(1)
        config = ShiraConfig(r=2, target_modules=["lin1", "lin2"], random_seed=56)
        # creates a default SHiRA adapter
        peft_model = get_peft_model(mlp, config, adapter_name="first")
        config2 = ShiraConfig(r=3, target_modules=["lin1", "lin2", "lin3"], random_seed=67)
        peft_model.add_adapter("second", config2)

        assert torch.all(peft_model.base_model.model.lin1.shira_weight["first"] == 0)
        assert torch.all(peft_model.base_model.model.lin2.shira_weight["first"] == 0)
        assert torch.all(peft_model.base_model.model.lin1.shira_weight["second"] == 0)
        assert torch.all(peft_model.base_model.model.lin2.shira_weight["second"] == 0)
        assert torch.all(peft_model.base_model.model.lin3.shira_weight["second"] == 0)

        shira_assign_val1_f = torch.randn_like(peft_model.base_model.model.lin1.shira_weight["first"])
        peft_model.base_model.model.lin1.shira_weight["first"] = shira_assign_val1_f
        shira_indices1_f = peft_model.base_model.model.lin1.shira_indices["first"]
        shira_assign_val2_f = torch.randn_like(peft_model.base_model.model.lin2.shira_weight["first"])
        peft_model.base_model.model.lin2.shira_weight["first"] = shira_assign_val2_f
        shira_indices2_f = peft_model.base_model.model.lin2.shira_indices["first"]

        shira_assign_val1_s = torch.randn_like(peft_model.base_model.model.lin1.shira_weight["second"])
        peft_model.base_model.model.lin1.shira_weight["second"] = shira_assign_val1_s
        shira_indices1_s = peft_model.base_model.model.lin1.shira_indices["second"]
        shira_assign_val2_s = torch.randn_like(peft_model.base_model.model.lin2.shira_weight["second"])
        peft_model.base_model.model.lin2.shira_weight["second"] = shira_assign_val2_s
        shira_indices2_s = peft_model.base_model.model.lin2.shira_indices["second"]
        shira_assign_val3_s = torch.randn_like(peft_model.base_model.model.lin3.shira_weight["second"])
        peft_model.base_model.model.lin3.shira_weight["second"] = shira_assign_val3_s
        shira_indices3_s = peft_model.base_model.model.lin3.shira_indices["second"]

        input = torch.randn(5, 10)
        peft_model.set_adapter("first")
        output_first = peft_model(input)
        peft_model.set_adapter("second")
        output_second = peft_model(input)

        # sanity check
        assert not torch.allclose(output_first, output_second, atol=1e-3, rtol=1e-3)

        save_path = os.path.join(tmp_path, "shira")
        peft_model.save_pretrained(save_path)
        assert os.path.exists(os.path.join(save_path, "first", "adapter_config.json"))
        assert os.path.exists(os.path.join(save_path, "second", "adapter_config.json"))
        del peft_model

        torch.manual_seed(0)
        mlp = MLP()
        peft_model = PeftModel.from_pretrained(mlp, os.path.join(save_path, "first"), adapter_name="first")
        peft_model.load_adapter(os.path.join(save_path, "second"), "second")

        peft_model.set_adapter("first")
        output_first_loaded = peft_model(input)
        peft_model.set_adapter("second")
        output_second_loaded = peft_model(input)

        assert torch.all(output_first - output_first_loaded == 0)
        assert torch.all(output_second - output_second_loaded == 0)

        assert torch.all(shira_assign_val1_f - peft_model.base_model.model.lin1.shira_weight["first"] == 0)
        assert torch.all(shira_assign_val2_f - peft_model.base_model.model.lin2.shira_weight["first"] == 0)
        assert torch.all(shira_indices1_f - peft_model.base_model.model.lin1.shira_indices["first"] == 0)
        assert torch.all(shira_indices2_f - peft_model.base_model.model.lin2.shira_indices["first"] == 0)
        assert torch.all(shira_assign_val1_s - peft_model.base_model.model.lin1.shira_weight["second"] == 0)
        assert torch.all(shira_assign_val2_s - peft_model.base_model.model.lin2.shira_weight["second"] == 0)
        assert torch.all(shira_assign_val3_s - peft_model.base_model.model.lin3.shira_weight["second"] == 0)
        assert torch.all(shira_indices1_s - peft_model.base_model.model.lin1.shira_indices["second"] == 0)
        assert torch.all(shira_indices2_s - peft_model.base_model.model.lin2.shira_indices["second"] == 0)
        assert torch.all(shira_indices3_s - peft_model.base_model.model.lin3.shira_indices["second"] == 0)

        return peft_model

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_shira_dtypes(self, dtype):
        if dtype == torch.bfloat16:
            # skip if bf16 is not supported on hardware, see #1872
            if not is_bf16_available():
                pytest.skip("bfloat16 not supported on this system, skipping the test")

        model = MLP().to(dtype)
        config = ShiraConfig(r=2, target_modules=["lin1", "lin2"])
        peft_model = get_peft_model(model, config)
        inputs = torch.randn(5, 10).to(dtype)
        output = peft_model(inputs)  # should not raise
        assert output.dtype == dtype
