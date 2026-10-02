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
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from peft import LoHaConfig, get_peft_model
from peft.tuners.loha.layer import HadaWeight, HadaWeightCP


ADAPTER_PARAM_NAMES = ["hada_t1", "hada_w1_a", "hada_w1_b", "hada_t2", "hada_w2_a", "hada_w2_b"]


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.float64, requires_grad=True)


class TestLoHaCustomGradients:
    # HadaWeight and HadaWeightCP implement their backward pass manually, check it against numerical gradients

    def test_hada_weight_gradcheck(self):
        torch.manual_seed(0)
        r, out_features, in_features = 3, 5, 4
        args = (_rand(out_features, r), _rand(r, in_features), _rand(out_features, r), _rand(r, in_features))
        scale = torch.tensor(0.7, dtype=torch.float64)
        assert torch.autograd.gradcheck(lambda *tensors: HadaWeight.apply(*tensors, scale), args)

    @pytest.mark.parametrize("kernel_size", [(3, 3), (3, 1), (1, 3)])
    def test_hada_weight_cp_gradcheck(self, kernel_size):
        torch.manual_seed(0)
        r, out_channels, in_channels = 3, 5, 4
        args = (
            _rand(r, r, *kernel_size),
            _rand(r, out_channels),
            _rand(r, in_channels),
            _rand(r, r, *kernel_size),
            _rand(r, out_channels),
            _rand(r, in_channels),
        )
        scale = torch.tensor(0.7, dtype=torch.float64)
        assert torch.autograd.gradcheck(lambda *tensors: HadaWeightCP.apply(*tensors, scale), args)

    @pytest.mark.parametrize(
        "conv_cls, kernel_size, input_shape",
        [
            (nn.Conv2d, 3, (2, 4, 8, 8)),
            (nn.Conv2d, (3, 1), (2, 4, 8, 8)),
            (nn.Conv1d, 3, (2, 4, 9)),
        ],
    )
    def test_effective_conv2d_gradients_match_autograd(self, conv_cls, kernel_size, input_shape):
        # With use_effective_conv2d=True, the delta weight is built by HadaWeightCP. Its gradients should match the
        # ones autograd computes for the same delta weight built with plain einsum operations.
        torch.manual_seed(0)
        config = LoHaConfig(target_modules=["0"], r=2, alpha=2, use_effective_conv2d=True, init_weights=False)
        model = get_peft_model(nn.Sequential(conv_cls(4, 6, kernel_size=kernel_size)).double(), config)
        layer = model.base_model.model[0]
        assert "default" in layer.hada_t1  # the CP path is used

        x = torch.randn(*input_shape, dtype=torch.float64)
        model(x).pow(2).sum().backward()

        params = {
            name: getattr(layer, name)["default"].detach().clone().requires_grad_() for name in ADAPTER_PARAM_NAMES
        }
        rebuild1 = torch.einsum(
            "i j k l, j r, i p -> p r k l", params["hada_t1"], params["hada_w1_b"], params["hada_w1_a"]
        )
        rebuild2 = torch.einsum(
            "i j k l, j r, i p -> p r k l", params["hada_t2"], params["hada_w2_b"], params["hada_w2_a"]
        )
        conv = layer.get_base_layer()
        delta = (rebuild1 * rebuild2 * layer.scaling["default"]).reshape(conv.weight.shape)
        conv_fn = F.conv2d if isinstance(conv, nn.Conv2d) else F.conv1d
        conv_fn(x, conv.weight + delta, conv.bias).pow(2).sum().backward()

        for name in ADAPTER_PARAM_NAMES:
            torch.testing.assert_close(getattr(layer, name)["default"].grad, params[name].grad)
