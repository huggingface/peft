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

from peft.tuners.loha.layer import HadaWeight, HadaWeightCP


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

    # (3, 1) is also the layout used for Conv1d layers
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
