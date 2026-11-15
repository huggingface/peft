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

from peft.utils.merge_utils import (
    calculate_majority_sign_mask,
    dare_ties,
    ties,
)


class TestMergeUtils:
    def test_ties_majority_sign_unequal_weights(self):
        # Issue #4169: majority sign should be elected on weighted task tensors
        a = torch.tensor([1.0])
        b = torch.tensor([-2.0])
        weights = torch.tensor([1.0, 0.1])

        # Raw values: -2.0 has larger magnitude than 1.0.
        # But weighted values: a is 1.0, b is -0.2.
        # Sign election on weighted tensors correctly elects +1 (from adapter a).
        result = ties([a, b], weights, density=1.0)
        expected = torch.tensor([1.0])
        assert torch.allclose(result, expected)

    def test_ties_majority_sign_negative_weights(self):
        # Issue #4169: negative weights invert tensor sign prior to majority election
        a = torch.tensor([2.0, 2.0, -1.0])
        b = torch.tensor([1.0, 3.0, 1.0])
        weights = torch.tensor([1.0, -0.5])

        # Weighted contributions:
        # idx 0: [2.0, -0.5] -> sign +1 -> 2.0
        # idx 1: [2.0, -1.5] -> sign +1 -> 2.0
        # idx 2: [-1.0, -0.5] -> sign -1 (both agree) -> (-1.0 + -0.5) / 2 = -0.75
        result = ties([a, b], weights, density=1.0)
        expected = torch.tensor([2.0, 2.0, -0.75])
        assert torch.allclose(result, expected)

    def test_ties_equal_positive_weights_backward_compat(self):
        # When weights are equal and positive, behavior is identical to raw unweighted election
        a = torch.tensor([3.0, -2.0, 1.0])
        b = torch.tensor([1.0, -1.0, -2.0])
        weights = torch.tensor([1.0, 1.0])

        result = ties([a, b], weights, density=1.0)
        # idx 0: [3, 1] -> +1 -> (3+1)/2 = 2.0
        # idx 1: [-2, -1] -> -1 -> (-2 + -1)/2 = -1.5
        # idx 2: [1, -2] -> majority sign total: 1 + (-2) = -1 -> only b kept: -2.0
        expected = torch.tensor([2.0, -1.5, -2.0])
        assert torch.allclose(result, expected)

    def test_dare_ties_majority_sign_weighted(self):
        # dare_ties should also elect sign on weighted task tensors
        torch.manual_seed(42)
        a = torch.tensor([1.0])
        b = torch.tensor([-2.0])
        weights = torch.tensor([1.0, 0.1])

        result = dare_ties([a, b], weights, density=1.0)
        expected = torch.tensor([1.0])
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize("method", ["total", "frequency"])
    def test_calculate_majority_sign_mask_methods(self, method):
        stacked = torch.tensor([[1.0, -2.0], [2.0, 1.0]])
        mask = calculate_majority_sign_mask(stacked, method=method)
        assert mask.shape == stacked.shape
