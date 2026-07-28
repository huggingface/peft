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
    dare_linear,
    dare_ties,
    magnitude_prune,
    prune,
    random_pruning,
    ties,
)


MERGE_METHODS = {
    "ties": lambda tensors, weights, density: ties(tensors, weights, density),
    "dare_linear": dare_linear,
    "dare_ties": lambda tensors, weights, density: dare_ties(tensors, weights, density),
    "magnitude_prune": magnitude_prune,
}


@pytest.fixture
def task_tensors():
    tensor = torch.arange(1.0, 9.0).reshape(2, 4)
    return [tensor, tensor * 2]


@pytest.fixture
def weights():
    return torch.tensor([0.5, 0.5])


@pytest.mark.parametrize("method", sorted(MERGE_METHODS))
def test_merge_with_zero_density_prunes_everything(method, task_tensors, weights):
    """`density=0` means "prune all values", so the merged delta must be zeros rather than NaN.

    `random_pruning` rescales with `pruned / density` to preserve the expected value. At
    `density=0` everything is already pruned, so that divides zero by zero and turns the whole
    delta into NaN, which then propagates into the merged adapter weights with no error raised.
    The two magnitude-based methods were unaffected, which is what made it easy to miss.
    """
    merged = MERGE_METHODS[method]([tensor.clone() for tensor in task_tensors], weights, 0.0)

    assert not torch.isnan(merged).any()
    assert torch.equal(merged, torch.zeros_like(merged))


@pytest.mark.parametrize("method", sorted(MERGE_METHODS))
@pytest.mark.parametrize("density", [0.25, 0.5, 1.0])
def test_merge_with_nonzero_density_is_finite(method, density, task_tensors, weights):
    """Densities in (0, 1] keep working, and the rescaling still happens."""
    merged = MERGE_METHODS[method]([tensor.clone() for tensor in task_tensors], weights, density)

    assert torch.isfinite(merged).all()


def test_random_pruning_rescales_for_nonzero_density():
    """The `density=0` guard must not disable rescaling for the densities that do use it."""
    tensor = torch.ones(1000)

    pruned = random_pruning(tensor, density=0.5, rescale=True)

    # Surviving entries are scaled by 1 / density, so the ones that are kept become 2.0.
    kept = pruned[pruned != 0]
    assert len(kept) > 0
    assert torch.equal(kept, torch.full_like(kept, 2.0))


def test_prune_with_zero_density_returns_zeros():
    """Both pruning methods agree that `density=0` drops everything."""
    tensor = torch.arange(1.0, 9.0).reshape(2, 4)

    for method in ("magnitude", "random"):
        pruned = prune(tensor.clone(), density=0.0, method=method, rescale=True)
        assert not torch.isnan(pruned).any(), method
        assert torch.equal(pruned, torch.zeros_like(pruned)), method
