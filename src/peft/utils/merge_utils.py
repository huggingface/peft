# coding=utf-8
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

from typing import Literal

import torch


def magnitude_based_pruning(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """
    Prune the smallest values of the task tensors and retain the top-k values based on the specified fraction `density`.

    Args:
    tensor (`torch.Tensor`):The tensor to prune.
    density (`float`):The fraction of values to preserve. Should be in [0,1].
    """
    mask = torch.zeros_like(tensor).view(-1)
    k = int(density * tensor.view(-1).shape[0])
    top_k = torch.topk(tensor.abs().view(-1), k=k, largest=True)
    mask[top_k[1]] = 1
    return tensor * mask.view(tensor.shape)


def random_pruning(tensor: torch.Tensor, density: float, rescale: bool) -> torch.Tensor:
    """
    Prune the smallest values of the task tensors and retain the top-k values based on the specified fraction `density`.

    Args:
    tensor (`torch.Tensor`):The tensor to prune.
    density (`float`):The fraction of values to preserve. Should be in [0,1].
    rescale (`bool`):Whether to rescale the result to preserve the expected value of the original tensor.
    """
    mask = torch.bernoulli(torch.full_like(input=tensor, fill_value=density))
    pruned_tensor = tensor * mask
    if rescale:
        torch.div(input=pruned_tensor, other=density)
    return pruned_tensor


def prune(
    tensor: torch.Tensor, density: float, method: Literal["magnitude", "random"], rescale: bool = False
) -> torch.Tensor:
    """
    Prune the values of task tensors based on the `method`.

    Args:
    tensor (`torch.Tensor`):The tensor to prune.
    density (`float`):The fraction of values to preserve. Should be in [0,1].
    method (`str`):The method to use to prune. Should be one of ["magnitude", "random"].
    rescale (`bool`):Whether to rescale the result to preserve the expected value of the original tensor.
    """
    if density >= 1:
        return tensor
    elif density < 0:
        raise ValueError("Density should be >= 0, got {density}")
    if method == "magnitude":
        return magnitude_based_pruning(tensor, density)
    elif method == "random":
        return random_pruning(tensor, density, rescale=rescale)
    else:
        raise ValueError(f"Unknown method {method}")


def calculate_majority_sign_mask(tensor: torch.Tensor, method: Literal["total", "frequency"] = "total"):
    """
    Get the mask of the majority sign across the task tensors. Task tensors are stacked on dimension 0.

    Args:
    tensor (`torch.Tensor`):The tensor to get the mask from.
    method (`str`):The method to use to get the mask. Should be one of ["total", "frequency"].
    """

    sign = tensor.sign()
    if method == "total":
        sign_magnitude = (sign * tensor.abs()).sum(dim=0)
    elif method == "frequency":
        sign_magnitude = sign.sum(dim=0)
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')
    majority_sign = torch.where(sign_magnitude >= 0, 1, -1)
    return sign == majority_sign


def disjoint_merge(task_tensors, majority_sign_mask):
    mixed_task_tensors = (task_tensors * majority_sign_mask).sum(dim=0)
    num_params_preserved = majority_sign_mask.sum(dim=0)
    return mixed_task_tensors / torch.clamp(num_params_preserved, min=1.0)


def task_arthimetic(task_tensors, weights):
    task_tensors = torch.stack(task_tensors, dim=0)
    # weighted task tensors
    while len(task_tensors.shape) > len(weights.shape):
        weights.unsqueeze_(-1)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = weighted_task_tensors.sum(dim=0)
    return mixed_task_tensors


def ties(task_tensors, weights, density, majority_sign_method="total"):
    # sparsify
    task_tensors = [prune(tensor, density, method="magnitude") for tensor in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    # weighted task tensors
    while len(task_tensors.shape) > len(weights.shape):
        weights.unsqueeze_(-1)
    weighted_task_tensors = task_tensors * weights
    # Elect Sign
    majority_sign_mask = calculate_majority_sign_mask(weighted_task_tensors, method=majority_sign_method)
    # Disjoint Merge
    mixed_task_tensors = disjoint_merge(weighted_task_tensors, majority_sign_mask)
    return mixed_task_tensors


def dare_linear(task_tensors, weights, density):
    # sparsify
    task_tensors = [prune(tensor, density, method="random", rescale=True) for tensor in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    # weighted task tensors
    while len(task_tensors.shape) > len(weights.shape):
        weights.unsqueeze_(-1)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = weighted_task_tensors.sum(dim=0)
    return mixed_task_tensors


def dare_ties(task_tensors, weights, density, majority_sign_method="total"):
    # sparsify
    task_tensors = [prune(tensor, density, method="random", rescale=True) for tensor in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    # weighted task tensors
    while len(task_tensors.shape) > len(weights.shape):
        weights.unsqueeze_(-1)
    weighted_task_tensors = task_tensors * weights
    # Elect Sign
    majority_sign_mask = calculate_majority_sign_mask(weighted_task_tensors, method=majority_sign_method)
    # Disjoint Merge
    mixed_task_tensors = disjoint_merge(weighted_task_tensors, majority_sign_mask)
    return mixed_task_tensors
