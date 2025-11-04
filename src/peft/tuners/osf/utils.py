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
"""Utilities for Orthogonal Subspace Learning with Adaptive OSF."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import nn


# Note: OSF now relies on OSFLayer + BaseTuner; no model-level helpers required here.


__all__ = [
    "decompose_weight_matrix",
    "project_gradient_to_orthogonal_space",
    "reconstruct_weight_matrix",
]


def _wait_if_async(tensor):
    """Wait for AsyncCollectiveTensor if needed, otherwise return tensor as-is."""
    if hasattr(tensor, "wait"):
        return tensor.wait()
    return tensor


def decompose_weight_matrix(weight: torch.Tensor, top_k: int) -> dict[str, Any]:
    """Perform an SVD of ``weight`` and split it into frozen and trainable parts."""
    device_local = weight.device
    orig_dtype = weight.dtype
    W = weight.to(torch.float32)
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = min(top_k, S.shape[0])

    svd = {
        "U_high": U[:, :k].contiguous().detach().to(device=device_local, dtype=orig_dtype),
        "S_high": S[:k].contiguous().detach().to(device=device_local, dtype=orig_dtype),
        "V_high": Vt[:k, :].contiguous().detach().to(device=device_local, dtype=orig_dtype),
        "U_low": nn.Parameter(U[:, k:].contiguous().detach().to(device=device_local, dtype=orig_dtype)),
        "S_low": nn.Parameter(S[k:].contiguous().detach().to(device=device_local, dtype=orig_dtype)),
        "V_low": nn.Parameter(Vt[k:, :].contiguous().detach().to(device=device_local, dtype=orig_dtype)),
        "rank_high": k,
    }
    return svd


def reconstruct_weight_matrix(svd_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Reconstruct a weight matrix from its SVD components."""
    U_high = svd_dict["U_high"]
    S_high = svd_dict["S_high"]
    V_high = svd_dict["V_high"]
    U_low = svd_dict["U_low"]
    S_low = svd_dict["S_low"]
    V_low = svd_dict["V_low"]

    high_part = (
        torch.mm(U_high * S_high.unsqueeze(0), V_high)
        if U_high.numel() > 0 and S_high.numel() > 0
        else torch.zeros(U_low.size(0), V_low.size(1), device=U_high.device)
    )
    low_part = (
        torch.mm(U_low * S_low.unsqueeze(0), V_low)
        if U_low.numel() > 0 and S_low.numel() > 0
        else torch.zeros(U_high.size(0), V_high.size(1), device=U_low.device)
    )
    return high_part + low_part


def project_gradient_to_orthogonal_space(svd_dict: dict[str, Any]) -> None:
    """Project gradients of ``U_low`` and ``V_low`` to be orthogonal to the high rank space."""
    if svd_dict["U_low"].grad is None and svd_dict["S_low"].grad is None and svd_dict["V_low"].grad is None:
        return

    U_high = svd_dict["U_high"]
    V_high = svd_dict["V_high"]

    # Project U_low gradients to space orthogonal to U_high
    if svd_dict["U_low"].grad is not None:
        dU = svd_dict["U_low"].grad
        # Support distributed tensors by operating on the local shard
        local_U_high = getattr(U_high, "to_local", lambda: U_high)()
        local_dU = getattr(dU, "to_local", lambda: dU)()

        # Perform projection computation using memory-efficient operations
        # Memory-optimized projection: dU = dU - U_high @ (U_high.T @ dU)
        # Use addmm_ for efficient in-place operation
        # Compute local contribution to (U_high^T @ dU); all-reduce to get global projection
        proj_coeff = torch.mm(local_U_high.transpose(0, 1), local_dU)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(proj_coeff, op=dist.ReduceOp.SUM)
        # Apply projection using only local rows of U_high
        local_dU.addmm_(local_U_high, proj_coeff, alpha=-1.0)

        if hasattr(dU, "_local_tensor"):
            dU._local_tensor.copy_(local_dU)
        else:
            dU.copy_(local_dU)

    # Repeat projection for V_low using V_high
    if svd_dict["V_low"].grad is not None:
        dV = svd_dict["V_low"].grad
        local_V_high = getattr(V_high, "to_local", lambda: V_high)()
        local_dV = getattr(dV, "to_local", lambda: dV)()

        # Compute Gram matrix G = V_high^T @ V_high for global projection across row-sharded V_high
        # Assumes column dimension is consistent across ranks (row sharding over singular vectors)
        G_local = torch.mm(local_V_high.transpose(0, 1), local_V_high)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(G_local, op=dist.ReduceOp.SUM)

        # Apply projection: dV = dV - dV @ G (use local shard of dV)
        update = torch.mm(local_dV, G_local)
        local_dV.add_(update, alpha=-1.0)

        if hasattr(dV, "_local_tensor"):
            dV._local_tensor.copy_(local_dV)
        else:
            dV.copy_(local_dV)
