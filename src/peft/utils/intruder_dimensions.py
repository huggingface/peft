# Copyright 2023-present the HuggingFace Inc. team.
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

"""
Intruder dimension detection and mitigation for LoRA adapters.

Based on: "LoRA vs Full Fine-tuning: An Illusion of Equivalence"
https://arxiv.org/abs/2410.21228

Reference implementation: https://github.com/reeceshuttle/intruder-dimensions
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class IntruderDetectionResult:
    """Container for intruder dimension detection results.

    Attributes:
        intruder_indices (`torch.Tensor`):
            Indices of detected intruder dimensions within top_k singular vectors.
        left_vectors (`torch.Tensor`):
            Left singular vectors (U) corresponding to intruder dimensions.
            Shape: (out_features, num_intruders)
        singular_values (`torch.Tensor`):
            Singular values corresponding to intruder dimensions.
            Shape: (num_intruders,)
        right_vectors (`torch.Tensor`):
            Rows of Vᵀ corresponding to intruder dimensions.
            Shape: (num_intruders, in_features)
        similarity_matrix (`torch.Tensor`):
            Full cosine similarity matrix between fine-tuned and pre-trained left singular vectors.
            Shape: (top_k, min(out_features, in_features))
        max_similarities (`torch.Tensor`):
            Maximum similarity for each fine-tuned top-k singular vector.
            Shape: (top_k,)
        layer_name (`str`, *optional*):
            Name of the layer where intruders were detected.
    """

    intruder_indices: torch.Tensor
    left_vectors: torch.Tensor
    singular_values: torch.Tensor
    right_vectors: torch.Tensor
    similarity_matrix: torch.Tensor
    max_similarities: torch.Tensor
    layer_name: Optional[str] = None

    def __post_init__(self):
        """Validate tensor shapes are consistent."""
        num_intruders = len(self.intruder_indices)
        if num_intruders > 0:
            assert self.left_vectors.shape[1] == num_intruders, "left_vectors dim 1 must match num_intruders"
            assert len(self.singular_values) == num_intruders, "singular_values length must match num_intruders"
            assert self.right_vectors.shape[0] == num_intruders, "right_vectors dim 0 must match num_intruders"


def detect_intruder_dimensions(
    w_pretrained: torch.Tensor,
    w_finetuned: torch.Tensor,
    top_k: int = 10,
    epsilon: float = 0.5,
    device: Union[str, torch.device, None] = None,
) -> IntruderDetectionResult:
    """
    Detect intruder dimensions by comparing SVD of pre-trained and fine-tuned weights.

    An intruder dimension is a singular vector in the fine-tuned weights that is dissimilar
    to all singular vectors in the pre-trained weights. These dimensions cause catastrophic
    forgetting without contributing to task performance.

    Args:
        w_pretrained (`torch.Tensor`):
            Pre-trained weight matrix W₀, shape (out_features, in_features).
        w_finetuned (`torch.Tensor`):
            Fine-tuned weight matrix W, shape (out_features, in_features).
        top_k (`int`, *optional*, defaults to `10`):
            Number of highest-ranking singular vectors to examine.
        epsilon (`float`, *optional*, defaults to `0.5`):
            Cosine similarity threshold (< epsilon => intruder).
        device (`str`, `torch.device`, or `None`, *optional*):
            Device to perform computation on. If `None`, uses the device of `w_pretrained`.

    Returns:
        `IntruderDetectionResult`: Detection results containing intruder indices and components.

    Raises:
        `AssertionError`: If weight matrices have different shapes.
        `AssertionError`: If epsilon is not in [0, 1].
        `AssertionError`: If top_k is not positive.

    Example:
        ```python
        >>> import torch
        >>> from peft.utils.intruder_dimensions import detect_intruder_dimensions
        >>>
        >>> w0 = torch.randn(100, 50)  # Pre-trained weights
        >>> wt = w0 + torch.randn(100, 50) * 0.1  # Fine-tuned weights
        >>>
        >>> results = detect_intruder_dimensions(w0, wt, top_k=10, epsilon=0.5)
        >>> print(f"Found {len(results.intruder_indices)} intruders")
        ```
    """
    assert w_pretrained.shape == w_finetuned.shape, (
        f"Weight matrices must have same shape, got {w_pretrained.shape} and {w_finetuned.shape}"
    )
    assert 0 <= epsilon <= 1, f"Epsilon must be in [0, 1], got {epsilon}"
    assert top_k > 0, f"top_k must be positive, got {top_k}"

    if device is None:
        device = w_pretrained.device

    w0 = w_pretrained.detach().to(device, dtype=torch.float32)
    wt = w_finetuned.detach().to(device, dtype=torch.float32)

    u0, s0, v0 = torch.linalg.svd(w0, full_matrices=False)
    ut, st, vt = torch.linalg.svd(wt, full_matrices=False)

    ut_top = ut[:, :top_k]  # Shape: (d_out, top_k)

    u0_normalized = u0 / (torch.norm(u0, dim=0, keepdim=True) + 1e-12)
    ut_top_normalized = ut_top / (torch.norm(ut_top, dim=0, keepdim=True) + 1e-12)

    similarity_matrix = torch.abs(ut_top_normalized.T @ u0_normalized)

    max_similarities, _ = torch.max(similarity_matrix, dim=1)
    intruder_mask = max_similarities < epsilon

    intruder_indices = torch.where(intruder_mask)[0]

    if len(intruder_indices) > 0:
        left_vecs = ut_top[:, intruder_indices]  # U[:, intruder_idx]
        singular_vals = st[intruder_indices]  # S[intruder_idx]
        right_vecs = vt[intruder_indices, :]
    else:
        left_vecs = torch.empty((ut.shape[0], 0), device=device)
        singular_vals = torch.empty(0, device=device)
        right_vecs = torch.empty((0, vt.shape[1]), device=device)

    return IntruderDetectionResult(
        intruder_indices=intruder_indices.cpu(),
        left_vectors=left_vecs.cpu(),
        singular_values=singular_vals.cpu(),
        right_vectors=right_vecs.cpu(),
        similarity_matrix=similarity_matrix.cpu(),
        max_similarities=max_similarities.cpu(),
    )


def mitigate_intruder_dimensions(
    w_pretrained: torch.Tensor,
    delta_w: torch.Tensor,
    intruder_results: Union[IntruderDetectionResult, dict[str, torch.Tensor]],
    lambda_factor: float = 0.75,
) -> torch.Tensor:
    """
    Mitigate intruder dimensions by scaling their contributions.

    Applies the mitigation formula: W₀ + ΔW + (λ - 1) * Σ uᵢσᵢvᵢᵀ
    where uᵢσᵢvᵢᵀ is the rank-1 intruder component.

    Args:
        w_pretrained (`torch.Tensor`):
            Pre-trained weight matrix (W₀), shape (out_features, in_features).
        delta_w (`torch.Tensor`):
            The LoRA delta weight (ΔW), shape (out_features, in_features).
        intruder_results (`IntruderDetectionResult` or `Dict[str, torch.Tensor]`):
            Detection results from `detect_intruder_dimensions`.
        lambda_factor (`float`, *optional*, defaults to `0.75`):
            Mitigation strength (1.0 = no mitigation, 0.0 = complete removal).

    Returns:
        `torch.Tensor`: The mitigated full weight matrix (W₀ + ΔW_mitigated).

    Raises:
        `ValueError`: If shapes don't match or lambda_factor is invalid.

    Example:
        ```python
        >>> import torch
        >>> from peft.utils.intruder_dimensions import detect_intruder_dimensions, mitigate_intruder_dimensions
        >>>
        >>> w0 = torch.randn(100, 50)
        >>> delta_w = torch.randn(100, 50) * 0.1
        >>> wt = w0 + delta_w
        >>>
        >>> results = detect_intruder_dimensions(w0, wt)
        >>> w_mitigated = mitigate_intruder_dimensions(w0, delta_w, results, lambda_factor=0.75)
        ```
    """
    if w_pretrained.shape != delta_w.shape:
        raise ValueError(f"Shape mismatch: w_pretrained {w_pretrained.shape} vs delta_w {delta_w.shape}")

    if not (0.0 <= lambda_factor <= 2.0):
        raise ValueError(f"lambda_factor must be in [0, 2], got {lambda_factor}")

    if isinstance(intruder_results, IntruderDetectionResult):
        left_vecs = intruder_results.left_vectors
        singular_vals = intruder_results.singular_values
        right_vecs = intruder_results.right_vectors
    else:
        required_keys = ["left_vectors", "singular_values", "right_vectors"]
        for key in required_keys:
            if key not in intruder_results:
                raise ValueError(f"intruder_results missing required key: {key}")
        left_vecs = intruder_results["left_vectors"]
        singular_vals = intruder_results["singular_values"]
        right_vecs = intruder_results["right_vectors"]

    if len(singular_vals) == 0:
        return delta_w

    device = w_pretrained.device
    target_dtype = torch.float32

    left_vecs = left_vecs.to(device=device, dtype=target_dtype)
    singular_vals = singular_vals.to(device=device, dtype=target_dtype)
    right_vecs = right_vecs.to(device=device, dtype=target_dtype)

    w0_float = w_pretrained.to(dtype=target_dtype)
    delta_w_float = delta_w.to(dtype=target_dtype)

    # Compute intruder component: Σ uᵢσᵢvᵢᵀ
    # Shapes: (d_out, num) @ (num, num) @ (num, d_in) -> (d_out, d_in)
    sigma_diag = torch.diag(singular_vals)
    intruder_component = left_vecs @ sigma_diag @ right_vecs

    scaling_factor = lambda_factor - 1.0
    delta_w_mitigated = delta_w_float + scaling_factor * intruder_component

    return delta_w_mitigated.to(dtype=w_pretrained.dtype)


def project_delta_to_lora(
    delta_w: torch.Tensor,
    rank: int,
    scaling: float,
    preserve_rank: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project a modified delta weight back onto LoRA A and B matrices using SVD.

    This enables non-destructive mitigation by keeping the LoRA format intact.

    Args:
        delta_w (`torch.Tensor`):
            Modified weight update of shape (out_features, in_features).
        rank (`int`):
            The target rank for LoRA matrices.
        scaling (`float`):
            The scaling factor (alpha/r) used by LoRA.
        preserve_rank (`bool`, *optional*, defaults to `True`):
            If `True`, truncates SVD to the original rank.
            If `False`, uses the effective rank of the modified matrix.

    Returns:
        `Tuple[torch.Tensor, torch.Tensor]`: (lora_A, lora_B) matrices where:
            - lora_A shape: (rank, in_features)
            - lora_B shape: (out_features, rank)

    Example:
        ```python
        >>> import torch
        >>> from peft.utils.intruder_dimensions import project_delta_to_lora
        >>>
        >>> delta_w_modified = torch.randn(100, 50)  # Modified delta weight
        >>> lora_A, lora_B = project_delta_to_lora(delta_w_modified, rank=8, scaling=16.0/8)
        >>> reconstructed = lora_B @ lora_A * (16.0/8)
        ```
    """
    target_dtype = delta_w.dtype if delta_w.dtype in (torch.float32, torch.float64) else torch.float32
    delta_w = delta_w.to(dtype=target_dtype)

    u, s, v = torch.linalg.svd(delta_w, full_matrices=False)  # v is Vᵀ

    if preserve_rank:
        effective_rank = min(rank, len(s))
        s_trunc = torch.diag(s[:effective_rank])
        lora_B = u[:, :effective_rank] @ s_trunc
        lora_A = v[:effective_rank, :] / scaling  # v is Vᵀ, so v[:rank, :] gives (rank, d_in)
    else:
        effective_rank = (s > 1e-6).sum().item()
        effective_rank = min(effective_rank, len(s))
        s_trunc = torch.diag(s[:effective_rank])
        lora_B = u[:, :effective_rank] @ s_trunc
        lora_A = v[:effective_rank, :] / scaling

    lora_A = lora_A.to(dtype=delta_w.dtype, device=delta_w.device)
    lora_B = lora_B.to(dtype=delta_w.dtype, device=delta_w.device)

    return lora_A, lora_B


def compute_reconstruction_error(
    original_matrix: torch.Tensor,
    reconstructed_matrix: torch.Tensor,
    metric: str = "relative_frobenius",
) -> float:
    """
    Compute the reconstruction error between two matrices.

    Args:
        original_matrix (`torch.Tensor`):
            The original matrix.
        reconstructed_matrix (`torch.Tensor`):
            The reconstructed matrix.
        metric (`str`, *optional*, defaults to `"relative_frobenius"`):
            The error metric to use: 'frobenius', 'relative_frobenius', or 'max_abs'.

    Returns:
        `float`: The computed error.

    Raises:
        `ValueError`: If the metric is unknown.

    Example:
        ```python
        >>> import torch
        >>> from peft.utils.intruder_dimensions import compute_reconstruction_error
        >>>
        >>> original = torch.randn(100, 50)
        >>> reconstructed = original + torch.randn(100, 50) * 0.01
        >>> error = compute_reconstruction_error(original, reconstructed, metric="relative_frobenius")
        >>> print(f"Relative error: {error:.4f}")
        ```
    """
    if original_matrix.shape != reconstructed_matrix.shape:
        raise ValueError("Original and reconstructed matrices must have the same shape")

    diff = original_matrix - reconstructed_matrix

    if metric == "frobenius":
        return torch.norm(diff, p="fro").item()
    elif metric == "relative_frobenius":
        denom = torch.norm(original_matrix, p="fro")
        if denom < 1e-12:
            return float("inf")
        return (torch.norm(diff, p="fro") / denom).item()
    elif metric == "max_abs":
        return torch.max(torch.abs(diff)).item()
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'frobenius', 'relative_frobenius', or 'max_abs'.")
