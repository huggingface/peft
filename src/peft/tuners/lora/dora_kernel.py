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

"""DoRA kernel implementation.

This module implements a memory-efficient DoRA forward pass that avoids materializing the full ``lora_B @ lora_A``
matrix. Instead of computing the column-wise weight norm ``||W + s*BA||_c`` on the full ``out x in`` matrix, it
decomposes the squared norm using the low-rank structure of ``BA = B @ A``::

    ||W + s*BA||²_c = ||W||²_c + 2*s*<W, BA>_c + s²*||BA||²_c

where::

    <W, BA>_c = (W @ Aᵀ * B).sum(dim=1) # O(out*r) peak memory ||BA||²_c = (B @ (A@Aᵀ) * B).sum(dim=1) # O(r² + out*r)
    peak memory

This reduces peak memory from O(out*in) to O(out*r + r²) and can also reduce runtime by avoiding the full matmul.

Limitations (first version):

- Only ``nn.Linear`` base layers are supported (no conv / embedding).
- ``fan_in_fan_out=True`` is not supported (raise error).
- When ``lora_dropout > 0`` and the model is in training mode, the kernel falls back to computing the full ``BA``
  matrix (same as the reference DoRA path).
- ``base_result`` must be provided for the low-rank path (i.e. dropout is ``nn.Identity`` or the model is in eval mode
  -- the normal DoRA fast path).
"""

from copy import deepcopy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from peft.utils.integrations import dequantize_module_weight, gather_params_ctx


class DoraKernelLinearLayer(nn.Module):
    """DoRA layer using the low-rank decomposition of the weight norm.

    See module docstring for details.
    """

    def __init__(self, fan_in_fan_out: bool):
        super().__init__()
        self.fan_in_fan_out = fan_in_fan_out

    # ------------------------------------------------------------------
    # weight norm computation (low-rank decomposition)
    # ------------------------------------------------------------------
    @staticmethod
    def get_weight_norm_lowrank(
        weight: torch.Tensor,
        lora_A_weight: torch.Tensor,
        lora_B_weight: torch.Tensor,
        scaling: float,
    ) -> torch.Tensor:
        """Compute ``||W + s*BA||_c`` without materializing BA.

        Args:
            weight: base weight, shape ``(out, in)``.
            lora_A_weight: shape ``(r, in)``.
            lora_B_weight: shape ``(out, r)``.
            scaling: LoRA scaling factor ``alpha / r``.

        Returns:
            ``weight_norm`` of shape ``(out,)`` — the L2 norm of each row of ``W + s*BA``.
        """
        # ||W||²_c  -- shape (out,)
        w_norm_sq = (weight * weight).sum(dim=1)

        # <W, BA>_c = (W @ Aᵀ * B).sum(dim=1)
        wa = weight @ lora_A_weight.t()  # (out, r)
        cross = (wa * lora_B_weight).sum(dim=1)  # (out,)

        # ||BA||²_c = (B @ (A@Aᵀ) * B).sum(dim=1)
        aa = lora_A_weight @ lora_A_weight.t()  # (r, r)
        ba_sq = (lora_B_weight @ aa * lora_B_weight).sum(dim=1)  # (out,)

        norm_sq = w_norm_sq + 2.0 * scaling * cross + scaling * scaling * ba_sq
        # clamp to avoid sqrt of tiny negative due to float rounding
        norm_sq = norm_sq.clamp(min=0.0)
        return torch.sqrt(norm_sq)

    def get_weight_norm(self, weight, lora_weight, scaling, adapter_name: Optional[str] = None):
        """Compute ``||W + s*lora_weight||_c`` for merge/unmerge compatibility.

        During merge, ``lora_weight`` is the full delta weight (already materialized as ``B@A``). We compute the
        column-wise norm directly.
        """
        weight = weight + scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    # ------------------------------------------------------------------
    # update_layer (called at init time, same interface as DoraLinearLayer)
    # ------------------------------------------------------------------
    def update_layer(self, *, base_layer, lora_A, lora_B, scaling, place_on_cpu=False) -> None:
        # temporarily convert fp16 to fp32, as fp16 can cause trouble on CPU with PyTorch < 2.2
        dtype_is_fp16 = lora_A.dtype == torch.float16
        if dtype_is_fp16:
            lora_A = lora_A.float()
            lora_B = lora_B.float()

        with gather_params_ctx(base_layer.parameters()):
            if base_layer.__class__.__name__ == "Linear4bit":
                base_layer = deepcopy(base_layer)

            weight = dequantize_module_weight(base_layer)

            if dtype_is_fp16:
                lora_A = lora_A.half()
                lora_B = lora_B.half()

            # Compute weight norm using low-rank decomposition (float32 for stability)
            weight_f = weight.to(lora_A.device).to(torch.float32)
            a_f = lora_A.to(torch.float32)
            b_f = lora_B.to(torch.float32)
            weight_norm = self.get_weight_norm_lowrank(weight_f, a_f, b_f, float(scaling))
            weight_norm = weight_norm.to(weight.dtype)

        if place_on_cpu:
            weight_norm = weight_norm.to("cpu")
        self.weight = nn.Parameter(weight_norm, requires_grad=True)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x, *, lora_A, lora_B, scaling, base_layer, base_result=None, adapter_name="default"):
        """DoRA forward using low-rank weight norm.

        Requires ``base_result`` to be provided (the base layer output without bias). If ``base_result`` is ``None``
        (training mode with dropout), we fall back to the standard DoRA path that materializes BA.
        """
        magnitude = self.weight  # (out,)

        if base_result is None:
            # Training mode with dropout: fall back to full materialization.
            return self._forward_full(x, lora_A, lora_B, scaling, base_layer, magnitude)

        # Low-rank kernel path (eval mode or no dropout)
        lora_A_weight = lora_A.weight  # (r, in)
        lora_B_weight = lora_B.weight  # (out, r)

        weight = dequantize_module_weight(base_layer)
        weight = weight.to(x.dtype)

        # Compute weight norm without materializing BA (float32 for stability)
        with torch.no_grad():
            weight_f = weight.to(torch.float32)
            a_f = lora_A_weight.to(torch.float32)
            b_f = lora_B_weight.to(torch.float32)
            weight_norm = self.get_weight_norm_lowrank(weight_f, a_f, b_f, float(scaling))
        weight_norm = weight_norm.detach().to(x.dtype)

        mag_norm_scale = (magnitude / weight_norm).view(1, -1)  # (1, out)

        lora_result = lora_B(lora_A(x))  # (batch, out)

        # Remove bias from base_result if present, so the DoRA scaling only
        # applies to the directional component
        bias = base_layer.bias
        if bias is not None:
            base_result = base_result - bias

        result_dora = (mag_norm_scale - 1) * base_result + mag_norm_scale * lora_result * scaling
        return result_dora

    def _forward_full(self, x, lora_A, lora_B, scaling, base_layer, magnitude):
        """Fallback: materializes full BA (same as DoraLinearLayer.forward)."""
        lora_weight = lora_B(
            lora_A(torch.eye(lora_A.weight.shape[1], device=lora_A.weight.device, dtype=lora_A.weight.dtype))
        ).T
        lora_weight = lora_weight.to(x.dtype)

        weight = dequantize_module_weight(base_layer)
        weight = weight.to(x.dtype)
        weight_norm = self.get_weight_norm_lowrank(
            weight.to(torch.float32),
            lora_A.weight.to(torch.float32),
            lora_B.weight.to(torch.float32),
            float(scaling),
        )
        weight_norm = weight_norm.detach().to(x.dtype)
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)

        base_result = F.linear(x, weight)

        result_dora = (mag_norm_scale - 1) * base_result + mag_norm_scale * lora_B(lora_A(x)) * scaling
        return result_dora

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.dora_kernel." + rep
