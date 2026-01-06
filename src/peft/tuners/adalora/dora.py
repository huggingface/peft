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

"""
AdaDoRA layer: DoRA adapted for AdaLoRA's SVD decomposition.

This module extends DoraLinearLayer to handle AdaLoRA's 3-factor decomposition
(A, E, B matrices) instead of standard LoRA's 2-factor decomposition (A, B).
"""

import torch
import torch.nn.functional as F
from torch import nn

from peft.tuners.lora.dora import DoraLinearLayer
from peft.utils.integrations import dequantize_module_weight, gather_params_ctx
from peft.utils.other import transpose


class AdaDoraLinearLayer(DoraLinearLayer):
    """
    DoRA layer adapted for AdaLoRA's SVD decomposition.

    Unlike standard LoRA where ΔW = B @ A, AdaLoRA uses ΔW = B @ (A * E) / ranknum.
    This class overrides the weight norm computation to account for the E matrix
    (singular values) in the decomposition.

    The key differences from DoraLinearLayer:
    1. get_weight_norm accepts lora_E and ranknum parameters
    2. update_layer initializes magnitude with SVD-aware weight norm
    3. forward computes output using SVD decomposition
    4. update_magnitude_after_pruning handles AdaLoRA's rank pruning
    """

    def get_weight_norm(
        self, weight, lora_A, lora_B, lora_E, scaling, ranknum
    ) -> torch.Tensor:
        """
        Compute L2 norm of (W + ΔW) for AdaLoRA's SVD decomposition.

        Args:
            weight: Base layer weight (out_features x in_features)
            lora_A: A matrix (r x in_features) - right singular vectors
            lora_B: B matrix (out_features x r) - left singular vectors
            lora_E: E matrix (r x 1) - singular values
            scaling: LoRA scaling factor (lora_alpha)
            ranknum: Current rank number for normalization

        Returns:
            Weight norm tensor of shape (out_features,)
        """
        # compute lora weight: B @ (A * E) * scaling / ranknum
        lora_weight = (lora_B @ (lora_A * lora_E)) * (scaling / (ranknum + 1e-5))

        # add to base weight and compute L2 norm per row
        weight = weight + lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    def update_layer(
        self, *, base_layer, lora_A, lora_B, lora_E, scaling, ranknum, place_on_cpu=False
    ) -> None:
        """
        Initialize magnitude vector for AdaLoRA.

        Overrides parent to accept lora_E and ranknum parameters for
        SVD-aware magnitude initialization.

        Args:
            base_layer: The base nn.Linear layer
            lora_A: A matrix parameter
            lora_B: B matrix parameter
            lora_E: E matrix parameter (singular values)
            scaling: LoRA scaling factor
            ranknum: Current rank number
            place_on_cpu: Whether to place magnitude on CPU
        """
        # temporarily convert fp16 to fp32 for numerical stability
        dtype_is_fp16 = lora_A.dtype == torch.float16
        if dtype_is_fp16:
            lora_A = lora_A.float()
            lora_B = lora_B.float()
            lora_E = lora_E.float()

        with gather_params_ctx(base_layer.parameters()):
            weight = dequantize_module_weight(base_layer)
            weight_norm = self.get_weight_norm(
                weight.to(lora_A.device),
                lora_A, lora_B, lora_E,
                scaling, ranknum
            )

        if dtype_is_fp16:
            weight_norm = weight_norm.half()

        if place_on_cpu:
            weight_norm = weight_norm.to("cpu")

        self.weight = nn.Parameter(weight_norm, requires_grad=True)

    def forward(
        self, x, *, lora_A, lora_B, lora_E, scaling, ranknum, base_layer, base_result=None
    ):
        """
        DoRA forward pass for AdaLoRA.

        Computes the DoRA output using AdaLoRA's SVD decomposition:
        output = (m / ||W + ΔW|| - 1) * base_result + (m / ||W + ΔW||) * lora_result

        Args:
            x: Input tensor
            lora_A: A matrix (r x in_features)
            lora_B: B matrix (out_features x r)
            lora_E: E matrix (r x 1) - singular values
            scaling: LoRA scaling factor
            ranknum: Current rank number
            base_layer: The base nn.Linear layer
            base_result: Pre-computed base layer output (optional, for efficiency)

        Returns:
            DoRA-adjusted output tensor
        """
        magnitude = self.weight
        weight = dequantize_module_weight(base_layer).to(x.dtype)

        # weight norm is detached per DoRA paper section 4.3:
        # "[...] we suggest treating ||V + ΔV||_c as a constant, thereby
        # detaching it from the gradient graph."
        weight_norm = self.get_weight_norm(
            weight, lora_A.detach(), lora_B.detach(), lora_E.detach(),
            scaling, ranknum
        )
        weight_norm = weight_norm.detach()
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)

        # compute adapter output: x @ (A*E).T @ B.T * scaling / ranknum
        lora_result = (x @ (lora_A * lora_E).T @ lora_B.T) * (scaling / (ranknum + 1e-5))

        # handle base result (subtract bias if present)
        if base_result is not None:
            bias = base_layer.bias
            if bias is not None:
                base_result = base_result - bias
        else:
            base_result = F.linear(x, transpose(weight, self.fan_in_fan_out))

        # DoRA formula: (m/||W+ΔW|| - 1) * base + m/||W+ΔW|| * Δ
        result_dora = (mag_norm_scale - 1) * base_result + mag_norm_scale * lora_result

        return result_dora

    def update_magnitude_after_pruning(
        self, base_layer, lora_A, lora_B, lora_E, scaling, ranknum
    ) -> None:
        """
        Hook called after AdaLoRA rank pruning.

        In DoRA, the magnitude vector should evolve via gradients and diverge
        from the weight norm ||W + ΔW||. This divergence is what distinguishes
        DoRA from standard LoRA - the ratio m/||W+ΔW|| controls output scaling.

        Previously, this method reset magnitude = ||W + ΔW||, which made the
        ratio equal to 1 and effectively disabled DoRA (identical to AdaLoRA).

        Now we let the magnitude remain at its learned value and allow gradients
        to naturally adjust it after pruning changes the effective rank.

        Args:
            base_layer: The base nn.Linear layer (unused)
            lora_A: A matrix parameter (unused)
            lora_B: B matrix parameter (unused)
            lora_E: E matrix parameter (unused)
            scaling: LoRA scaling factor (unused)
            ranknum: Current rank number (unused)
        """
        # don't reset magnitude - let gradients handle adaptation
        pass

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "adalora.dora." + rep
