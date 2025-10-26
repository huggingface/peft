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

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from peft.utils.integrations import dequantize_module_weight, gather_params_ctx
from peft.utils.other import transpose


class WoraLinearLayer(nn.Module):
    """
    WoRA (Weighted-Direction Low-Rank Adaptation) layer for linear transformations.

    WoRA extends DoRA by adding learnable scalar parameters (alpha, beta) that weight
    the base weights and low-rank update before normalization, allowing the model to
    learn the optimal trade-off between pretrained knowledge and task-specific adaptations.

    Args:
        fan_in_fan_out (bool): Whether the layer uses fan_in_fan_out weight layout.
    """
    def __init__(self, fan_in_fan_out):
        super().__init__()
        self.fan_in_fan_out = fan_in_fan_out

    def get_weight_norm(self, weight, lora_weight, scaling, alpha, beta) -> torch.Tensor:
        """
        Calculate L2 norm of the weighted weight matrix, column-wise.

        This is the core innovation of WoRA: weighted combination before normalization.
        Formula: norm(beta * W_0 + alpha * scaling * BA)

        Args:
            weight: Base weight matrix W_0
            lora_weight: Low-rank weight BA
            scaling: LoRA scaling factor
            alpha: Weight for the low-rank update
            beta: Weight for the base weight

        Returns:
            Column-wise L2 norms of the weighted weight matrix
        """
        # calculate L2 norm of weighted weight matrix, column-wise
        weight = transpose(weight, self.fan_in_fan_out)
        # WoRA's key innovation: weighted combination before normalization
        weight = beta * weight + alpha * scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    def update_layer(self, *, base_layer, lora_A, lora_B, scaling, alpha, beta, place_on_cpu=False) -> None:
        """
        Initialize the WoRA layer by computing the initial weight norms.

        Args:
            base_layer: The base layer to adapt
            lora_A: LoRA A matrix weights
            lora_B: LoRA B matrix weights
            scaling: LoRA scaling factor
            alpha: Alpha parameter (weight for low-rank update)
            beta: Beta parameter (weight for base weight)
            place_on_cpu: Whether to place the magnitude vector on CPU for memory efficiency
        """
        # temporarily convert fp16 to fp32, as fp16 can cause trouble on CPU with PyTorch < 2.2
        dtype_is_fp16 = lora_A.dtype == torch.float16
        if dtype_is_fp16:
            lora_A = lora_A.float()
            lora_B = lora_B.float()

        with gather_params_ctx(base_layer.parameters()):
            if base_layer.__class__.__name__ == "Linear4bit":
                # We have to create a copy of the base layer, otherwise, FSDP will throw an error. 8bit does not work
                # yet because Int8Params cannot be correctly deep-copied (attributes vanish)
                base_layer = deepcopy(base_layer)

            weight = dequantize_module_weight(base_layer)
            if weight.data.ndim >= 3:  # For handling LoRAs applied to Conv layers.
                r = lora_A.shape[0]
                lora_weight = torch.mm(lora_B.view([-1, r]), lora_A.view([r, -1]))
                lora_weight = lora_weight.reshape(weight.shape)
            else:
                lora_weight = lora_B @ lora_A

            if dtype_is_fp16:
                lora_weight = lora_weight.half()

            # Convert alpha and beta to tensors if they're Parameters
            alpha_val = alpha.data if isinstance(alpha, nn.Parameter) else alpha
            beta_val = beta.data if isinstance(beta, nn.Parameter) else beta

            weight_norm = self.get_weight_norm(weight.to(lora_A.device), lora_weight, scaling, alpha_val, beta_val)

        if place_on_cpu:
            weight_norm = weight_norm.to("cpu")
        self.weight = nn.Parameter(weight_norm, requires_grad=True)

    def forward(self, x, *, lora_A, lora_B, scaling, alpha, beta, base_layer, base_result=None):
        """
        Forward pass for WoRA layer.

        Computes the output with weighted direction and magnitude scaling:
        output = m * (beta * W_0 + alpha * scaling * BA) / ||beta * W_0 + alpha * scaling * BA||

        Args:
            x: Input tensor
            lora_A: LoRA A layer
            lora_B: LoRA B layer
            scaling: LoRA scaling factor
            alpha: Alpha parameter
            beta: Beta parameter
            base_layer: The base layer
            base_result: Pre-computed base layer output (optional, for efficiency)

        Returns:
            WoRA output to be added to the base result
        """
        # Don't use `lora_weight = lora_B.weight @ lora_A.weight` because this causes errors with FSDP. Instead,
        # calculate the same but using forward.
        x_eye = torch.eye(lora_A.weight.shape[1], device=lora_A.weight.device, dtype=x.dtype)
        lora_weight = lora_B(lora_A(x_eye)).T

        magnitude = self.weight
        weight = dequantize_module_weight(base_layer)
        weight = weight.to(x.dtype)

        # Convert alpha and beta to scalars for computation
        alpha_val = alpha.item() if isinstance(alpha, (torch.Tensor, nn.Parameter)) else alpha
        beta_val = beta.item() if isinstance(beta, (torch.Tensor, nn.Parameter)) else beta

        weight_norm = self.get_weight_norm(weight, lora_weight.detach(), scaling, alpha_val, beta_val)
        # see section 4.3 of DoRA (https://huggingface.co/papers/2402.09353)
        # "[...] we suggest treating ||V +∆V ||_c in
        # Eq. (5) as a constant, thereby detaching it from the gradient
        # graph. This means that while ||V + ∆V ||_c dynamically
        # reflects the updates of ∆V , it won't receive any gradient
        # during backpropagation"
        weight_norm = weight_norm.detach()
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)

        lora_result = lora_B(lora_A(x))

        bias = None
        if base_result is not None:
            bias = base_layer.bias
            if bias is not None:
                base_result = base_result - bias
        else:
            base_result = F.linear(x, transpose(weight, self.fan_in_fan_out))

        # WoRA applies the weighted scaling: (beta - 1) for base + alpha for LoRA
        result_wora = (beta_val * mag_norm_scale - 1) * base_result + alpha_val * mag_norm_scale * lora_result * scaling

        return result_wora

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.wora." + rep


class WoraEmbeddingLayer(WoraLinearLayer):
    """WoRA layer for embedding transformations."""

    def forward(self, x, *, lora_A, lora_B, scaling, alpha, beta, base_layer, embed_fn):
        """
        Forward pass for WoRA embedding layer.

        Args:
            x: Input tensor (token indices)
            lora_A: LoRA A matrix
            lora_B: LoRA B matrix
            scaling: LoRA scaling factor
            alpha: Alpha parameter
            beta: Beta parameter
            base_layer: The base embedding layer
            embed_fn: Embedding function

        Returns:
            Tuple of (mag_norm_scale, result_wora)
        """
        lora_weight = (lora_A @ lora_B).T
        magnitude = self.weight
        weight = base_layer.weight

        # Convert alpha and beta to scalars
        alpha_val = alpha.item() if isinstance(alpha, (torch.Tensor, nn.Parameter)) else alpha
        beta_val = beta.item() if isinstance(beta, (torch.Tensor, nn.Parameter)) else beta

        weight_norm = self.get_weight_norm(weight, lora_weight.detach(), scaling, alpha_val, beta_val)
        # see section 4.3 of DoRA (https://huggingface.co/papers/2402.09353)
        weight_norm = weight_norm.detach()
        mag_norm_scale = magnitude / weight_norm

        # WoRA weighted scaling
        result_wora = beta_val * mag_norm_scale * (embed_fn(x, lora_A) @ lora_B) * scaling
        return mag_norm_scale, result_wora

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.wora." + rep


class _WoraConvNdLayer(WoraLinearLayer):
    """Base WoRA layer for convolutional transformations."""

    def get_weight_norm(self, weight, lora_weight, scaling, alpha, beta) -> torch.Tensor:
        """Calculate L2 norm for convolutional weights."""
        # calculate L2 norm of weighted weight matrix, column-wise
        # WoRA weighted combination
        weight = beta * weight + alpha * scaling * lora_weight
        # the following is needed to have compatibility with the 4/5D weight tensors of Conv2D/3D
        dim = tuple(range(1, weight.dim()))
        weight_norm = weight.norm(p=2, dim=dim, keepdim=True).transpose(1, 0)
        return weight_norm

    def forward(self, x, *, lora_A, lora_B, scaling, alpha, beta, base_layer, base_result=None):
        """Forward pass for WoRA convolutional layer."""
        weight = base_layer.weight
        r = lora_A.weight.shape[0]
        lora_weight = torch.mm(lora_B.weight.view([-1, r]), lora_A.weight.view([r, -1]))
        lora_weight = lora_weight.reshape(weight.shape)
        magnitude = self.weight

        # Convert alpha and beta to scalars
        alpha_val = alpha.item() if isinstance(alpha, (torch.Tensor, nn.Parameter)) else alpha
        beta_val = beta.item() if isinstance(beta, (torch.Tensor, nn.Parameter)) else beta

        weight_norm = self.get_weight_norm(weight, lora_weight.detach(), scaling, alpha_val, beta_val)
        # see section 4.3 of DoRA (https://huggingface.co/papers/2402.09353)
        weight_norm = weight_norm.detach()
        mag_norm_scale = magnitude / weight_norm

        if base_result is None:
            base_result = self.conv_fn(
                x,
                weight,
                bias=None,
                stride=base_layer.stride,
                padding=base_layer.padding,
                dilation=base_layer.dilation,
                groups=base_layer.groups,
            )
        else:
            bias = base_layer.bias
            if bias is not None:
                # reshape bias to (1, -1, 1, ...)
                bias_shape = (1, -1) + (1,) * (base_result.dim() - 2)
                base_result = base_result - bias.view(*bias_shape)

        # WoRA weighted scaling
        result_wora = (beta_val * mag_norm_scale - 1) * base_result + alpha_val * mag_norm_scale * lora_B(lora_A(x)) * scaling
        return result_wora

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.wora." + rep


class WoraConv1dLayer(_WoraConvNdLayer):
    """WoRA layer for 1D convolutions."""
    def __init__(self, fan_in_fan_out):
        super().__init__(fan_in_fan_out)
        self.conv_fn = F.conv1d


class WoraConv2dLayer(_WoraConvNdLayer):
    """WoRA layer for 2D convolutions."""
    def __init__(self, fan_in_fan_out):
        super().__init__(fan_in_fan_out)
        self.conv_fn = F.conv2d


class WoraConv3dLayer(_WoraConvNdLayer):
    """WoRA layer for 3D convolutions."""
    def __init__(self, fan_in_fan_out):
        super().__init__(fan_in_fan_out)
        self.conv_fn = F.conv3d
