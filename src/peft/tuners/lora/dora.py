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


class DoraLinearLayer(nn.Module):
    def __init__(self, fan_in_fan_out):
        super().__init__()
        self.fan_in_fan_out = fan_in_fan_out

    def get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        weight = transpose(weight, self.fan_in_fan_out)
        weight = weight + scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    def update_layer(self, *, base_layer, lora_A, lora_B, scaling, place_on_cpu=False) -> None:
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
            weight_norm = self.get_weight_norm(weight.to(lora_A.device), lora_weight, scaling)

        if place_on_cpu:
            weight_norm = weight_norm.to("cpu")
        self.weight = nn.Parameter(weight_norm, requires_grad=True)

    def forward(self, x, *, lora_A, lora_B, scaling, base_layer, base_result=None):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        """
        # Don't use `lora_weight = lora_B.weight @ lora_A.weight` because this causes errors with FSDP. Instead,
        # calculate the same but using forward.
        x_eye = torch.eye(lora_A.weight.shape[1], device=lora_A.weight.device, dtype=x.dtype)
        lora_weight = lora_B(lora_A(x_eye)).T

        magnitude = self.weight
        weight = dequantize_module_weight(base_layer)
        weight = weight.to(x.dtype)
        weight_norm = self.get_weight_norm(weight, lora_weight.detach(), scaling)
        # see section 4.3 of DoRA (https://huggingface.co/papers/2402.09353)
        # "[...] we suggest treating ||V +∆V ||_c in
        # Eq. (5) as a constant, thereby detaching it from the gradient
        # graph. This means that while ||V + ∆V ||_c dynamically
        # reflects the updates of ∆V , it won’t receive any gradient
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

        result_dora = (mag_norm_scale - 1) * base_result + mag_norm_scale * lora_result * scaling

        return result_dora

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.dora." + rep


class DoraEmbeddingLayer(DoraLinearLayer):
    def forward(self, x, *, lora_A, lora_B, scaling, base_layer, embed_fn):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        """
        lora_weight = (lora_A @ lora_B).T
        magnitude = self.weight
        weight = base_layer.weight
        weight_norm = self.get_weight_norm(weight, lora_weight.detach(), scaling)
        # see section 4.3 of DoRA (https://huggingface.co/papers/2402.09353)
        # "[...] we suggest treating ||V +∆V ||_c in
        # Eq. (5) as a constant, thereby detaching it from the gradient
        # graph. This means that while ||V + ∆V ||_c dynamically
        # reflects the updates of ∆V , it won’t receive any gradient
        # during backpropagation"
        weight_norm = weight_norm.detach()
        mag_norm_scale = magnitude / weight_norm
        result_dora = mag_norm_scale * (embed_fn(x, lora_A) @ lora_B) * scaling
        return mag_norm_scale, result_dora

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.dora." + rep


class _DoraConvNdLayer(DoraLinearLayer):
    def get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        weight = weight + scaling * lora_weight
        # the following is needed to have compatibility with the 4/5D weight tensors of Conv2D/3D
        dim = tuple(range(1, weight.dim()))
        weight_norm = weight.norm(p=2, dim=dim, keepdim=True).transpose(1, 0)
        return weight_norm

    def forward(self, x, *, lora_A, lora_B, scaling, base_layer, base_result=None):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        """
        weight = base_layer.weight
        r = lora_A.weight.shape[0]
        lora_weight = torch.mm(lora_B.weight.view([-1, r]), lora_A.weight.view([r, -1]))
        lora_weight = lora_weight.reshape(weight.shape)
        magnitude = self.weight
        weight_norm = self.get_weight_norm(weight, lora_weight.detach(), scaling)
        # see section 4.3 of DoRA (https://huggingface.co/papers/2402.09353)
        # "[...] we suggest treating ||V +∆V ||_c in
        # Eq. (5) as a constant, thereby detaching it from the gradient
        # graph. This means that while ||V + ∆V ||_c dynamically
        # reflects the updates of ∆V , it won’t receive any gradient
        # during backpropagation"
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

        result_dora = (mag_norm_scale - 1) * base_result + mag_norm_scale * lora_B(lora_A(x)) * scaling
        return result_dora

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.dora." + rep


class DoraConv1dLayer(_DoraConvNdLayer):
    def __init__(self, fan_in_fan_out):
        super().__init__(fan_in_fan_out)
        self.conv_fn = F.conv1d


class DoraConv2dLayer(_DoraConvNdLayer):
    def __init__(self, fan_in_fan_out):
        super().__init__(fan_in_fan_out)
        self.conv_fn = F.conv2d


class DoraConv3dLayer(_DoraConvNdLayer):
    def __init__(self, fan_in_fan_out):
        super().__init__(fan_in_fan_out)
        self.conv_fn = F.conv3d
