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

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _get_group_dim(in_features: int, num_groups: int) -> int:
    return (in_features + num_groups - 1) // num_groups


def _reshape_to_grouped_subtokens(x: torch.Tensor, num_groups: int) -> torch.Tensor:
    group_dim = _get_group_dim(x.shape[-1], num_groups)
    padded_features = num_groups * group_dim
    if x.shape[-1] != padded_features:
        x = F.pad(x, (0, padded_features - x.shape[-1]))
    return x.reshape(-1, num_groups, group_dim)


def _compress_activations(x: torch.Tensor, embed: torch.Tensor, num_groups: int) -> torch.Tensor:
    grouped = _reshape_to_grouped_subtokens(x, num_groups)
    return torch.einsum("tgd,d->tg", grouped, embed)


def _reconstruct_activations(
    compressed: torch.Tensor,
    embed: torch.Tensor,
    in_features: int,
    velora_scale: float,
) -> torch.Tensor:
    grouped = compressed.unsqueeze(-1) * embed.view(1, 1, -1)
    reconstructed = grouped.reshape(-1, compressed.shape[-1] * embed.numel())
    return reconstructed[:, :in_features] * velora_scale


def _normalize_projection(embed: torch.Tensor) -> torch.Tensor:
    embed = embed.float()
    norm = torch.linalg.vector_norm(embed)
    if norm == 0:
        embed = torch.ones_like(embed)
        norm = torch.linalg.vector_norm(embed)
    return embed / norm


class VeloraFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        embed: torch.Tensor,
        num_groups: int,
        velora_scale: float,
    ) -> torch.Tensor:
        output = F.linear(input, weight, bias)

        ctx.input_shape = tuple(input.shape)
        ctx.input_dtype = input.dtype
        ctx.in_features = input.shape[-1]
        ctx.num_groups = num_groups
        ctx.velora_scale = velora_scale
        ctx.bias_dtype = None if bias is None else bias.dtype

        compressed = _compress_activations(input, embed.to(dtype=input.dtype), num_groups)
        ctx.save_for_backward(compressed, weight, embed)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        compressed, weight, embed = ctx.saved_tensors
        compute_dtype = grad_output.dtype

        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1]).to(compute_dtype)
        weight_compute = weight.to(compute_dtype)

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output_2d @ weight_compute
            grad_input = grad_input.reshape(ctx.input_shape).to(dtype=ctx.input_dtype)

        if ctx.needs_input_grad[1]:
            reconstructed = _reconstruct_activations(
                compressed.to(compute_dtype),
                embed.to(compute_dtype),
                ctx.in_features,
                ctx.velora_scale,
            )
            grad_weight = grad_output_2d.transpose(0, 1) @ reconstructed
            grad_weight = grad_weight.to(dtype=weight.dtype)

        if ctx.needs_input_grad[2]:
            grad_bias = grad_output_2d.sum(dim=0)
            grad_bias = grad_bias.to(dtype=ctx.bias_dtype)

        return grad_input, grad_weight, grad_bias, None, None, None
