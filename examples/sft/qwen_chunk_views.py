# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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

"""Chunk views and row gradients for the Transformers 5.16.1 Qwen fallback.

The recurrence is adapted from Transformers' modeling_qwen3_5.py. Its arithmetic
and dtype conversions are retained. Disjoint chunk gradients are assembled with
unbind/stack, and triangular-row gradients are accumulated directly into their
slices. Higher derivatives use the original differentiable row recurrence.
"""

import torch
import torch.nn.functional as F


def _l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


class _TriangularRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn):
        work = attn.clone()
        rows = []
        subs = []
        for i in range(1, attn.shape[-1]):
            row = work[..., i, :i].clone()
            sub = work[..., :i, :i].clone()
            work[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
            rows.append(row)
            subs.append(sub)
        ctx.size = attn.shape[-1]
        ctx.save_for_backward(attn, *(rows + subs))
        return work

    @staticmethod
    def backward(ctx, output_grad):
        count = ctx.size - 1
        saved = ctx.saved_tensors
        attn = saved[0]
        if torch.is_grad_enabled():
            # Rebuild the ordinary autograd graph when higher derivatives are requested.
            work = attn.clone()
            for i in range(1, ctx.size):
                row = work[..., i, :i].clone()
                sub = work[..., :i, :i].clone()
                work[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
            return torch.autograd.grad(work, attn, output_grad, create_graph=True)[0]
        rows = saved[1 : count + 1]
        subs = saved[count + 1 :]
        grad = output_grad.clone()
        for i in range(ctx.size - 1, 0, -1):
            incoming = grad[..., i, :i].clone()
            grad[..., i, :i] = incoming + (incoming.unsqueeze(-2) * subs[i - 1]).sum(-1)
            grad[..., :i, :i].add_(rows[i - 1].unsqueeze(-1) * incoming.unsqueeze(-2))
        return grad


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    attn = _TriangularRows.apply(attn)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    chunk_outputs = []
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # Local attention products are independent of the recurrent state.
    attention_chunks = (query @ key.transpose(-1, -2) * decay_mask).unbind(2)

    # for each chunk
    query_chunks = query.unbind(2)
    key_chunks = key.unbind(2)
    value_chunks = value.unbind(2)
    k_cumdecay_chunks = k_cumdecay.unbind(2)
    g_chunks = g.unbind(2)
    for i in range(total_sequence_length // chunk_size):
        q_i, k_i, v_i = query_chunks[i], key_chunks[i], value_chunks[i]
        attn = attention_chunks[i]
        v_prime = (k_cumdecay_chunks[i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g_chunks[i][:, :, :, None].exp()) @ last_recurrent_state
        chunk_outputs.append(attn_inter + attn @ v_new)
        last_recurrent_state = (
            last_recurrent_state * g_chunks[i][:, :, -1, None, None].exp()
            + (k_i * (g_chunks[i][:, :, -1, None] - g_chunks[i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = torch.stack(chunk_outputs, dim=2)
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def enable_chunk_views(model):
    """Use the validated fallback for Qwen text SFT without changing other models."""
    import transformers

    if transformers.__version__ != "5.16.1":
        return False

    from transformers.utils import is_kernels_available

    if is_kernels_available():
        return False
    if model.config.get_text_config().model_type != "qwen3_5_text":
        return False

    from transformers.integrations.hub_kernels import use_kernel_func_from_hub_with_fallback
    from transformers.models.qwen3_5 import modeling_qwen3_5

    # Preserve the upstream preference for an installed FLA implementation.
    modeling_qwen3_5.torch_chunk_gated_delta_rule = use_kernel_func_from_hub_with_fallback(
        "chunk_gated_delta_rule", "fla"
    )(torch_chunk_gated_delta_rule)
    return True
