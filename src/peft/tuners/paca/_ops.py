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
"""Column swap, matmul and restore of PaCA as opaque custom ops for `torch.compile`.

The eager implementation writes the adapted columns into the base weight in place. Under `torch.compile`, in-place
writes to a graph input are functionalized, i.e. every layer gets an out-of-place copy of its full weight, which costs
memory and time. The ops below restore the columns before they return, so from the outside they do not mutate their
inputs; registering them as custom ops keeps the compiler from looking inside and copying the weight.

`torch.library` infers the op schemas from the type annotations.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def _swap_in(weight, indices, scalings, deltas):
    originals, adapted = [], []
    for idx, scaling, delta in zip(indices, scalings, deltas):
        original = weight.index_select(1, idx)
        columns = torch.empty_like(original)
        torch.add(original, delta, alpha=scaling, out=columns)
        weight.index_copy_(1, idx, columns)
        originals.append(original)
        adapted.append(columns)
    return originals, adapted


def _write(weight, indices, columns, reverse=False):
    pairs = list(zip(indices, columns))
    for idx, cols in pairs[::-1] if reverse else pairs:
        weight.index_copy_(1, idx, cols)


def _grad_dtype(dtype):
    # half precision gradients of the adapter are returned in fp32, other dtypes are kept
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _mm_grad(a, b):
    dtype = _grad_dtype(a.dtype)
    if dtype == a.dtype:
        return torch.mm(a, b)
    try:
        return torch.mm(a, b, out_dtype=dtype)
    except (TypeError, RuntimeError):
        return torch.mm(a, b).to(dtype)


@torch.library.custom_op("peft::paca_linear_forward", mutates_args=())
def paca_linear_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    indices: list[torch.Tensor],
    scalings: list[float],
    deltas: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Returns [output, *x_parts, *original_columns, *adapted_columns]."""
    w = weight if weight.dtype == x.dtype else weight.to(x.dtype)
    with torch.no_grad():
        originals, adapted = _swap_in(w, indices, scalings, deltas)
        try:
            result = F.linear(x, w, None if bias is None else bias.to(x.dtype))
        finally:
            _write(w, indices, originals, reverse=True)
        x_parts = [x.index_select(-1, idx) for idx in indices]
    return [result, *x_parts, *originals, *adapted]


@paca_linear_forward.register_fake
def _(x, weight, bias, indices, scalings, deltas):
    lead = x.shape[:-1]
    out = [x.new_empty((*lead, weight.shape[0]))]
    out += [x.new_empty((*lead, idx.shape[0])) for idx in indices]
    cols = [x.new_empty((weight.shape[0], idx.shape[0])) for idx in indices]
    return out + cols + [c.new_empty(c.shape) for c in cols]


@torch.library.custom_op("peft::paca_linear_backward", mutates_args=())
def paca_linear_backward(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    indices: list[torch.Tensor],
    scalings: list[float],
    x_parts: list[torch.Tensor],
    originals: list[torch.Tensor],
    adapted: list[torch.Tensor],
    need_grad_x: bool,
) -> list[torch.Tensor]:
    """Returns [grad_x, *grad_deltas]; grad_x is empty if not needed, half precision grad_deltas come as fp32."""
    with torch.no_grad():
        if need_grad_x:
            w = weight if weight.dtype == grad_output.dtype else weight.to(grad_output.dtype)
            _write(w, indices, adapted)
            try:
                grad_x = grad_output.matmul(w)
            finally:
                _write(w, indices, originals, reverse=True)
        else:
            grad_x = grad_output.new_empty(0)
        g2 = grad_output.reshape(-1, grad_output.shape[-1])
        grad_deltas = []
        for x_part, scaling in zip(x_parts, scalings):
            grad = _mm_grad(g2.t(), x_part.reshape(-1, x_part.shape[-1]).to(g2.dtype))
            if scaling != 1:
                grad.mul_(scaling)
            grad_deltas.append(grad)
    return [grad_x, *grad_deltas]


@paca_linear_backward.register_fake
def _(grad_output, weight, indices, scalings, x_parts, originals, adapted, need_grad_x):
    if need_grad_x:
        grad_x = grad_output.new_empty((*grad_output.shape[:-1], weight.shape[1]))
    else:
        grad_x = grad_output.new_empty(0)
    return [grad_x] + [
        grad_output.new_empty((weight.shape[0], idx.shape[0]), dtype=_grad_dtype(grad_output.dtype)) for idx in indices
    ]


def _setup_context(ctx, inputs, output):
    x, weight, bias, indices, scalings, deltas = inputs
    n = len(indices)
    ctx.n = n
    ctx.scalings = list(scalings)
    ctx.x_dtype = x.dtype
    ctx.delta_dtypes = [d.dtype for d in deltas]
    ctx.has_bias = bias is not None
    ctx.save_for_backward(weight, *indices, *output[1:])


def _backward(ctx, grads):
    n = ctx.n
    saved = ctx.saved_tensors
    weight, indices, rest = saved[0], list(saved[1 : 1 + n]), saved[1 + n :]
    x_parts, originals, adapted = list(rest[:n]), list(rest[n : 2 * n]), list(rest[2 * n :])
    grad_output = grads[0]
    need_grad_x = ctx.needs_input_grad[0]
    res = paca_linear_backward(grad_output, weight, indices, ctx.scalings, x_parts, originals, adapted, need_grad_x)
    grad_x = res[0].to(ctx.x_dtype) if need_grad_x else None
    grad_bias = None
    if ctx.has_bias and ctx.needs_input_grad[2]:
        grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
    grad_deltas = [g.to(dt) for g, dt in zip(res[1:], ctx.delta_dtypes)]
    # the structure of the returned gradients must match the inputs, including the lists
    return grad_x, None, grad_bias, [None] * n, None, grad_deltas


torch.library.register_autograd("peft::paca_linear_forward", _backward, setup_context=_setup_context)
