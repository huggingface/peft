# Copyright The FMS HF Tuning Authors
# Copyright 2024 Cute Kernels
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

# Third Party
import torch

# Local
from .compileable_ops import compileable_bincount, group, group_bwd_W, scatter2scatter

BLOCK_M = 128
torch._dynamo.config.capture_scalar_outputs = True


def padded_block_indices(
    sorted_experts_idxs: torch.Tensor, k: int, N_BLOCK_SIZE: int = BLOCK_M
):
    # there is an overhead of launching a custom op so we only use the custom op when compiling
    if torch.compiler.is_compiling():
        expert_counts = compileable_bincount(sorted_experts_idxs, k)
    else:
        expert_counts = sorted_experts_idxs.bincount(minlength=k)

    padded_block_counts = ((expert_counts - 1) // N_BLOCK_SIZE) + 1
    padded_expert_block_end = padded_block_counts.cumsum(-1)
    expert_boundaries_end = expert_counts.cumsum(-1)
    expert_boundaries_start = expert_boundaries_end - expert_counts
    padded_expert_block_start = padded_expert_block_end - padded_block_counts

    block_idxs = torch.arange(
        padded_expert_block_end[-1],
        dtype=sorted_experts_idxs.dtype,
        device=sorted_experts_idxs.device,
    ).unsqueeze(1)

    block_mask = (block_idxs < padded_expert_block_start) | (
        block_idxs >= padded_expert_block_end
    )
    expanded_block_idxs = (
        N_BLOCK_SIZE * (block_idxs - padded_expert_block_start)
        + expert_boundaries_start
    )
    expanded_block_idxs = expanded_block_idxs.masked_fill(block_mask, 0).sum(-1)

    return expanded_block_idxs, expert_boundaries_end


class _ScatteredExperts(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        expert_weights,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        padded_block_idxs,
        expert_offsets,
        gates=None,
        grouped_in=False,
        grouped_out=False,
        expert_lora_A=None,
        expert_lora_B=None,
        lora_alp: float = 0.0,
    ):
        output = torch.empty(
            sorted_expert_idxs.size(0),
            expert_weights.size(-1),
            device=x.device,
            dtype=x.dtype,
        )

        scatter2scatter(
            X=x,
            W=expert_weights,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            padded_block_idxs=padded_block_idxs,
            out=output,
            FAN_OUT=k,
            x_grouped=grouped_in,
            y_grouped=grouped_out,
            A=expert_lora_A,
            B=expert_lora_B,
            lora_alp=lora_alp,
        )

        _extra_tensors_to_save = ()
        if lora_alp > 0 and expert_lora_A is not None and expert_lora_B is not None:
            _extra_tensors_to_save = (expert_lora_A, expert_lora_B)

            # save some extra context
            ctx.lora_r = expert_lora_A.size(-1)
            ctx.lora_alp = lora_alp

        if gates is None:
            output_expanded = None
        else:
            output_expanded = output.view(gates.size(0), gates.size(1), output.size(-1))
            output = torch.bmm(gates.unsqueeze(1), output_expanded).squeeze(1)

        ctx.save_for_backward(
            x,
            expert_weights,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates,
            output_expanded,
            *_extra_tensors_to_save,
        )

        ctx.grouped_in = grouped_in
        ctx.grouped_out = grouped_out
        ctx.k = k

        return output

    @staticmethod
    def backward(ctx, grad_out):
        (
            x,
            expert_weights,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates,
            output_expanded,
            *_extra_saved_tensors,
        ) = ctx.saved_tensors
        k = ctx.k
        grouped_in = ctx.grouped_in
        grouped_out = ctx.grouped_out

        use_lora = False
        if hasattr(ctx, "lora_r"):
            lora_r = ctx.lora_r
            lora_alp = ctx.lora_alp
            expert_lora_A, expert_lora_B = _extra_saved_tensors
            use_lora = True

        if gates is None:
            d_gates = None
            gates_flat = None
            gate_fan = 1
            # grouped_grad_out = None
        else:
            # calculate gates gradient
            d_gates = torch.bmm(output_expanded, grad_out.unsqueeze(2)).squeeze(-1)
            gates_flat = gates.flatten()
            gate_fan = gates.size(1)
            # print("expanded and grouping")
            # grouped_grad_out = output_expanded.flatten(0, 1)  # reuse expanded buffer later

        if grouped_out:
            grouped_grad_out = grad_out
        else:
            grouped_grad_out = torch.zeros(
                (grad_out.shape[0] * gate_fan, grad_out.shape[1]),
                dtype=grad_out.dtype,
                device=grad_out.device,
            )
            group(
                A=grad_out,
                sorted_expert_idxs=sorted_scattered_idxs,
                out=grouped_grad_out,
                coeff=gates_flat,
                fan_out=gate_fan,
            )

        if grouped_in:
            grouped_x = x
            d_expanded_input = torch.empty(
                sorted_expert_idxs.size(0),
                expert_weights.size(1),
                device=x.device,
                dtype=x.dtype,
            )
        else:
            grouped_x = torch.empty(
                sorted_scattered_idxs.size(0), x.size(1), dtype=x.dtype, device=x.device
            )
            group(
                A=x,
                sorted_expert_idxs=sorted_scattered_idxs,
                out=grouped_x,
                fan_out=k,
            )

            d_expanded_input = grouped_x

        d_weights = torch.zeros(
            expert_weights.size(0),
            grouped_grad_out.size(-1),
            grouped_x.size(-1),
            device=grouped_grad_out.device,
            dtype=grouped_grad_out.dtype,
        ).permute(0, 2, 1)

        group_bwd_W(
            DY=grouped_grad_out,
            X=grouped_x,
            expert_offsets=expert_offsets,
            DW=d_weights,
            E=expert_weights.size(0),
        )

        _extra_scatter_kwargs = {}
        _extra_grads_to_return = (None, None)
        if use_lora:
            d_weights_A = (
                d_weights @ expert_lora_B.permute(0, 2, 1) * (lora_alp / lora_r)
            )
            d_weights_B = (
                expert_lora_A.permute(0, 2, 1) @ d_weights * (lora_alp / lora_r)
            )
            d_weights = None  # zero it

            _extra_scatter_kwargs = {
                "A": expert_lora_B.permute(0, 2, 1),  # B^T
                "B": expert_lora_A.permute(0, 2, 1),  # A^T
                "lora_alp": lora_alp,
            }
            _extra_grads_to_return = (d_weights_A, d_weights_B)

        scatter2scatter(
            X=grouped_grad_out,
            W=expert_weights.permute(0, 2, 1),
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            padded_block_idxs=padded_block_idxs,
            out=d_expanded_input,
            FAN_OUT=1,
            x_grouped=True,
            y_grouped=grouped_in,
            **_extra_scatter_kwargs,
        )

        if k == 1:
            d_input = d_expanded_input
        else:
            d_input = d_expanded_input.view(
                x.size(0), k, d_expanded_input.size(-1)
            ).sum(-2)

        # print("backward end.")
        return (
            # x, expert_weights, k,
            d_input,
            d_weights,
            None,
            # sorted_expert_idxs, sorted_scattered_idxs,
            None,
            None,
            # padded_block_idxs, expert_offsets,
            None,
            None,
            # gates
            d_gates,
            None,
            None,
            # adapter stuff
            *_extra_grads_to_return,
            None,
        )


def scattered_experts(
    inputs,
    expert_weights,
    k,
    sorted_expert_idxs,
    sorted_scattered_idxs,
    padded_block_idxs,
    expert_offsets,
    gates=None,
    grouped_in=False,
    grouped_out=False,
    expert_lora_A=None,
    expert_lora_B=None,
    lora_alp: float = 0.0,
):
    return _ScatteredExperts.apply(
        inputs,
        expert_weights,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        padded_block_idxs,
        expert_offsets,
        gates,
        grouped_in,
        grouped_out,
        expert_lora_A,
        expert_lora_B,
        lora_alp,
    )