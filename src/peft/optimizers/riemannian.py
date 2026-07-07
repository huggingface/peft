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
This module contains the implementation of the Riemannian-preconditioned LoRA optimizer.
"""

from collections.abc import Callable
from typing import Optional

import torch
from torch.optim import Optimizer

from ..peft_model import PeftModel


def _collect_lora_pairs(
    model: torch.nn.Module,
) -> list[tuple[torch.nn.Parameter, torch.nn.Parameter]]:
    """Return the list of `(lora_A, lora_B)` weight parameter pairs of `model`.

    Pairs are matched by name substitution: for every parameter whose name contains `.lora_A.` the sibling `lora_B`
    parameter is looked up by substituting the substring. Only 2D weight matrices that both require gradients are
    returned, which is exactly what the `r x r` preconditioner is defined for. Any auxiliary trainable parameters that
    don't match the `lora_A` / `lora_B` naming (for example DoRA's magnitude vector) are ignored — the preconditioner
    is only defined for the low-rank product.
    """
    params = dict(model.named_parameters())
    pairs = []
    for name, param_a in params.items():
        if ".lora_A." not in name:
            continue
        param_b = params.get(name.replace(".lora_A.", ".lora_B."))
        if param_b is None:
            continue
        if param_a.ndim != 2 or param_b.ndim != 2:
            continue
        if not (param_a.requires_grad and param_b.requires_grad):
            continue
        pairs.append((param_a, param_b))
    return pairs


class _RiemannianPreconditioner:
    """Applies the `r x r` Riemannian preconditioner in-place to LoRA gradients.

    For each `(lora_A, lora_B)` pair, the Euclidean gradients are rescaled to the scaled-gradient / Riemannian
    directions defined by the paper: `g_A` is replaced by `(B^T B + reg * I_r)^-1 @ g_A`, and `g_B` is replaced by `g_B
    @ (A A^T + reg * I_r)^-1`. Both preconditioners are `r x r`, so storage and runtime overhead are small in the LoRA
    rank.

    The scalar `reg` is a damping term added to the `r x r` matrix's diagonal before inversion; it stabilizes the
    inverse when a factor is (near) rank-deficient.
    """

    def __init__(
        self,
        lora_pairs: list[tuple[torch.nn.Parameter, torch.nn.Parameter]],
        reg: float,
    ) -> None:
        self.lora_pairs = lora_pairs
        self.reg = reg

    @staticmethod
    def _damped_inverse(mat: torch.Tensor, reg: float) -> torch.Tensor:
        r = mat.shape[-1]
        eye = torch.eye(r, device=mat.device, dtype=mat.dtype)
        # pinv keeps this well-behaved even if reg is small and the factor is (near) rank-deficient.
        return torch.linalg.pinv(mat + reg * eye)

    @torch.no_grad()
    def step(self) -> None:
        for param_a, param_b in self.lora_pairs:
            grad_a = param_a.grad
            grad_b = param_b.grad
            if grad_a is None and grad_b is None:
                continue

            # Compute the preconditioners in at least float32 for numerical stability when training in bf16, while
            # preserving higher precision if the parameters already use it. Cast the result back to each grad's dtype.
            compute_dtype = torch.promote_types(param_a.dtype, torch.float32)
            a = param_a.detach().to(compute_dtype)
            b = param_b.detach().to(compute_dtype)

            if grad_a is not None:
                # g_A <- (B^T B + reg I)^-1 g_A
                precond = self._damped_inverse(b.T @ b, self.reg)
                new_grad_a = precond @ grad_a.to(compute_dtype)
                grad_a.copy_(new_grad_a.to(grad_a.dtype))

            if grad_b is not None:
                # g_B <- g_B (A A^T + reg I)^-1
                precond = self._damped_inverse(a @ a.T, self.reg)
                new_grad_b = grad_b.to(compute_dtype) @ precond
                grad_b.copy_(new_grad_b.to(grad_b.dtype))


def create_riemannian_optimizer(
    model: PeftModel,
    optimizer_cls: type[Optimizer],
    *,
    lr: float,
    reg: float = 1e-2,
    **kwargs,
) -> Optimizer:
    """
    Creates a Riemannian-preconditioned optimizer for a LoRA-adapted model.

    Riemannian Preconditioned LoRA for Fine-Tuning Foundation Models: https://huggingface.co/papers/2402.02347

    Reference: https://github.com/pilancilab/Riemannian_Preconditioned_LoRA

    The returned optimizer behaves exactly like `optimizer_cls` (e.g. `torch.optim.AdamW` or `torch.optim.SGD`) except
    that, on every `step`, the gradients of the LoRA `A` and `B` matrices are first multiplied by an `r x r` Riemannian
    preconditioner. Non-LoRA parameters are updated unchanged.

    Only `nn.Linear`-shaped LoRA layers (`lora_A` / `lora_B`) are preconditioned. LoRA on embedding layers
    (`lora_embedding_A` / `lora_embedding_B`, stored as `nn.Parameter`) is left unpreconditioned.

    Args:
        model (`torch.nn.Module`): The PEFT model containing LoRA-adapted parameters.
        optimizer_cls (`type[torch.optim.Optimizer]`): The base optimizer class to wrap, e.g. `torch.optim.AdamW`.
        lr (`float`): Learning rate passed to the base optimizer.
        reg (`float`): Damping added to the `r x r` matrix diagonal before inversion; stabilizes the inverse when a
            LoRA factor is (near) rank-deficient.
        kwargs (`dict`): Additional keyword arguments forwarded to the base optimizer (e.g. `weight_decay`, `betas`).

    Returns:
        `torch.optim.Optimizer`: A subclass instance of `optimizer_cls` that preconditions LoRA gradients on each
        `step`. Only `lora_A` / `lora_B` weight matrices are preconditioned; every other trainable parameter is updated
        by `optimizer_cls` unchanged.
    """
    if not issubclass(optimizer_cls, Optimizer):
        raise TypeError(f"optimizer_cls must be a subclass of torch.optim.Optimizer, got {optimizer_cls!r}.")

    lora_pairs = _collect_lora_pairs(model)
    if not lora_pairs:
        raise ValueError(
            "create_riemannian_optimizer did not find any trainable lora_A/lora_B parameter pairs on the model. "
            "The Riemannian preconditioner only applies to LoRA-style adapters."
        )

    preconditioner = _RiemannianPreconditioner(lora_pairs, reg=reg)
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    class RiemannianPreconditionedOptimizer(optimizer_cls):
        @torch.no_grad()
        def step(self, closure: Optional[Callable] = None):
            preconditioner.step()
            return super().step(closure)

    optimizer = RiemannianPreconditionedOptimizer(trainable_params, lr=lr, **kwargs)
    return optimizer
