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

"""
This module contains the implementation of the LoRA-FA optimizer.
"""

from __future__ import annotations

from typing import Tuple, Callable, Iterable

import math

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.cuda.amp import autocast

from ..peft_model import PeftModel


class lorafa(Optimizer):
    """
    Implements the LoRA-FA optimizer designed specifically for training Low-Rank Adaptation (LoRA) parameters efficiently.

    Args:
        params (Iterable[nn.parameter.Parameter]): Parameters to optimize.
        lr (float, optional): Learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): Coefficients for computing running averages of gradient and squared gradient (default: (0.9, 0.999)).
        eps (float, optional): Term added to denominator to improve numerical stability (default: 1e-6).
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0.0).
        correct_bias (bool, optional): Whether to apply bias correction as in original Adam (default: True).

    Reference:
        - LoRA-FA: https://arxiv.org/abs/2308.03303
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)

    def is_same(self, name_list):
        return name_list[0].split(".")[:-3] == name_list[1].split(".")[:-3]

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            scaling_factor = group["scaling_factor"]
            param_list = []
            name_list = []
            for p, n in zip(group["params"], group["names"]):
                # Skip non-lora no-grad module, since we need lora_A which is no-grad.
                if "lora" not in n and p.grad is None:
                    continue
                grad = p.grad

                if "lora" in n:
                    param_list.append(p)
                    name_list.append(n)
                    if len(param_list) == 2:
                        name = n[: n.find("lora")] + "lora"
                    elif len(param_list) == 1:
                        continue
                else:
                    name = n

                # param_list -> [A,B]

                state = self.state[name]
                # State initialization
                if len(state) == 0:
                    if len(param_list) == 2:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg_B"] = torch.zeros_like(param_list[1])
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq_B"] = torch.zeros_like(param_list[1])
                    else:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p)

                if len(param_list) == 2:
                    A = param_list[0]
                    B = param_list[1]
                    grad_B_orin = B.grad

                    # projection
                    delta = 1e-8

                    # computing the inverse matrix
                    AA_T = A @ A.T
                    AA_T_inv = torch.linalg.pinv(
                        AA_T + delta * torch.eye(A.shape[0]).to(A.device)
                    )

                    with autocast(dtype=torch.bfloat16):
                        grad_B = (1 / scaling_factor**2) * (grad_B_orin @ AA_T_inv)
                    if grad_B.dtype != B.grad.dtype:
                        grad_B = grad_B.to(B.grad.dtype)

                    exp_avg_B, exp_avg_sq_B = state["exp_avg_B"], state["exp_avg_sq_B"]
                    beta1, beta2 = group["betas"]
                    state["step"] += 1
                    exp_avg_B.mul_(beta1).add_(grad_B, alpha=(1.0 - beta1))
                    exp_avg_sq_B.mul_(beta2).addcmul_(grad_B, grad_B, value=1.0 - beta2)

                    denom_B = exp_avg_sq_B.sqrt().add_(group["eps"])
                    step_size = group["lr"]
                    if group["correct_bias"]:  # No bias correction for Bert
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size = (
                            step_size * math.sqrt(bias_correction2) / bias_correction1
                        )
                    B.addcdiv_(exp_avg_B, denom_B, value=-step_size)
                    if group["weight_decay"] > 0.0:
                        B.add_(B, alpha=(-group["lr"] * group["weight_decay"]))
                    param_list = []
                    name_list = []
                else:
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]

                    state["step"] += 1

                    # Decay the first and second moment running average coefficient
                    # In-place operations to update the averages at the same time
                    exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                    step_size = group["lr"]
                    if group["correct_bias"]:  # No bias correction for Bert
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size = (
                            step_size * math.sqrt(bias_correction2) / bias_correction1
                        )

                    p.addcdiv_(exp_avg, denom, value=-step_size)

                    # Just adding the square of the weights to the loss function is *not*
                    # the correct way of using L2 regularization/weight decay with Adam,
                    # since that will interact with the m and v parameters in strange ways.
                    #
                    # Instead we want to decay the weights in a manner that doesn't interact
                    # with the m/v parameters. This is equivalent to adding the square
                    # of the weights to the loss with plain (non-momentum) SGD.
                    # Add weight decay at the end (fixed version)
                    if group["weight_decay"] > 0.0:
                        p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss


def create_lorafa_optimizer(
    model: PeftModel, r: int, lora_alpha: int, learning_rate: float, **kwargs
) -> Optimizer:
    """
    Helper function to instantiate a lorafa optimizer specifically configured for a given model using the LoRA method.

    This function will:
    - Disable gradient updates for the "lora_A" parameters (these are typically frozen during LoRA training).
    - Compute the scaling factor based on provided `lora_alpha` and rank `r` for proper gradient projection.
    - Create and configure parameter groups for the optimizer including specified learning rate, weight decay, and additional optimizer options.

    Args:
        model (PeftModel): The model containing LoRA-adapted parameters.
        r (int): Rank of the LoRA decomposition.
        lora_alpha (int): Scaling factor for LoRA parameterization.
        learning_rate (float): Learning rate for optimizer updates.
        **kwargs: Additional optimizer configuration parameters (e.g., weight_decay).

    Returns:
        Optimizer: Configured lorafa optimizer instance ready for training.
    """
    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad_(False)
    lora_scaling = lora_alpha / r
    weight_decay = kwargs.pop("weight_decay", 0.0)
    param_groups = [
        {
            "params": model.parameters(),
            "lr": learning_rate,
            "names": [name for name, _ in model.named_parameters()],
            "scaling_factor": lora_scaling,
            "betas": (0.9, 0.999),
            "weight_decay": weight_decay,
        }
    ]
    return lorafa(param_groups)
