# Copyright 2025-present the HuggingFace Inc. team.
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

import math
from collections.abc import Iterable
from typing import Callable

import torch
import torch.nn as nn
from accelerate.utils.imports import is_bf16_available
from torch import autocast
from torch.optim import Optimizer

from ..peft_model import PeftModel
from ..utils.other import infer_device


class LoraFAOptimizer(Optimizer):
    """
    Implements the LoRA-FA optimizer designed specifically for training Low-Rank Adaptation (LoRA) parameters
    efficiently. Note that LoraFAOptimizer is based on adamw-hf in transformers, with only LoRA part modified. Without
    LoRA it will fall back to adamw-hf.

    Args:
        params (Iterable[nn.parameter.Parameter]): Parameters to optimize.
        lr (float, optional): Learning rate (default: 1e-3).
        betas (Tuple[float, float], optional):
            Coefficients for computing running averages of gradient and squared gradient (default: (0.9, 0.999)).
        eps (float, optional): Term added to denominator to improve numerical stability (default: 1e-6).
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0.0).
        correct_bias (bool, optional): Whether to apply bias correction as in original Adam (default: True).

    Args in sub-function step:
        closure (Callable, optional): A closure that reevaluates the model and returns the loss.

    Reference:
        - LoRA-FA: https://huggingface.co/papers/2308.03303
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
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
                # param_list contains a pair of A and B adapters
                # i.e., param_list -> [A,B]

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

                # Below is the LoRA-FA part
                # 1. In this part, we optimize the gradient of B as:
                #    g^B = \left(\frac{r}{\alpha}\right)^2 (A^\top A)^{-1} g_{\text{LoRA-FA}}^B
                #    to min the func as described below:
                #    \min_{g^B} \|\hat{g}_\text{LoRA-FA} - g\|_F^2
                # 2. After the gradient of B is ready, update the optimizer state
                if len(param_list) == 2:
                    A = param_list[0]
                    B = param_list[1]
                    grad_B_orin = B.grad

                    # projection
                    delta = 1e-8

                    # computing the inverse matrix
                    AA_T = A @ A.T
                    AA_T_inv = torch.linalg.pinv(AA_T + delta * torch.eye(A.shape[0]).to(A.device))

                    device_type = infer_device()

                    if is_bf16_available():
                        with autocast(device_type=device_type, dtype=torch.bfloat16):
                            grad_B = (1 / scaling_factor**2) * (grad_B_orin @ AA_T_inv)
                    else:
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
                        step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                    B.addcdiv_(exp_avg_B, denom_B, value=-step_size)
                    if group["weight_decay"] > 0.0:
                        B.add_(B, alpha=(-group["lr"] * group["weight_decay"]))
                    param_list = []
                    name_list = []

                # Below is the original AdamW
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
                        step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

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
    model: PeftModel, r: int, lora_alpha: int, lr: float, weight_decay: float = 0.0, use_rslora: bool = False
) -> Optimizer:
    """
    Helper function to instantiate a lorafa optimizer specifically configured for a given model using the LoRA method.

    This function will:
    - Disable gradient updates for the "lora_A" parameters (these are typically frozen during LoRA training).
    - Compute the scaling factor based on provided `lora_alpha` and rank `r` for proper gradient projection.
    - Create and configure parameter groups for the optimizer including specified learning rate, weight decay, and
      additional optimizer options.

    For hyper-params, LoRA-FA uses the same hyper-params as AdamW, except for the LoRA hyper-params (r, lora_alpha,
    use_rslora). One can always use the same hyper-params such as lr and weight_decay, as AdamW in LoRA tuning.

    Args:
        model (PeftModel): The model containing LoRA-adapted parameters.
        r (int): Rank of the LoRA decomposition.
        lora_alpha (int): Scaling factor for LoRA parameterization.
        lr (float): Learning rate for optimizer updates.
        weight_decay (float): Weight decay for AdamW.
        use_rslora (bool):
            whether to use rslora. In rslora, the lora scaling factor becomes to lora_alpha / math.sqrt(r) instead of
            lora_alpha / r.

    Returns:
        Optimizer: Configured lorafa optimizer instance ready for training.
    """
    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad_(False)
    lora_scaling = lora_alpha / math.sqrt(r) if use_rslora else lora_alpha / r
    param_groups = [
        {
            "params": model.parameters(),
            "lr": lr,
            "names": [name for name, _ in model.named_parameters()],
            "scaling_factor": lora_scaling,
            "betas": (0.9, 0.999),
            "weight_decay": weight_decay,
        }
    ]
    return LoraFAOptimizer(param_groups)
