from operator import attrgetter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Callable, Iterable, Tuple
import warnings
from transformers.utils.versions import require_version
import math

from ..peft_model import PeftModel



def create_riemannian_optimizer(model: PeftModel, optimizer_cls: type[Optimizer], optimizer_kwargs: dict, lr_embedding: float=1e-6, reg: float=1e-6) -> Optimizer:
    """
    Creates a Riemmanian optimizer.
    Implementation: https://github.com/pilancilab/Riemannian_Preconditioned_LoRA
    Reference:  https://arxiv.org/pdf/2402.02347

    Args:
        model (`torch.nn.Module`): The model to be optimized.
        optimizer_cls (`torch.optim.Optimizer`): The optimizer class to be used.
        optimizer_kwargs (`dict`): Additional keyword arguments to be passed to the optimizer.
            - lr_embedding (`float`): The learning rate to be used for the embedding layer. Defaults to lr_embedding
    """

    """TEST VERSION FOR ADAMW"""
    assert optimizer_cls.__name__=='AdamW', 'TEST version only supports AdamW optimizer'
    from ..tuners.lora.layer import Embedding


    param_groups = {
        "lora_params": {},
        "other_params": {},
        "embedding": {}
    }


    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # print(name, param.shape)
        module = attrgetter(name)(model)
        if isinstance(module, Embedding):
            param_groups["embedding"][name] = param
        elif "lora" in name:
            param_groups["lora_params"][name] = param
        else:
            param_groups["other_params"][name] = param


    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    optimizer_grouped_parameters = [
        {
            "params": list(param_groups["lora_params"].values()),
            "weight_decay": weight_decay,
            "lr": lr,
            'is_lora': True
        },
        {
            "params": list(param_groups["embedding"].values()),
            "weight_decay": weight_decay,
            "lr": lr_embedding,
            'is_lora': False
        },
        {
            "params": list(param_groups["other_params"].values()),
            "weight_decay": weight_decay,
            "lr": lr,
            'is_lora': False
        }
    ]

    optimizer_kwargs.update({'reg':reg})
    optimizer = riemannian_AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
    if optimizer_cls.__name__ == "Adam8bit":
        raise Exception('bitsandbytes not supported yet')
    
    return optimizer


class riemannian_AdamW(Optimizer):
    """
    Adapt from AdamW code in transformers.optimization.AdamW

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
        reg: float = 1e-6
    ):
        print('CREATE Riemannian AdamW')
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias, "reg": reg}
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
            if group['is_lora']:
                for p1, p2 in list(zip(group["params"],group["params"][1:]))[::2]:
                    grad = p1.grad
                    if grad.is_sparse:
                        raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                    state = self.state[p1]
                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p1)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p1)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]
                    state["step"] += 1
                    scaler = p2.data
                    scaler_temp = p1.data
                    try:
                        reg_I = self.defaults['reg']*torch.eye(min(p2.shape)).to(p2.device)
                        scaler = torch.inverse(scaler@scaler.T+reg_I) if p2.shape[0]<p2.shape[1] \
                                                    else torch.inverse(scaler.T@scaler+reg_I)
                        assert scaler.shape[0]==min(p2.data.shape), 'wrong dimension'
                    except:
                        print('invalid condition')
                        scaler = None

                    # apply riemannian conditioner
                    if scaler is not None:
                        grad = grad@scaler if grad.shape[1]==scaler.shape[0] else scaler@grad
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

                    p1.addcdiv_(exp_avg, denom, value=-step_size)
                    if group["weight_decay"] > 0.0:
                        p1.add_(p1, alpha=(-group["lr"] * group["weight_decay"]))

                    grad = p2.grad
                    if grad.is_sparse:
                        raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                    state = self.state[p2]
                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p2)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p2)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]
                    state["step"] += 1
                    scaler = scaler_temp
                    try:
                        reg_I = self.defaults['reg']*torch.eye(min(p1.shape)).to(p1.device)
                        scaler = torch.inverse(scaler@scaler.T+reg_I) if p1.shape[0]<p1.shape[1] \
                                                            else torch.inverse(scaler.T@scaler+reg_I)
                        assert scaler.shape[0]==min(p1.data.shape), 'wrong dimension'
                    except:
                        print('invalid condition')
                        scaler = None

                    # apply riemannian conditioner
                    if scaler is not None:
                        grad = grad@scaler if grad.shape[1]==scaler.shape[0] else scaler@grad
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

                    p2.addcdiv_(exp_avg, denom, value=-step_size)

                    if group["weight_decay"] > 0.0:
                        p2.add_(p2, alpha=(-group["lr"] * group["weight_decay"]))
                    
            else:     
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p)

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