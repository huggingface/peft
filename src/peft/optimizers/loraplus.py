from operator import attrgetter

import torch.nn as nn
from torch.optim import Optimizer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from ..peft_model import PeftModel


def create_loraplus_optimizer(model: PeftModel, optimizer_cls: type[Optimizer], optimizer_kwargs: dict, loraplus_lr_embedding: float=1e-6) -> Optimizer:
    """
    Creates a LoraPlus optimizer.
    Implementing LoRA+ https://arxiv.org/abs/2402.12354
    Reference: https://github.com/nikhil-ghosh-berkeley/loraplus/

    Args:
        model (`torch.nn.Module`): The model to be optimized.
        optimizer_cls (`torch.optim.Optimizer`): The optimizer class to be used.
        optimizer_kwargs (`dict`): Additional keyword arguments to be passed to the optimizer.
            - **loraplus_lr_ratio** (`float`): The ratio of the learning rate to be used for the embedding layer. Defaults to loraplus_lr_ratio
            - loraplus_lr_embedding (`float`): The learning rate to be used for the embedding layer. Defaults to loraplus_lr_embedding
    """
    from ..tuners.lora.layer import Embedding
    loraplus_lr_ratio = optimizer_kwargs.pop("loraplus_lr_ratio")

    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    param_groups = {
        "groupA": {},
        "groupB": {},
        "groupB_no_decay": {},
        "embedding": {},
    }

    # raise Exception('line 36 in loraplus.py')

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        module = attrgetter(name)(model)
        if isinstance(module, Embedding):
            param_groups["embedding"][name] = param
        elif "lora_B" in name or param.ndim == 1:
            if name in decay_parameters:
                param_groups["groupB"][name] = param
            else:
                param_groups["groupB_no_decay"][name] = param
        else:
            param_groups["groupA"][name] = param

    assigned_param_groups = ""
    for group in param_groups:
        assigned_param_groups += f"{group}\n {list(param_groups[group].keys())}\n\n"

    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    optimizer_grouped_parameters = [
        {
            "params": list(param_groups["groupA"].values()),
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": list(param_groups["embedding"].values()),
            "weight_decay": weight_decay,
            "lr": loraplus_lr_embedding,
        },
        {
            "params": list(param_groups["groupB"].values()),
            "weight_decay": weight_decay,
            "lr": lr * loraplus_lr_ratio,
        },
        {
            "params": list(param_groups["groupB_no_decay"].values()),
            "weight_decay": 0.0,
            "lr": lr * loraplus_lr_ratio,
        },
    ]

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    if optimizer_cls.__name__ == "Adam8bit":
        import bitsandbytes
        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                manager.register_module_override(module, "weight", {"optim_bits": 32})
    return optimizer