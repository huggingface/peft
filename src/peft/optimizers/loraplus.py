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
This module contains the implementation of the LoraPlus optimizer.
"""

from __future__ import annotations

from operator import attrgetter

import torch.nn as nn
from torch.optim import Optimizer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from ..peft_model import PeftModel
from ..tuners.lora.layer import Embedding


def create_loraplus_optimizer(
    model: PeftModel, optimizer_cls: type[Optimizer], *, lr: float, loraplus_lr_ratio: float, **kwargs
) -> Optimizer:
    """
    Creates a LoraPlus optimizer.

    Efficient Low Rank Adaptation of Large Models: https://huggingface.co/papers/2402.12354

    Reference: https://github.com/nikhil-ghosh-berkeley/loraplus/

    Args:
        model (`torch.nn.Module`): The model to be optimized.
        optimizer_cls (`torch.optim.Optimizer`): The optimizer class to be used.
        lr (`float`): The learning rate to be used for the optimizer.
        loraplus_lr_ratio (`float`):
            The ratio of learning ηB/ηA where ηA (lr) is passed in as the optimizer learning rate. Should be ≥1. Should
            be set in tandem with the optimizer learning rate (lr); should be larger when the task is more difficult
            and the model needs to update its features to learn well. In this case, it helps to make the learning rate
            slightly smaller (e.g., by a factor of 2) than typical vanilla LoRA learning rates
        loraplus_lr_embedding (optional `float`):
            If LoRA modules are added to embedding layers your can specify a different learning rate for them. Default
            value 1e-6.
        kwargs (`dict`): Additional keyword arguments to be passed to the optimizer.

    Returns:
        `torch.optim.Optimizer`: An instance of the specified optimizer class configured with the model's parameters
        organized into groups with custom learning rates.
    """

    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    param_groups = {
        "groupA": {},
        "groupB": {},
        "groupB_no_decay": {},
        "embedding": {},
    }

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

    kwargs["lr"] = lr
    loraplus_weight_decay = kwargs.pop("loraplus_weight_decay", 0.0)
    loraplus_lr_embedding = kwargs.pop("loraplus_lr_embedding", 1e-6)

    optimizer_grouped_parameters = [
        {
            "params": list(param_groups["groupA"].values()),
            "weight_decay": loraplus_weight_decay,
            "lr": lr,
        },
        {
            "params": list(param_groups["embedding"].values()),
            "weight_decay": loraplus_weight_decay,
            "lr": loraplus_lr_embedding,
        },
        {
            "params": list(param_groups["groupB"].values()),
            "weight_decay": loraplus_weight_decay,
            "lr": lr * loraplus_lr_ratio,
        },
        {
            "params": list(param_groups["groupB_no_decay"].values()),
            "weight_decay": 0.0,
            "lr": lr * loraplus_lr_ratio,
        },
    ]

    optimizer = optimizer_cls(optimizer_grouped_parameters, **kwargs)
    eight_bit_names = ["Adam8bit", "AdamW8bit", "PagedAdam8bit", "PagedAdamW8bit"]
    if optimizer_cls.__name__ in eight_bit_names:
        import bitsandbytes

        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                manager.register_module_override(module, "weight", {"optim_bits": 32})
    return optimizer
