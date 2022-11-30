# todo
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers.pytorch_utils import Conv1D

import loralib as lora
from loralib import mark_only_lora_as_trainable

from ..utils import PETConfig


@dataclass
class LoRAConfig(PETConfig):
    r: int = field(default=8, metadata={"help": "LoRA attention dimension"})
    target_modules: Optional[list] = field(default=None, metadata={"help": "List of modules to replace with LoRA"})
    lora_alpha: int = field(default=None, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "LoRA dropout"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the LoRA model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_lora: Optional[list[bool]] = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})
    bias: str = field(default="none", metadata={"help": "Bias type for LoRA. Can be 'none', 'all' or 'lora_only'"})


class LoRAModel(torch.nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
        self.find_and_replace()
        mark_only_lora_as_trainable(self.model, self.config.bias)

    def find_and_replace(self):
        kwargs = {
            "r": self.config.r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "fan_in_fan_out": self.config.fan_in_fan_out,
            "merge_weights": self.config.merge_weights,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if any(key.endswith(target_key) for target_key in self.config.target_modules):
                parent, target, target_name = self.get_submodules(key)
                # print(parent, target, target_name)
                if isinstance(target, torch.nn.Linear):
                    new_module = lora.Linear(target.in_features, target.out_features, **kwargs)
                elif isinstance(target, Conv1D):
                    kwargs.update({"enable_lora": self.config.enable_lora})
                    in_features, out_features = target.weight.shape
                    new_module = lora.MergedLinear(in_features, out_features, **kwargs)
                self.replace_module(parent, target_name, new_module, target)

    def get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
