# todo
from typing import Callable, Optional
import torch
from transformers.pytorch_utils import Conv1D
from dataclasses import dataclass, asdict, field

import loralib as lora
from loralib import mark_only_lora_as_trainable, lora_state_dict  # flake8: noqa

from ..utils import PETConfig


@dataclass
class LoRAConfig(PETConfig):
    r: int = field(default=None, metadata={"help": "LoRA attention dimension"})
    lora_alpha: int = field(default=None, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "LoRA dropout"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the LoRA model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    target_modules: Optional[list] = field(default=None, metadata={"help": "List of modules to replace with LoRA"})
    bias: str = field(default="none", metadata={"help": "Bias type for LoRA. Can be 'none', 'all' or 'lora_only'"})


class LoRAModel(torch.nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
        self.find_and_replace()
        mark_only_lora_as_trainable(self.model, self.config.bias)

    def find_and_replace(self):
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if any(key.endswith(target_key) for target_key in self.config.target_module_keys):
                parent, target, target_name = self.get_submodules(key)
                if isinstance(target, torch.nn.Linear):
                    new_module = lora.Linear(target.in_features, target.out_features, **asdict(self.config))
                elif isinstance(target, Conv1D):
                    in_features, out_features = target.weight.shape
                    new_module = lora.MergedLinear(in_features, out_features, **asdict(self.config))
                self.replace_module(parent, target_name, new_module)

    def get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[:-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight.clone()
        if old_module.bias is not None:
            new_module.bias = old_module.bias.clone()
