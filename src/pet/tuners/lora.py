# todo
import torch
from transformers import Conv1D

import loralib as lora


class LoRAModel(torch.nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

    def find_and_replace(self):
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if any(key.endswith(target_key) for target_key in self.config["target_module_keys"]):
                parent, target_name, target = self.get_submodules(key)
                if isinstance(target, torch.nn.Linear):
                    new_module = lora.Linear(
                        target.in_features, target.out_features, **self.config["prompt_encoder_config"]
                    )
                elif isinstance(target, torch.nn.Conv1d, Conv1D):
                    new_module = lora.LoRAConv1d(
                        target.in_channels, target.out_channels, target.kernel_size, bias=target.bias is not None
                    )
                self.replace_module(parent, target_name, new_module)

    def get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[:-1]
        target = self.model.get_submodule(key)
        return parent, target_name, target

    def replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight.clone()
        if old_module.bias is not None:
            new_module.bias = old_module.bias.clone()
