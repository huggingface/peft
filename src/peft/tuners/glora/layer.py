import math
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer


class GLoraLayer(BaseTunerLayer):
    def __init__(self, in_features: int, out_features: int, r: int, adapter_name: str, **kwargs):
        super().__init__()
        # Initialize parameters
        self.r = {}
        self.r[adapter_name] = r

        # Initialize learnable parameters
        self.glora_Ad, self.glora_Au = self._make_param((out_features, in_features), f"LoRA_{r}")
        self.glora_Bd, self.glora_Bu = self._make_param((out_features, in_features), f"LoRA_{r}")
        self.glora_Cd, self.glora_Cu = self._make_param((in_features, 1), f"LoRA_{r}")
        self.glora_D = nn.Parameter(torch.zeros(out_features))
        self.glora_E = nn.Parameter(torch.zeros(out_features))

        # Generate configurations
        config_A_B = [f"LoRA_{r}", "vector", "constant", "none"]
        config_C = [f"LoRA_{r}", "vector", "none"]
        config_D_E = ["constant", "none", "vector"]
        self.configs = [{"A": A, "B": B, "C": C, "D": D, "E": E} 
                        for A in config_A_B
                        for B in config_A_B
                        for C in config_C
                        for D in config_D_E
                        for E in config_D_E]

        # Initialize parameter weights using Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.glora_Au, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.glora_Bu, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.glora_Cu, a=math.sqrt(5))

        # Mark the weight as unmerged ?????
        #self.merged = False
        if self.disable_adapters:
            if self.merged:
                self.unmerge()

    def _make_param(self, shape, config=None):
        if config and "LoRA" in config:
            try:
                rank = int(config.split("_")[1])
            except ValueError:
                rank = 4
            return nn.Parameter(torch.zeros(shape[0], rank)), nn.Parameter(torch.zeros(rank, shape[1]))
        return nn.Parameter(torch.zeros(*shape))


class Linear(nn.Module, GLoraLayer):
    # GLora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        bias: bool = True,
        **kwargs
        ) -> None:
        super().__init__()
        GLoraLayer.__init__(self, in_features, out_features, r, adapter_name)
        #Initialize nn.Linear parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if kwargs.get("bias", True):
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

       # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters(self)

        self.active_adapter = adapter_name
        self.to(self.weight.device)

    def merge(self):
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        path_config = self.eval_config
        A = self.prepare_path(path_config["A"], self.glora_Ad, self.glora_Au).to(self.weight.dtype)
        B = self.prepare_path(path_config["B"], self.glora_Bd, self.glora_Bu).to(self.weight.dtype)
        C = self.prepare_path(path_config["C"], self.glora_Cd, self.glora_Cu).to(self.weight.dtype)
        D = self.prepare_path(path_config["D"], self.glora_D).to(self.weight.dtype)
        E = self.prepare_path(path_config["E"], self.glora_E).to(self.weight.dtype)
        self.weight.data += self.weight * A + B
        if torch.is_tensor(self.bias):
            self.bias.data += self.bias * D + E + torch.matmul(self.weight, C).squeeze()
        else:
            self.bias = nn.Parameter(E + torch.matmul(self.weight, C).squeeze())
        self.merged = True

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if self.eval_config is not None:
            path_config = self.eval_config
        else:
            path_config = random.choice(self.configs)
        A = self.prepare_path(path_config["A"], self.glora_Ad, self.glora_Au).to(self.weight.dtype)
        B = self.prepare_path(path_config["B"], self.glora_Bd, self.glora_Bu).to(self.weight.dtype)
        C = self.prepare_path(path_config["C"], self.glora_Cd, self.glora_Cu).to(self.weight.dtype)
        D = self.prepare_path(path_config["D"], self.glora_D).to(self.weight.dtype)
        E = self.prepare_path(path_config["E"], self.glora_E).to(self.weight.dtype)
        if torch.is_tensor(self.bias):
            result = F.linear(
                x,
                self.weight + self.weight * A + B,
                bias=self.bias + self.bias * D + E + torch.matmul(self.weight, C).squeeze(),
            )
        else:
            result = F.linear(x, self.weight + self.weight * A + B, bias=E + torch.matmul(self.weight, C).squeeze())
        result = result.to(previous_dtype)

        return result

    def prepare_path(self, config, Xd, Xu=None):
        if Xu is not None:
            if "LoRA" in config:
                rank = int(config.split("_")[1])
                X = torch.matmul(Xd[:, :rank], Xu[:rank, :])
            elif "vector" in config:
                X = Xd[:, 0].unsqueeze(1)
            elif "constant" in config:
                X = Xd[0, 0]
            elif "none" in config:
                X = torch.zeros(Xd.shape[0], Xu.shape[1]).to(self.weight.device)
            else:
                raise ValueError
        else:
            if "vector" in config:
                X = Xd
            elif "constant" in config:
                X = Xd[0]
            elif "none" in config:
                X = torch.zeros(1).to(self.weight.device)
            else:
                raise ValueError
        return X
