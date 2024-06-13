import random
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import GLoraLayer


class Linear(nn.Module, GLoraLayer):
    # GLora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        GLoraLayer.__init__(self, in_features=in_features, out_features=out_features, r=r, adapter_name=adapter_name)

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        nn.Linear.reset_parameters(self)
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
