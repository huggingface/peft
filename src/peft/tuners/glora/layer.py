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

import itertools
import math
import random
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


random.seed(56)


class GLoraLayer:
    def __init__(self, adapter_name: str, in_features: int, out_features: int, r: int = 0, **kwargs):
        self.r = {}
        self.r[adapter_name] = r
        self.glora_Ad, self.glora_Au = self.make_param((out_features, in_features), f"LoRA_{r}")
        self.glora_Bd, self.glora_Bu = self.make_param((out_features, in_features), f"LoRA_{r}")
        self.glora_Cd, self.glora_Cu = self.make_param((in_features, 1), f"LoRA_{r}")
        self.glora_D = nn.Parameter(torch.zeros(out_features))
        self.glora_E = nn.Parameter(torch.zeros(out_features))
        # Initialize the parameters for GLora
        self.eval_config = kwargs.get("eval_config", None)
        nn.init.kaiming_uniform_(self.glora_Au, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.glora_Bu, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.glora_Cu, a=math.sqrt(5))
        # Mark the weight as unmerged
        self.merged = False
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs
        config_A_B = [f"LoRA_{r}", "vector", "constant", "none"]
        config_C = [f"LoRA_{r}", "vector", "none"]
        config_D_E = ["constant", "none", "vector"]
        # All possible configurations for A, B, C, D, E
        self.configs = [
            {"A": a, "B": b, "C": c, "D": d, "E": e}
            for a, b, c, d, e in itertools.product(config_A_B, config_A_B, config_C, config_D_E, config_D_E)
        ]

    def make_param(self, shape, config=None):
        if "LoRA" in config:
            out_feature = shape[0]
            in_feature = shape[1]
            rank = int(config.split("_")[1])
            return nn.Parameter(torch.zeros(out_feature, rank)), nn.Parameter(torch.zeros(rank, in_feature))
        return nn.Parameter(torch.zeros(*shape))


class Linear(nn.Linear, GLoraLayer):
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        GLoraLayer.__init__(
            self, in_features=in_features, out_features=out_features, r=r, adapter_name=adapter_name, **kwargs
        )

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.active_adapter = adapter_name
        self.to(self.weight.device)

    # TODO: add safe_merge and adapter_names here
    def merge(self):
        if self.merged:
            return

        # If eval_config is not set, use a random config for merging
        if self.eval_config is None:
            warnings.warn("eval_config not set for GLora layer, using a random config for merge.")
            path_config = random.choice(self.configs)
        else:
            path_config = self.eval_config

        device, dtype = self.weight.device, self.weight.dtype
        # Prepare paths (A, B, C, D, E) based on path_config
        A = self.prepare_path(path_config["A"], self.glora_Ad, self.glora_Au, device=device, dtype=dtype)
        B = self.prepare_path(path_config["B"], self.glora_Bd, self.glora_Bu, device=device, dtype=dtype)
        C = self.prepare_path(path_config["C"], self.glora_Cd, self.glora_Cu, device=device, dtype=dtype)
        D = self.prepare_path(path_config["D"], self.glora_D, device=device, dtype=dtype)
        E = self.prepare_path(path_config["E"], self.glora_E, device=device, dtype=dtype)

        # Merge logic: W' = W + W*A + B
        #             bias' = bias + bias*D + E + W*C
        self.weight.data += (self.weight * A) + B  # Add B

        if self.bias is not None:
            self.bias.data += (self.bias * D) + E + torch.matmul(self.weight, C).squeeze(-1)
        elif E.numel() > 0 or C.numel() > 0:  # If no original bias, but E or C can create one
            # If original bias is None, create it if E or (W*C) are non-zero
            new_bias_val = E + torch.matmul(self.weight, C).squeeze(-1)
            if not torch.all(new_bias_val == 0):  # only create if non-zero
                self.bias = nn.Parameter(new_bias_val)

        self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return F.linear(x, self.weight, self.bias)

        previous_dtype = x.dtype
        if self.eval_config is not None:
            path_config = self.eval_config
        else:
            path_config = random.choice(self.configs)

        # Ensure they are on the same device and dtype as self.weight
        device, dtype = self.weight.device, self.weight.dtype

        # Prepare paths (A, B, C, D, E) based on path_config
        A = self.prepare_path(path_config["A"], self.glora_Ad, self.glora_Au, device=device, dtype=dtype)
        B = self.prepare_path(path_config["B"], self.glora_Bd, self.glora_Bu, device=device, dtype=dtype)
        C = self.prepare_path(path_config["C"], self.glora_Cd, self.glora_Cu, device=device, dtype=dtype)
        D = self.prepare_path(path_config["D"], self.glora_D, device=device, dtype=dtype)
        E = self.prepare_path(path_config["E"], self.glora_E, device=device, dtype=dtype)

        weight_eff = self.weight.clone()
        weight_eff = weight_eff + weight_eff * A + B

        bias_eff = None
        if self.bias is not None:
            bias_eff = self.bias.clone()
            bias_eff = bias_eff + bias_eff * D + E + torch.matmul(self.weight, C).squeeze(-1)
        else:
            new_bias_val = E + torch.matmul(self.weight, C).squeeze(-1)
            if not torch.all(new_bias_val == 0) or path_config["E"] != "none" or path_config["C"] != "none":
                bias_eff = new_bias_val

        result = F.linear(x, weight_eff, bias=bias_eff)
        return result.to(previous_dtype)

    def prepare_path(self, config: str, Xd: nn.Parameter, Xu: Optional[nn.Parameter] = None, device=None, dtype=None):
        device = device or Xd.device
        dtype = dtype or Xd.dtype

        if Xu is not None:
            if "LoRA" in config:
                rank = int(config.split("_")[1])
                X = torch.matmul(Xd[:, :rank], Xu[:rank, :])
            elif "vector" in config:
                X = Xd[:, 0].unsqueeze(1)
            elif "constant" in config:
                X = Xd[0, 0]
            elif "none" in config:
                X = torch.zeros(Xd.shape[0], Xu.shape[1], device=device, dtype=dtype)
            else:
                raise ValueError(f"Unknown config choice: {config} for decomposable path")
        else:
            if "vector" in config:
                X = Xd
            elif "constant" in config:
                X = Xd[0]
            elif "none" in config:
                X = torch.zeros(1, device=device, dtype=dtype)
            else:
                raise ValueError(f"Unknown config choice: {config} for non-decomposable path")
        return X.to(device=device, dtype=dtype)
