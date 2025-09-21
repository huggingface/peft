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

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GLoraLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        # Important: explicitly initialize nn.Module; do not rely on MRO with nn.Linear
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = {}
        # Per-adapter parameter dicts
        self.r = {}
        self.glora_Ad = nn.ParameterDict()
        self.glora_Au = nn.ParameterDict()
        self.glora_Bd = nn.ParameterDict()
        self.glora_Bu = nn.ParameterDict()
        self.glora_Cd = nn.ParameterDict()
        self.glora_Cu = nn.ParameterDict()
        self.glora_D = nn.ParameterDict()
        self.glora_E = nn.ParameterDict()
        self.eval_config = {}
        self.merged_adapters = []
        self._disable_adapters = False
        self.active_adapters = []

    def add_adapter(self, adapter_name: str, r: int, config_A_B: str, config_C: str, config_D_E: str):
        self.r[adapter_name] = r
        # Create parameters for this adapter
        Ad, Au = self.make_param((self.out_features, self.in_features), f"LoRA_{r}")
        Bd, Bu = self.make_param((self.out_features, self.in_features), f"LoRA_{r}")
        Cd, Cu = self.make_param((self.in_features, 1), f"LoRA_{r}")
        D = nn.Parameter(torch.zeros(self.out_features))
        E = nn.Parameter(torch.zeros(self.out_features))
        self.glora_Ad[adapter_name] = Ad
        self.glora_Au[adapter_name] = Au
        self.glora_Bd[adapter_name] = Bd
        self.glora_Bu[adapter_name] = Bu
        self.glora_Cd[adapter_name] = Cd
        self.glora_Cu[adapter_name] = Cu
        self.glora_D[adapter_name] = D
        self.glora_E[adapter_name] = E
        self.eval_config[adapter_name] = {
            "A": config_A_B,
            "B": config_A_B,
            "C": config_C,
            "D": config_D_E,
            "E": config_D_E,
        }
        self.reset_glora_parameters(adapter_name)
        if adapter_name not in self.active_adapters:
            self.active_adapters.append(adapter_name)

    def reset_glora_parameters(self, adapter_name):
        nn.init.kaiming_uniform_(self.glora_Au[adapter_name], a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.glora_Bu[adapter_name], a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.glora_Cu[adapter_name], a=math.sqrt(5))

    def make_param(self, shape, config=None):
        if "LoRA" in config:
            out_feature = shape[0]
            in_feature = shape[1]
            rank = int(config.split("_")[1])
            return nn.Parameter(torch.zeros(out_feature, rank)), nn.Parameter(torch.zeros(rank, in_feature))
        return nn.Parameter(torch.zeros(*shape)), nn.Parameter(torch.zeros(1, 1))

    def set_adapter(self, adapter_name_or_list):
        if isinstance(adapter_name_or_list, str):
            self.active_adapters = [adapter_name_or_list]
        else:
            self.active_adapters = list(adapter_name_or_list)

    def delete_adapter(self, adapter_name):
        for d in [
            self.glora_Ad,
            self.glora_Au,
            self.glora_Bd,
            self.glora_Bu,
            self.glora_Cd,
            self.glora_Cu,
            self.glora_D,
            self.glora_E,
        ]:
            if adapter_name in d:
                del d[adapter_name]
        if adapter_name in self.r:
            del self.r[adapter_name]
        if adapter_name in self.eval_config:
            del self.eval_config[adapter_name]
        if adapter_name in self.active_adapters:
            self.active_adapters.remove(adapter_name)
        if adapter_name in self.merged_adapters:
            self.merged_adapters.remove(adapter_name)

    def enable_adapters(self):
        self._disable_adapters = False

    def disable_adapters(self):
        self._disable_adapters = True

    @property
    def merged(self):
        return len(self.merged_adapters) > 0

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None):
        if adapter_names is None:
            adapter_names = self.active_adapters
        for adapter_name in adapter_names:
            if adapter_name in self.merged_adapters:
                continue
            # Merge logic: W' = W + W*A + B, bias' = bias + bias*D + E + W*C
            path_config = self.eval_config[adapter_name]
            device, dtype = self.weight.device, self.weight.dtype
            A = self.prepare_path(
                path_config["A"], self.glora_Ad[adapter_name], self.glora_Au[adapter_name], device=device, dtype=dtype
            )
            B = self.prepare_path(
                path_config["B"], self.glora_Bd[adapter_name], self.glora_Bu[adapter_name], device=device, dtype=dtype
            )
            C = self.prepare_path(
                path_config["C"], self.glora_Cd[adapter_name], self.glora_Cu[adapter_name], device=device, dtype=dtype
            )
            D = self.prepare_path(path_config["D"], self.glora_D[adapter_name], device=device, dtype=dtype)
            E = self.prepare_path(path_config["E"], self.glora_E[adapter_name], device=device, dtype=dtype)
            if safe_merge:
                orig_weight = self.weight.data.clone()
                orig_bias = self.bias.data.clone() if self.bias is not None else None
                merged_weight = orig_weight + orig_weight * A + B
                if not torch.isfinite(merged_weight).all():
                    raise ValueError(f"NaNs detected in merged weights for adapter {adapter_name}")
                self.weight.data = merged_weight
                if self.bias is not None:
                    merged_bias = orig_bias + orig_bias * D + E + torch.matmul(self.weight, C).squeeze(-1)
                    if not torch.isfinite(merged_bias).all():
                        raise ValueError(f"NaNs detected in merged bias for adapter {adapter_name}")
                    self.bias.data = merged_bias
            else:
                self.weight.data += (self.weight * A) + B
                if self.bias is not None:
                    self.bias.data += (self.bias * D) + E + torch.matmul(self.weight, C).squeeze(-1)
                elif E.numel() > 0 or C.numel() > 0:
                    new_bias_val = E + torch.matmul(self.weight, C).squeeze(-1)
                    if not torch.all(new_bias_val == 0):
                        self.bias = nn.Parameter(new_bias_val)
            self.merged_adapters.append(adapter_name)

    def unmerge(self, adapter_names: Optional[list[str]] = None):
        if adapter_names is None:
            adapter_names = list(self.merged_adapters)
        for adapter_name in adapter_names:
            if adapter_name not in self.merged_adapters:
                continue
            # Unmerge logic: W = W - (W*A + B), bias = bias - (bias*D + E + W*C)
            path_config = self.eval_config[adapter_name]
            device, dtype = self.weight.device, self.weight.dtype
            A = self.prepare_path(
                path_config["A"], self.glora_Ad[adapter_name], self.glora_Au[adapter_name], device=device, dtype=dtype
            )
            B = self.prepare_path(
                path_config["B"], self.glora_Bd[adapter_name], self.glora_Bu[adapter_name], device=device, dtype=dtype
            )
            C = self.prepare_path(
                path_config["C"], self.glora_Cd[adapter_name], self.glora_Cu[adapter_name], device=device, dtype=dtype
            )
            D = self.prepare_path(path_config["D"], self.glora_D[adapter_name], device=device, dtype=dtype)
            E = self.prepare_path(path_config["E"], self.glora_E[adapter_name], device=device, dtype=dtype)
            self.weight.data -= (self.weight * A) + B
            if self.bias is not None:
                self.bias.data -= (self.bias * D) + E + torch.matmul(self.weight, C).squeeze(-1)
            self.merged_adapters.remove(adapter_name)

    def forward(self, x: torch.Tensor, adapter_names: Optional[list[str]] = None) -> torch.Tensor:
        if self._disable_adapters or not self.active_adapters:
            return F.linear(x, self.weight, self.bias)
        if adapter_names is not None:
            # Mixed-batch inference: each sample uses a different adapter
            result = F.linear(x, self.weight, self.bias)
            unique_adapters = set(adapter_names)
            sub_batch_indices_list = [
                [i for i, a in enumerate(adapter_names) if a == adapter] for adapter in unique_adapters
            ]
            for i, active_adapter in enumerate(unique_adapters):
                if active_adapter not in self.glora_Ad:
                    continue
                path_config = self.eval_config[active_adapter]
                device, dtype = self.weight.device, self.weight.dtype
                A = self.prepare_path(
                    path_config["A"],
                    self.glora_Ad[active_adapter],
                    self.glora_Au[active_adapter],
                    device=device,
                    dtype=dtype,
                )
                B = self.prepare_path(
                    path_config["B"],
                    self.glora_Bd[active_adapter],
                    self.glora_Bu[active_adapter],
                    device=device,
                    dtype=dtype,
                )
                C = self.prepare_path(
                    path_config["C"],
                    self.glora_Cd[active_adapter],
                    self.glora_Cu[active_adapter],
                    device=device,
                    dtype=dtype,
                )
                D = self.prepare_path(path_config["D"], self.glora_D[active_adapter], device=device, dtype=dtype)
                E = self.prepare_path(path_config["E"], self.glora_E[active_adapter], device=device, dtype=dtype)
                sub_batch = x[sub_batch_indices_list[i]]
                weight_eff = self.weight + self.weight * A + B
                bias_eff = self.bias
                if bias_eff is not None:
                    bias_eff = bias_eff + bias_eff * D + E + torch.matmul(self.weight, C).squeeze(-1)
                else:
                    new_bias_val = E + torch.matmul(self.weight, C).squeeze(-1)
                    if not torch.all(new_bias_val == 0):
                        bias_eff = new_bias_val
                result[sub_batch_indices_list[i]] = F.linear(sub_batch, weight_eff, bias=bias_eff)
            return result
        # Standard: use all active adapters (sum their effects)
        result = F.linear(x, self.weight, self.bias)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.glora_Ad:
                continue
            path_config = self.eval_config[active_adapter]
            device, dtype = self.weight.device, self.weight.dtype
            A = self.prepare_path(
                path_config["A"],
                self.glora_Ad[active_adapter],
                self.glora_Au[active_adapter],
                device=device,
                dtype=dtype,
            )
            B = self.prepare_path(
                path_config["B"],
                self.glora_Bd[active_adapter],
                self.glora_Bu[active_adapter],
                device=device,
                dtype=dtype,
            )
            C = self.prepare_path(
                path_config["C"],
                self.glora_Cd[active_adapter],
                self.glora_Cu[active_adapter],
                device=device,
                dtype=dtype,
            )
            D = self.prepare_path(path_config["D"], self.glora_D[active_adapter], device=device, dtype=dtype)
            E = self.prepare_path(path_config["E"], self.glora_E[active_adapter], device=device, dtype=dtype)
            weight_eff = self.weight + self.weight * A + B
            bias_eff = self.bias
            if bias_eff is not None:
                bias_eff = bias_eff + bias_eff * D + E + torch.matmul(self.weight, C).squeeze(-1)
            else:
                new_bias_val = E + torch.matmul(self.weight, C).squeeze(-1)
                if not torch.all(new_bias_val == 0):
                    bias_eff = new_bias_val
            result = F.linear(x, weight_eff, bias=bias_eff)
        return result

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


class GLoraLinear(nn.Linear, GLoraLayer):
    def __init__(self, *args, **kwargs):
        # Extract custom GLoraLayer arguments from kwargs
        config_A_B = kwargs.pop("config_A_B", None)
        config_C = kwargs.pop("config_C", None)
        config_D_E = kwargs.pop("config_D_E", None)
        # Initialize nn.Linear first
        in_features = args[0] if len(args) >= 1 else kwargs.get("in_features")
        out_features = args[1] if len(args) >= 2 else kwargs.get("out_features")
        nn.Linear.__init__(self, *args, **kwargs)
        # Initialize GLoraLayer explicitly
        GLoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Store the custom configs for later use if needed
        self._glora_config_A_B = config_A_B
        self._glora_config_C = config_C
        self._glora_config_D_E = config_D_E
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self._disable_adapters = False
        self.active_adapters = []
        self.merged_adapters = []

    # All other methods are inherited from GLoraLayer
