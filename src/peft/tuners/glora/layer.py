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
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


class GloraLayer(BaseTunerLayer):
    adapter_layer_names = (
        "glora_Ad",
        "glora_Au",
        "glora_Bd",
        "glora_Bu",
        "glora_Cd",
        "glora_Cu",
        "glora_D",
        "glora_E",
    )
    other_param_names = ("r", "eval_config")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.in_features: int
        self.out_features: int
        base = self.get_base_layer()
        # Use exact type check: bitsandbytes.Linear4bit subclasses nn.Linear but is not compatible with GLORA math.
        if type(base) is not nn.Linear:
            raise NotImplementedError(
                f"GLORA only supports torch.nn.Linear as base_layer, got {type(base).__name__}. "
                "Quantized Linear subclasses are not supported."
            )
        self.in_features = base.in_features
        self.out_features = base.out_features

        base.weight.requires_grad = False
        if getattr(base, "bias", None) is not None:
            base.bias.requires_grad = False

        self.r: dict[str, int] = {}
        self.glora_Ad: nn.ParameterDict = nn.ParameterDict()
        self.glora_Au: nn.ParameterDict = nn.ParameterDict()
        self.glora_Bd: nn.ParameterDict = nn.ParameterDict()
        self.glora_Bu: nn.ParameterDict = nn.ParameterDict()
        self.glora_Cd: nn.ParameterDict = nn.ParameterDict()
        self.glora_Cu: nn.ParameterDict = nn.ParameterDict()
        self.glora_D: nn.ParameterDict = nn.ParameterDict()
        self.glora_E: nn.ParameterDict = nn.ParameterDict()
        self.eval_config: dict[str, dict[str, object]] = {}
        self.merged_adapters: list[str] = []
        self._disable_adapters: bool = False
        self._active_adapter: str | list[str] = []
        self.kwargs: dict[str, Any] = dict(kwargs)

    def add_adapter(self, adapter_name: str, r: int, config_A_B: str, config_C: str, config_D_E: str) -> None:
        self.r[adapter_name] = r
        Ad, Au = self.make_param((self.out_features, self.in_features), f"lora_{r}")
        Bd, Bu = self.make_param((self.out_features, self.in_features), f"lora_{r}")
        Cd, Cu = self.make_param((self.in_features, 1), f"lora_{r}")
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
        active = list(self.active_adapters)
        if adapter_name not in active:
            active.append(adapter_name)
        self.set_adapter(active, inference_mode=False)
        self._move_adapter_to_device_of_base_layer(adapter_name)

    def reset_glora_parameters(self, adapter_name: str) -> None:
        nn.init.kaiming_uniform_(self.glora_Au[adapter_name], a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.glora_Bu[adapter_name], a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.glora_Cu[adapter_name], a=math.sqrt(5))

    def make_param(self, shape, config=None):
        if config is not None and "lora" in str(config).lower():
            out_feature = shape[0]
            in_feature = shape[1]
            try:
                rank = int(str(config).split("_")[1])
            except (ValueError, IndexError):
                rank = 4
            return nn.Parameter(torch.zeros(out_feature, rank)), nn.Parameter(torch.zeros(rank, in_feature))
        return nn.Parameter(torch.zeros(*shape)), nn.Parameter(torch.zeros(1, 1))

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        for adapter_name in adapter_names:
            if adapter_name not in self.eval_config:
                continue
            path_config = self.eval_config[adapter_name]
            weight = self.weight
            if not isinstance(weight, torch.Tensor):
                raise TypeError(f"weight must be a torch.Tensor, got {type(weight)}")
            bias = self.bias
            if bias is not None and not isinstance(bias, torch.Tensor):
                raise TypeError(f"bias must be a torch.Tensor or None, got {type(bias)}")
            device, dtype = weight.device, weight.dtype
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
            w0 = weight.data.clone()
            b0 = bias.data.clone() if bias is not None else None
            base_layer = self.get_base_layer()
            if safe_merge:
                merged_weight = w0 + w0 * A + B
                if not torch.isfinite(merged_weight).all():
                    raise ValueError(f"NaNs detected in merged weights for adapter {adapter_name}")
                base_layer.weight.data = merged_weight
                if bias is not None:
                    merged_bias = b0 + b0 * D + E + torch.matmul(w0, C).squeeze(-1)
                    if not torch.isfinite(merged_bias).all():
                        raise ValueError(f"NaNs detected in merged bias for adapter {adapter_name}")
                    base_layer.bias.data = merged_bias
            else:
                base_layer.weight.data += (w0 * A) + B
                if bias is not None:
                    base_layer.bias.data += (b0 * D) + E + torch.matmul(w0, C).squeeze(-1)
                elif E.numel() > 0 or C.numel() > 0:
                    new_bias_val = E + torch.matmul(w0, C).squeeze(-1)
                    if not torch.all(new_bias_val == 0):
                        base_layer.register_parameter("bias", nn.Parameter(new_bias_val.to(base_layer.weight.dtype)))
            self.merged_adapters.append(adapter_name)

    def unmerge(self, adapter_names: Optional[list[str]] = None) -> None:
        if adapter_names is None:
            adapter_names = list(self.merged_adapters)
        for adapter_name in adapter_names:
            if adapter_name not in self.merged_adapters:
                continue
            if adapter_name not in self.eval_config:
                continue
            path_config = self.eval_config[adapter_name]
            weight = self.weight
            if not isinstance(weight, torch.Tensor):
                raise TypeError(f"weight must be a torch.Tensor, got {type(weight)}")
            bias = self.bias
            if bias is not None and not isinstance(bias, torch.Tensor):
                raise TypeError(f"bias must be a torch.Tensor or None, got {type(bias)}")
            device, dtype = weight.device, weight.dtype
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
            w_merged = weight.data.clone()
            b_merged = bias.data.clone() if bias is not None else None
            w0 = (w_merged - B) / (1.0 + A)
            base_layer = self.get_base_layer()
            base_layer.weight.data = w0
            if bias is not None:
                base_layer.bias.data = (b_merged - E - torch.matmul(w0, C).squeeze(-1)) / (1.0 + D)
            self.merged_adapters.remove(adapter_name)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        adapter_names = kwargs.pop("adapter_names", None)
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)
        if not self.active_adapters:
            return self.base_layer(x, *args, **kwargs)
        if adapter_names is not None:
            result = self.base_layer(x, *args, **kwargs)
            base_result = result.clone()
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
                idx = sub_batch_indices_list[i]
                sub_batch = x[idx]
                weight_eff = self.weight + self.weight * A + B
                bias_eff = self.bias
                if bias_eff is not None:
                    bias_eff = bias_eff + bias_eff * D + E + torch.matmul(self.weight, C).squeeze(-1)
                else:
                    new_bias_val = E + torch.matmul(self.weight, C).squeeze(-1)
                    if not torch.all(new_bias_val == 0):
                        bias_eff = new_bias_val
                adapted = F.linear(sub_batch, weight_eff, bias=bias_eff)
                result[idx] = result[idx] + (adapted - base_result[idx])
            return result
        if self.merged:
            return self.base_layer(x, *args, **kwargs)
        base_result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = base_result.dtype
        result = base_result
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
            adapted = F.linear(x, weight_eff, bias=bias_eff)
            result = result + (adapted - base_result)
        return result.to(torch_result_dtype)

    def prepare_path(
        self, config: str | object, Xd: nn.Parameter, Xu: Optional[nn.Parameter] = None, device=None, dtype=None
    ):
        config = str(config)  # ty linting
        device = device or Xd.device
        dtype = dtype or Xd.dtype
        if Xu is not None:
            if "lora" in config.lower():
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


class GloraLinear(nn.Module, GloraLayer):
    """GLORA adapter wrapping a dense [`~torch.nn.Linear`] `base_layer`."""

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        nn.Module.__init__(self)
        GloraLayer.__init__(self, base_layer, **kwargs)
        self._disable_adapters = False

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # Explicit dispatch: nn.Module precedes GloraLayer in the MRO, so we must not rely on inheriting forward.
        return GloraLayer.forward(self, x, *args, **kwargs)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Checkpoints from before base_layer composition used top-level weight/bias on the wrapper.
        legacy_weight = prefix + "weight"
        new_weight = prefix + "base_layer.weight"
        if legacy_weight in state_dict and new_weight not in state_dict:
            state_dict[new_weight] = state_dict.pop(legacy_weight)
        legacy_bias = prefix + "bias"
        new_bias = prefix + "base_layer.bias"
        if legacy_bias in state_dict and new_bias not in state_dict:
            state_dict[new_bias] = state_dict.pop(legacy_bias)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self) -> str:
        return "glora." + super().__repr__()
