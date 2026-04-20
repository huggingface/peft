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


class GloraPath(nn.Module):
    def forward(self):
        raise NotImplementedError


class GloraLoRAPath(GloraPath):
    def __init__(self, out_features, in_features, r):
        super().__init__()
        self.Xd = nn.Parameter(torch.zeros(out_features, r))
        self.Xu = nn.Parameter(torch.zeros(r, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.Xu, a=math.sqrt(5))
        nn.init.zeros_(self.Xd)

    def forward(self):
        return self.Xd @ self.Xu


class GloraVectorPath(GloraPath):
    def __init__(self, features, is_column=True):
        super().__init__()
        shape = (features, 1) if is_column else (features,)
        self.X = nn.Parameter(torch.zeros(*shape))

    def reset_parameters(self):
        nn.init.zeros_(self.X)

    def forward(self):
        return self.X


class GloraScalarPath(GloraPath):
    def __init__(self):
        super().__init__()
        self.X = nn.Parameter(torch.zeros(1))

    def reset_parameters(self):
        nn.init.zeros_(self.X)

    def forward(self):
        return self.X[0]


class GloraEmptyPath(GloraPath):
    def __init__(self, shape=None):
        super().__init__()
        if shape is not None:
            self.register_buffer("zeros", torch.zeros(shape))
        else:
            self.zeros = 0.0

    def forward(self):
        return self.zeros

    def reset_parameters(self):
        pass


class GloraLayer(BaseTunerLayer):
    adapter_layer_names = (
        "glora_A",
        "glora_B",
        "glora_C",
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
        self.glora_A: nn.ModuleDict = nn.ModuleDict()
        self.glora_B: nn.ModuleDict = nn.ModuleDict()
        self.glora_C: nn.ModuleDict = nn.ModuleDict()
        self.glora_D: nn.ModuleDict = nn.ModuleDict()
        self.glora_E: nn.ModuleDict = nn.ModuleDict()
        self.eval_config: dict[str, dict[str, object]] = {}
        self.merged_adapters: list[str] = []
        self._disable_adapters: bool = False
        self._active_adapter: str | list[str] = []
        self.kwargs: dict[str, Any] = dict(kwargs)

    def add_adapter(self, adapter_name: str, r: int, config_A_B: str, config_C: str, config_D_E: str) -> None:
        self.r[adapter_name] = r

        # Initialize A and B
        cfg_ab = config_A_B.lower()
        if "lora" in cfg_ab:
            self.glora_A[adapter_name] = GloraLoRAPath(self.out_features, self.in_features, r)
            self.glora_B[adapter_name] = GloraLoRAPath(self.out_features, self.in_features, r)
        elif "vector" in cfg_ab:
            self.glora_A[adapter_name] = GloraVectorPath(self.out_features, is_column=True)
            self.glora_B[adapter_name] = GloraVectorPath(self.out_features, is_column=True)
        elif "constant" in cfg_ab:
            self.glora_A[adapter_name] = GloraScalarPath()
            self.glora_B[adapter_name] = GloraScalarPath()
        else:
            self.glora_A[adapter_name] = GloraEmptyPath(shape=(self.out_features, self.in_features))
            self.glora_B[adapter_name] = GloraEmptyPath(shape=(self.out_features, self.in_features))

        # Initialize C
        cfg_c = config_C.lower()
        if "lora" in cfg_c:
            self.glora_C[adapter_name] = GloraLoRAPath(self.in_features, 1, r)
        elif "vector" in cfg_c:
            self.glora_C[adapter_name] = GloraVectorPath(self.in_features, is_column=True)
        elif "constant" in cfg_c:
            self.glora_C[adapter_name] = GloraScalarPath()
        else:
            self.glora_C[adapter_name] = GloraEmptyPath(shape=(self.in_features, 1))

        # Initialize D and E
        cfg_de = config_D_E.lower()
        if "vector" in cfg_de:
            self.glora_D[adapter_name] = GloraVectorPath(self.out_features, is_column=False)
            self.glora_E[adapter_name] = GloraVectorPath(self.out_features, is_column=False)
        elif "constant" in cfg_de:
            self.glora_D[adapter_name] = GloraScalarPath()
            self.glora_E[adapter_name] = GloraScalarPath()
        else:
            self.glora_D[adapter_name] = GloraEmptyPath(shape=(self.out_features,))
            self.glora_E[adapter_name] = GloraEmptyPath(shape=(self.out_features,))

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
        for path in ["A", "B", "C", "D", "E"]:
            module = getattr(self, f"glora_{path}")[adapter_name]
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        for adapter_name in adapter_names:
            if adapter_name not in self.eval_config:
                continue

            weight = self.weight
            if not isinstance(weight, torch.Tensor):
                raise TypeError(f"weight must be a torch.Tensor, got {type(weight)}")
            bias = self.bias
            if bias is not None and not isinstance(bias, torch.Tensor):
                raise TypeError(f"bias must be a torch.Tensor or None, got {type(bias)}")

            device, dtype = weight.device, weight.dtype

            A = self.glora_A[adapter_name]().to(device=device, dtype=dtype)
            B = self.glora_B[adapter_name]().to(device=device, dtype=dtype)
            C = self.glora_C[adapter_name]().to(device=device, dtype=dtype)
            D = self.glora_D[adapter_name]().to(device=device, dtype=dtype)
            E = self.glora_E[adapter_name]().to(device=device, dtype=dtype)

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

            weight = self.weight
            if not isinstance(weight, torch.Tensor):
                raise TypeError(f"weight must be a torch.Tensor, got {type(weight)}")
            bias = self.bias
            if bias is not None and not isinstance(bias, torch.Tensor):
                raise TypeError(f"bias must be a torch.Tensor or None, got {type(bias)}")

            device, dtype = weight.device, weight.dtype

            A = self.glora_A[adapter_name]().to(device=device, dtype=dtype)
            B = self.glora_B[adapter_name]().to(device=device, dtype=dtype)
            C = self.glora_C[adapter_name]().to(device=device, dtype=dtype)
            D = self.glora_D[adapter_name]().to(device=device, dtype=dtype)
            E = self.glora_E[adapter_name]().to(device=device, dtype=dtype)

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
            device, dtype = self.weight.device, self.weight.dtype
            for i, active_adapter in enumerate(unique_adapters):
                if active_adapter not in self.glora_A:
                    continue
                A = self.glora_A[active_adapter]().to(device=device, dtype=dtype)
                B = self.glora_B[active_adapter]().to(device=device, dtype=dtype)
                C = self.glora_C[active_adapter]().to(device=device, dtype=dtype)
                D = self.glora_D[active_adapter]().to(device=device, dtype=dtype)
                E = self.glora_E[active_adapter]().to(device=device, dtype=dtype)
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
        device, dtype = self.weight.device, self.weight.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self.glora_A:
                continue
            A = self.glora_A[active_adapter]().to(device=device, dtype=dtype)
            B = self.glora_B[active_adapter]().to(device=device, dtype=dtype)
            C = self.glora_C[active_adapter]().to(device=device, dtype=dtype)
            D = self.glora_D[active_adapter]().to(device=device, dtype=dtype)
            E = self.glora_E[active_adapter]().to(device=device, dtype=dtype)
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


class GloraLinear(nn.Module, GloraLayer):
    """GLORA adapter wrapping a dense [`~torch.nn.Linear`] `base_layer`."""

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        nn.Module.__init__(self)
        GloraLayer.__init__(self, base_layer, **kwargs)
        self._disable_adapters = False

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # Explicit dispatch: nn.Module precedes GloraLayer in the MRO, so we must not rely on inheriting forward.
        return GloraLayer.forward(self, x, *args, **kwargs)

    def __repr__(self) -> str:
        return "glora." + super().__repr__()
