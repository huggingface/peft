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

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


class GloraPath(nn.Module):
    def forward(self):
        raise NotImplementedError

    def reset_parameters(self, init_weights: bool = True):
        raise NotImplementedError


class GloraLoraPath(GloraPath):
    def __init__(self, out_features, in_features, r, init_weights: bool = True):
        super().__init__()
        self.Xd = nn.Parameter(torch.zeros(out_features, r))
        self.Xu = nn.Parameter(torch.zeros(r, in_features))
        self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights: bool = True):
        nn.init.kaiming_uniform_(self.Xu, a=math.sqrt(5))
        if init_weights:
            nn.init.zeros_(self.Xd)
        else:
            nn.init.kaiming_uniform_(self.Xd, a=math.sqrt(5))

    def forward(self):
        return self.Xd @ self.Xu


class GloraVectorPath(GloraPath):
    def __init__(self, features, is_column=True, init_weights: bool = True):
        super().__init__()
        shape = (features, 1) if is_column else (features,)
        self.X = nn.Parameter(torch.zeros(*shape))
        self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights: bool = True):
        if init_weights:
            nn.init.zeros_(self.X)
        else:
            nn.init.kaiming_uniform_(self.X.unsqueeze(0) if self.X.dim() == 1 else self.X, a=math.sqrt(5))

    def forward(self):
        return self.X


class GloraScalarPath(GloraPath):
    def __init__(self, init_weights: bool = True):
        super().__init__()
        self.X = nn.Parameter(torch.zeros(1))
        self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights: bool = True):
        if init_weights:
            nn.init.zeros_(self.X)
        else:
            nn.init.normal_(self.X, mean=0.0, std=0.02)

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

    def reset_parameters(self, init_weights: bool = True):
        pass


class GloraLayer(BaseTunerLayer):
    adapter_layer_names = (
        "glora_A",
        "glora_B",
        "glora_C",
        "glora_D",
        "glora_E",
    )
    other_param_names = ("r",)

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.in_features: int
        self.out_features: int
        base = self.get_base_layer()
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
        self.merged_adapters: list[str] = []
        self._disable_adapters: bool = False
        self._active_adapter: str | list[str] = []
        self.kwargs: dict[str, Any] = dict(kwargs)

    def update_layer(self, adapter_name: str, r: int, config, **kwargs) -> None:
        from .config import GloraConfig

        assert isinstance(config, GloraConfig)
        config_A_B = config.config_A_B
        config_C = config.config_C
        config_D_E = config.config_D_E
        init_weights = config.init_weights

        self.r[adapter_name] = r

        if config_A_B == "lora":
            self.glora_A[adapter_name] = GloraLoraPath(self.out_features, self.in_features, r, init_weights)
            self.glora_B[adapter_name] = GloraLoraPath(self.out_features, self.in_features, r, init_weights)
        elif config_A_B == "vector":
            self.glora_A[adapter_name] = GloraVectorPath(self.out_features, is_column=True, init_weights=init_weights)
            self.glora_B[adapter_name] = GloraVectorPath(self.out_features, is_column=True, init_weights=init_weights)
        elif config_A_B == "constant":
            self.glora_A[adapter_name] = GloraScalarPath(init_weights=init_weights)
            self.glora_B[adapter_name] = GloraScalarPath(init_weights=init_weights)
        else:
            self.glora_A[adapter_name] = GloraEmptyPath(shape=(self.out_features, self.in_features))
            self.glora_B[adapter_name] = GloraEmptyPath(shape=(self.out_features, self.in_features))

        base = self.get_base_layer()
        has_bias = getattr(base, "bias", None) is not None

        if has_bias and config_C == "lora":
            self.glora_C[adapter_name] = GloraLoraPath(self.in_features, 1, r, init_weights)
        elif has_bias and config_C == "vector":
            self.glora_C[adapter_name] = GloraVectorPath(self.in_features, is_column=True, init_weights=init_weights)
        else:
            self.glora_C[adapter_name] = GloraEmptyPath(shape=(self.in_features, 1))

        if has_bias and config_D_E == "vector":
            self.glora_D[adapter_name] = GloraVectorPath(self.out_features, is_column=False, init_weights=init_weights)
            self.glora_E[adapter_name] = GloraVectorPath(self.out_features, is_column=False, init_weights=init_weights)
        elif has_bias and config_D_E == "constant":
            self.glora_D[adapter_name] = GloraScalarPath(init_weights=init_weights)
            self.glora_E[adapter_name] = GloraScalarPath(init_weights=init_weights)
        else:
            self.glora_D[adapter_name] = GloraEmptyPath(shape=(self.out_features,))
            self.glora_E[adapter_name] = GloraEmptyPath(shape=(self.out_features,))

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=config.inference_mode)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        for adapter_name in adapter_names:
            if adapter_name not in self.glora_A:
                continue

            weight = self.weight
            bias = self.bias
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
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {adapter_name} seems to be broken"
                    )
                base_layer.weight.data = merged_weight
                if bias is not None:
                    merged_bias = b0 + b0 * D + E + torch.matmul(w0, C).squeeze(-1)
                    if not torch.isfinite(merged_bias).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {adapter_name} seems to be broken"
                        )
                    base_layer.bias.data = merged_bias
            else:
                base_layer.weight.data += (w0 * A) + B
                if bias is not None:
                    base_layer.bias.data += (b0 * D) + E + torch.matmul(w0, C).squeeze(-1)
            self.merged_adapters.append(adapter_name)

    def unmerge(self, adapter_names: Optional[list[str]] = None) -> None:
        if adapter_names is None:
            adapter_names = list(self.merged_adapters)
        for adapter_name in adapter_names:
            if adapter_name not in self.merged_adapters:
                continue
            if adapter_name not in self.glora_A:
                continue

            weight = self.weight
            bias = self.bias
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


class GloraLinear(nn.Module, GloraLayer):
    """GLoRA adapter wrapping a dense [`~torch.nn.Linear`] `base_layer`."""

    def __init__(self, base_layer: nn.Module, adapter_name: str, config, **kwargs) -> None:
        super().__init__()
        GloraLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config.r, config=config)

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
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
            if self.bias is not None:
                bias_eff = self.bias + self.bias * D + E + torch.matmul(self.weight, C).squeeze(-1)
                adapted = F.linear(sub_batch, weight_eff, bias=bias_eff)
            else:
                adapted = F.linear(sub_batch, weight_eff)
            result[idx] = result[idx] + (adapted - base_result[idx])
        return result

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
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
                if self.bias is not None:
                    bias_eff = self.bias + self.bias * D + E + torch.matmul(self.weight, C).squeeze(-1)
                    adapted = F.linear(x, weight_eff, bias=bias_eff)
                else:
                    adapted = F.linear(x, weight_eff)
                result = result + (adapted - base_result)
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        return "glora." + super().__repr__()
