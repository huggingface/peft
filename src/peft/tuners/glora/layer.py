# Copyright 2026-present the HuggingFace Inc. team.
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

from .config import GloraConfig


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

    def update_layer(self, adapter_name: str, r: int, config: GloraConfig, **kwargs) -> None:
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
        # we unmerge any previously merged adapters and re-merge all active adapters
        # check https://github.com/huggingface/peft/pull/3098#discussion_r3117924518
        # for details on why we do this instead of merging directly on top of already merged weights
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        # Unmerge any already-merged adapters so we start from the original W0,
        # then re-merge them all together. This avoids the sequential composition
        # problem where adapter 2's delta would be computed against the already-
        # modified weight instead of the original W0.
        previously_merged: list[str] = list(self.merged_adapters)
        if previously_merged:
            self.unmerge()
        all_adapters = previously_merged + [name for name in adapter_names if name in self.glora_A]

        base_layer = self.get_base_layer()
        weight = self.weight
        bias = self.bias
        device, dtype = weight.device, weight.dtype

        # we need a clone of the original weights to compute the merged weights
        # before they get mutated by the merge process
        w0 = weight.data.clone()
        b0 = bias.data.clone() if bias is not None else None

        for adapter_name in all_adapters:
            A = self.glora_A[adapter_name]().to(device=device, dtype=dtype)
            B = self.glora_B[adapter_name]().to(device=device, dtype=dtype)
            base_layer.weight.data += w0 * A + B
            if b0 is not None:
                C = self.glora_C[adapter_name]().to(device=device, dtype=dtype)
                D = self.glora_D[adapter_name]().to(device=device, dtype=dtype)
                E = self.glora_E[adapter_name]().to(device=device, dtype=dtype)
                base_layer.bias.data += b0 * D + E + torch.matmul(w0, C).squeeze(-1)

        if safe_merge:
            if not torch.isfinite(weight.data).all():
                raise ValueError("NaNs detected in the merged weights. The adapter seems to be broken")
            if bias is not None and not torch.isfinite(bias.data).all():
                raise ValueError("NaNs detected in the merged weights. The adapter seems to be broken")

        for adapter_name in all_adapters:
            self.merged_adapters.append(adapter_name)

    def unmerge(self, adapter_names: Optional[list[str]] = None) -> None:
        if adapter_names is None:
            adapter_names = list(self.merged_adapters)

        adapters_to_unmerge = [name for name in adapter_names if name in self.merged_adapters and name in self.glora_A]
        if not adapters_to_unmerge:
            return

        # Recover W0 by subtracting all merged adapter contributions.
        # W_merged = W0 * (1 + sum(Ai)) + sum(Bi), so:
        # W0 = (W_merged - sum(Bi)) / (1 + sum(Ai))
        base_layer = self.get_base_layer()
        weight = self.weight
        bias = self.bias
        device, dtype = weight.device, weight.dtype

        all_merged = [name for name in self.merged_adapters if name in self.glora_A]
        sum_A = torch.zeros_like(weight.data)
        sum_B = torch.zeros_like(weight.data)
        sum_D = torch.zeros_like(bias.data) if bias is not None else None
        sum_E = torch.zeros_like(bias.data) if bias is not None else None
        sum_WC = torch.zeros_like(bias.data) if bias is not None else None

        for adapter_name in all_merged:
            A = self.glora_A[adapter_name]().to(device=device, dtype=dtype)
            B = self.glora_B[adapter_name]().to(device=device, dtype=dtype)
            sum_A += A
            sum_B += B

        w0 = (weight.data - sum_B) / (1.0 + sum_A)

        if bias is not None:
            for adapter_name in all_merged:
                C = self.glora_C[adapter_name]().to(device=device, dtype=dtype)
                D = self.glora_D[adapter_name]().to(device=device, dtype=dtype)
                E = self.glora_E[adapter_name]().to(device=device, dtype=dtype)
                sum_D += D
                sum_E += E
                sum_WC += torch.matmul(w0, C).squeeze(-1)
            b0 = (bias.data - sum_E - sum_WC) / (1.0 + sum_D)

        # Set weights back to W0
        base_layer.weight.data = w0
        if bias is not None:
            base_layer.bias.data = b0

        for adapter_name in adapters_to_unmerge:
            self.merged_adapters.remove(adapter_name)

        # Re-merge the remaining adapters against W0
        remaining = [name for name in self.merged_adapters if name in self.glora_A]
        if remaining:
            # Clear merged_adapters and re-merge from scratch
            self.merged_adapters.clear()
            self.merge(adapter_names=remaining)


class GloraLinear(nn.Module, GloraLayer):
    """GLoRA adapter wrapping a dense [`~torch.nn.Linear`] `base_layer`."""

    def __init__(self, base_layer: nn.Module, adapter_name: str, config, **kwargs) -> None:
        super().__init__()
        GloraLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config.r, config=config)

    def _check_forward_args(self, x, *args, **kwargs):
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

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
        self._check_forward_args(x, *args, **kwargs)
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
            num_active_adapters = 0
            for active_adapter in self.active_adapters:
                if active_adapter not in self.glora_A:
                    continue
                num_active_adapters += 1
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
                result = result + adapted
            if num_active_adapters > 0:
                result = result - num_active_adapters * base_result
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        return "glora." + super().__repr__()
