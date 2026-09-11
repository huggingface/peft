# Copyright 2023-present the HuggingFace Inc. team.
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
import warnings
from typing import Any, Optional

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils import transpose

from .config import FineGatesConfig


SQRT_2 = math.sqrt(2.0)


class FineGatesLayer(BaseTunerLayer):
    adapter_layer_names: tuple[str, ...] = ("finegates_rows", "finegates_columns")
    other_param_names: tuple[str, ...] = (
        "target_sparsity",
        "sparsity_loss_weight",
        "gate_noise_std",
        "gate_init_mean",
        "gate_init_std",
        "init_weights",
    )

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.finegates_rows = nn.ParameterDict({})
        self.finegates_columns = nn.ParameterDict({})
        self.target_sparsity = {}
        self.sparsity_loss_weight = {}
        self.gate_noise_std = {}
        self.gate_init_mean = {}
        self.gate_init_std = {}
        self.init_weights = {}
        self._disable_adapters = False
        self.merged_adapters = []
        self._merge_backup_stack: list[tuple[str, torch.Tensor, Optional[torch.Tensor]]] = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise TypeError(
                f"Unsupported layer type '{type(base_layer)}' encountered, cannot apply FineGates adapter."
            )

        self.in_features = in_features
        self.out_features = out_features

    @property
    def _available_adapters(self) -> set[str]:
        return {*self.finegates_rows}

    def update_layer(self, adapter_name: str, config: FineGatesConfig, inference_mode: bool = False):
        self.target_sparsity[adapter_name] = config.target_sparsity
        self.sparsity_loss_weight[adapter_name] = config.sparsity_loss_weight
        self.gate_noise_std[adapter_name] = config.gate_noise_std
        self.gate_init_mean[adapter_name] = config.gate_init_mean
        self.gate_init_std[adapter_name] = config.gate_init_std
        self.init_weights[adapter_name] = config.init_weights

        self.finegates_rows[adapter_name] = nn.Parameter(torch.empty(self.out_features))
        self.finegates_columns[adapter_name] = nn.Parameter(torch.empty(self.in_features))
        self.reset_parameters(adapter_name)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def reset_parameters(self, adapter_name: str):
        if self.init_weights[adapter_name] is False:
            nn.init.normal_(
                self.finegates_rows[adapter_name].data,
                mean=0.0,
                std=self.gate_init_std[adapter_name],
            )
            nn.init.normal_(
                self.finegates_columns[adapter_name].data,
                mean=0.0,
                std=self.gate_init_std[adapter_name],
            )
            return

        nn.init.normal_(
            self.finegates_rows[adapter_name].data,
            mean=self.gate_init_mean[adapter_name],
            std=self.gate_init_std[adapter_name],
        )
        nn.init.normal_(
            self.finegates_columns[adapter_name].data,
            mean=self.gate_init_mean[adapter_name],
            std=self.gate_init_std[adapter_name],
        )

    @staticmethod
    def _hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x + 0.5, 0.0, 1.0)

    def _get_eval_gates(self, adapter_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        row_mu = torch.tanh(self.finegates_rows[adapter_name])
        col_mu = torch.tanh(self.finegates_columns[adapter_name])
        return self._hard_sigmoid(row_mu), self._hard_sigmoid(col_mu)

    def _get_runtime_gates(
        self, adapter_name: str, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = self.finegates_rows[adapter_name]
        cols = self.finegates_columns[adapter_name]
        row_mu = torch.tanh(rows)
        col_mu = torch.tanh(cols)

        if self.training and (self.gate_noise_std[adapter_name] > 0):
            row_noise = torch.randn_like(row_mu) * self.gate_noise_std[adapter_name]
            col_noise = torch.randn_like(col_mu) * self.gate_noise_std[adapter_name]
            row_gates = self._hard_sigmoid(row_mu + row_noise)
            col_gates = self._hard_sigmoid(col_mu + col_noise)
        else:
            row_gates = self._hard_sigmoid(row_mu)
            col_gates = self._hard_sigmoid(col_mu)

        return row_gates.to(device=device, dtype=dtype), col_gates.to(device=device, dtype=dtype)

    def get_sparsity_loss(self, adapter_name: str) -> torch.Tensor:
        row_mu = torch.tanh(self.finegates_rows[adapter_name])
        col_mu = torch.tanh(self.finegates_columns[adapter_name])
        row_loss = 0.5 - 0.5 * torch.erf((-0.5 - row_mu) / (0.5 * SQRT_2))
        col_loss = 0.5 - 0.5 * torch.erf((-0.5 - col_mu) / (0.5 * SQRT_2))
        target_keep = 1.0 - self.target_sparsity[adapter_name]
        means = torch.stack((row_loss.mean(), col_loss.mean()))
        return (means - target_keep).abs().mean() * self.sparsity_loss_weight[adapter_name]

    def get_compression_statistics(self, adapter_name: str) -> dict[str, float]:
        row_gates, col_gates = self._get_eval_gates(adapter_name)
        active_rows = int((row_gates > 0).sum().item())
        active_cols = int((col_gates > 0).sum().item())
        total_params = self.out_features * self.in_features
        active_params = active_rows * active_cols
        return {
            "active_rows": active_rows,
            "active_columns": active_cols,
            "total_rows": self.out_features,
            "total_columns": self.in_features,
            "active_params": active_params,
            "pruned_params": total_params - active_params,
            "param_sparsity": 1.0 - (active_params / total_params),
        }


class Linear(nn.Module, FineGatesLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        config: FineGatesConfig,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        FineGatesLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = config.fan_in_fan_out
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config=config)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)

        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        result = x
        base_dtype = x.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self._available_adapters:
                continue
            row_gates, col_gates = self._get_runtime_gates(active_adapter, device=result.device, dtype=result.dtype)
            result = result * col_gates
            result = self.base_layer(result, *args, **kwargs)
            result = result * row_gates
            result = result.to(base_dtype)
            return result

        return self.base_layer(x, *args, **kwargs)

    def _merged_weight_and_bias(self, adapter_name: str) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        base_layer = self.get_base_layer()
        row_gates, col_gates = self._get_eval_gates(adapter_name)
        weight = transpose(base_layer.weight.data, self.fan_in_fan_out)
        merged_weight = row_gates[:, None].to(weight.dtype) * weight * col_gates[None, :].to(weight.dtype)
        merged_weight = transpose(merged_weight, self.fan_in_fan_out)

        merged_bias = None
        if base_layer.bias is not None:
            merged_bias = row_gates.to(base_layer.bias.dtype) * base_layer.bias.data

        return merged_weight, merged_bias

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        base_layer = self.get_base_layer()
        for active_adapter in adapter_names:
            if active_adapter not in self._available_adapters:
                continue

            merged_weight, merged_bias = self._merged_weight_and_bias(active_adapter)
            if safe_merge:
                if not torch.isfinite(merged_weight).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )
                if (merged_bias is not None) and (not torch.isfinite(merged_bias).all()):
                    raise ValueError(
                        f"NaNs detected in the merged bias. The adapter {active_adapter} seems to be broken"
                    )

            backup_bias = None if base_layer.bias is None else base_layer.bias.data.detach().clone()
            self._merge_backup_stack.append((active_adapter, base_layer.weight.data.detach().clone(), backup_bias))

            base_layer.weight.data = merged_weight.to(base_layer.weight.dtype).contiguous()
            if merged_bias is not None:
                base_layer.bias.data = merged_bias.to(base_layer.bias.dtype).contiguous()
            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        base_layer = self.get_base_layer()
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self._available_adapters:
                continue
            if not self._merge_backup_stack:
                raise RuntimeError("Cannot unmerge FineGates adapter because no merge backup is available.")

            backup_adapter, backup_weight, backup_bias = self._merge_backup_stack.pop()
            if backup_adapter != active_adapter:
                raise RuntimeError(
                    f"FineGates merge backup stack is inconsistent: expected {active_adapter}, got {backup_adapter}."
                )

            base_layer.weight.data = backup_weight.to(base_layer.weight.dtype).contiguous()
            if base_layer.bias is not None and backup_bias is not None:
                base_layer.bias.data = backup_bias.to(base_layer.bias.dtype).contiguous()

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "finegates." + rep


def dispatch_default(
    target: nn.Module,
    adapter_name: str,
    finegates_config: FineGatesConfig,
    **kwargs,
) -> Optional[nn.Module]:
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, nn.Linear):
        return Linear(target, adapter_name, config=finegates_config, **kwargs)

    if isinstance(target_base_layer, Conv1D):
        return Linear(target, adapter_name, config=finegates_config, is_target_conv_1d_layer=True, **kwargs)

    return None
