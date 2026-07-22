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

import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, _get_in_out_features, check_adapters_to_merge
from peft.utils.other import transpose


class UniLoraLayer(BaseTunerLayer):
    other_param_names = (
        "r",
        "unilora_theta_d",
        "unilora_indices_A",
        "unilora_indices_B",
        "unilora_scales_A",
        "unilora_scales_B",
        "unilora_dropout",
    )

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.unilora_dropout = nn.ModuleDict({})
        self.unilora_indices_A = BufferDict({}, persistent=True)
        self.unilora_indices_B = BufferDict({}, persistent=True)
        self.unilora_scales_A = BufferDict({}, persistent=True)
        self.unilora_scales_B = BufferDict({}, persistent=True)

        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        in_features, out_features = _get_in_out_features(base_layer)
        if (in_features is None) or (out_features is None):
            raise ValueError(
                f"Could not infer input and output features for target module {base_layer}. "
                "UniLora currently supports linear-like modules with known in/out features."
            )

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name: str,
        unilora_theta_d,
        r: int,
        theta_d_length: int,
        unilora_dropout: float = 0.0,
    ):
        if r <= 0:
            raise ValueError(f"`r` {r} should be a positive integer value")

        self.r[adapter_name] = r

        if unilora_dropout > 0.0:
            unilora_dropout_layer = nn.Dropout(p=unilora_dropout)
        else:
            unilora_dropout_layer = nn.Identity()
        self.unilora_dropout.update(nn.ModuleDict({adapter_name: unilora_dropout_layer}))

        self.unilora_theta_d = unilora_theta_d
        self.reset_unilora_parameters(adapter_name, theta_d_length)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_unilora_parameters(self, adapter_name: str, theta_d_length: int) -> None:
        if adapter_name in self.unilora_theta_d.keys():
            indices_A = torch.randint(0, theta_d_length, (self.r[adapter_name], self.in_features), dtype=torch.long)
            indices_B = torch.randint(0, theta_d_length, (self.out_features, self.r[adapter_name]), dtype=torch.long)

            self.unilora_indices_A[adapter_name] = indices_A
            self.unilora_indices_B[adapter_name] = indices_B

    def update_scaling(
        self,
        adapter_name: str,
        unilora_scales_A,
        unilora_scales_B,
    ):
        if adapter_name in self.unilora_theta_d.keys():
            base_layer = self.get_base_layer()
            target_device = base_layer.weight.device
            target_dtype = base_layer.weight.dtype

            self.unilora_scales_A[adapter_name] = unilora_scales_A.to(device=target_device, dtype=target_dtype)
            self.unilora_scales_B[adapter_name] = unilora_scales_B.to(device=target_device, dtype=target_dtype)
            self._move_indices_to_device(adapter_name, target_device)

    def _move_indices_to_device(self, adapter_name: str, device: torch.device) -> None:
        for index_buffer in (self.unilora_indices_A, self.unilora_indices_B):
            if adapter_name in index_buffer:
                index_buffer[adapter_name] = index_buffer[adapter_name].to(device=device)

    def _move_adapter_to_device_of_base_layer(self, adapter_name: str, device: Optional[torch.device] = None) -> None:
        super()._move_adapter_to_device_of_base_layer(adapter_name, device=device)
        base_layer = self.get_base_layer()
        target_device = device or getattr(base_layer.weight, "device", None)
        if target_device is not None:
            self._move_indices_to_device(adapter_name, target_device)

    def enable_adapters(self, enabled: bool) -> None:
        super().enable_adapters(enabled)
        if not enabled:
            for theta_d in self.unilora_theta_d.values():
                theta_d.requires_grad_(False)

    def set_adapter(self, adapter_names: str | list[str], inference_mode: bool = False) -> None:
        super().set_adapter(adapter_names, inference_mode=inference_mode)
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        for adapter_name, theta_d in self.unilora_theta_d.items():
            theta_d.requires_grad_(adapter_name in adapter_names and not inference_mode)


class Linear(nn.Linear, UniLoraLayer):
    def __init__(
        self,
        base_layer,
        unilora_theta_d,
        adapter_name: str,
        r: int,
        theta_d_length: int,
        unilora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        UniLoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            unilora_theta_d,
            r,
            theta_d_length,
            unilora_dropout,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def supports_lora_conversion(self, adapter_name: str = "default") -> bool:
        return True

    def get_additive_delta(self, adapter_name: str = "default") -> torch.Tensor:
        return self.get_delta_weight(adapter_name)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.unilora_indices_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    if not torch.isfinite(self.unilora_theta_d[active_adapter]).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.unilora_indices_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def _get_lora_matrices(self, adapter: str, cast_to_fp32: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        unilora_indices_A = self.unilora_indices_A[adapter]
        unilora_indices_B = self.unilora_indices_B[adapter]
        unilora_theta_d = self.unilora_theta_d[adapter].to(unilora_indices_A.device)
        scales_A = self.unilora_scales_A[adapter].to(unilora_indices_A.device)
        scales_B = self.unilora_scales_B[adapter].to(unilora_indices_B.device)

        if cast_to_fp32:
            unilora_theta_d = unilora_theta_d.float()
            scales_A = scales_A.float()
            scales_B = scales_B.float()

        A = unilora_theta_d[unilora_indices_A] * scales_A
        B = unilora_theta_d[unilora_indices_B] * scales_B

        return A, B

    def get_delta_weight(self, adapter) -> torch.Tensor:
        device = self.unilora_indices_A[adapter].device
        dtype = self.unilora_theta_d[adapter].dtype

        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)
        A, B = self._get_lora_matrices(adapter, cast_to_fp32)
        output_tensor = transpose(B @ A, self.fan_in_fan_out)
        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)
        return output_tensor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.unilora_indices_A.keys():
                    continue

                A, B = self._get_lora_matrices(active_adapter)
                x = x.to(self.unilora_theta_d[active_adapter].dtype)
                dropout = self.unilora_dropout[active_adapter]

                result = result + F.linear(F.linear(dropout(x), A), B)

        result = result.to(previous_dtype)
        return result
