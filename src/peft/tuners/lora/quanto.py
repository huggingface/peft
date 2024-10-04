# Copyright 2024-present the HuggingFace Inc. team.
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
from torch.nn import functional as F

from peft.import_utils import is_quanto_available
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose


if is_quanto_available:
    # ensure that there are no quanto imports unless optimum.quanto is installed
    from optimum.quanto import QConv2d, QLinear
else:
    QConv2d, QLinear = None, None


class QuantoLoraLinear(torch.nn.Module, LoraLayer):
    """LoRA layer implementation for quanto QLinear"""

    def __init__(
        self,
        base_layer,
        adapter_name,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        init_lora_weights: bool = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ):
        if use_dora:
            raise ValueError(f"{self.__class__.__name__} does not support DoRA yet, please set it to False")

        super().__init__()
        LoraLayer.__init__(self, base_layer)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        result = self.base_layer(x)
        adapter_names = kwargs.pop("adapter_names", None)
        if adapter_names is not None:
            raise ValueError(f"{self.__class__.__name__} does not support mixed_batch_forward yet.")

        if self.disable_adapters:
            return result

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    x = x.to(lora_A.weight.dtype)

                output = lora_B(lora_A(dropout(x)))
                if requires_conversion:
                    output = output.to(expected_dtype)
                output = output * scaling
                result = result + output

        return result

    def get_delta_weight(self, adapter):
        return (
            transpose(self.lora_B[adapter].weight @ self.lora_A[adapter].weight, fan_in_fan_out=self.fan_in_fan_out)
            * self.scaling[adapter]
        )

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        from optimum.quanto import quantize_weight

        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        base_layer = self.get_base_layer()
        orig_weight = base_layer.weight

        for active_adapter in adapter_names:
            delta_weight = self.get_delta_weight(active_adapter)
            # note: no in-place for safe_merge=False
            new_weight_data = orig_weight + delta_weight
            if safe_merge and not torch.isfinite(new_weight_data).all():
                raise ValueError(
                    f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                )
            quantized = quantize_weight(new_weight_data, qtype=base_layer.qweight.qtype, axis=base_layer.qweight.axis)
            base_layer.weight._data = quantized._data
            base_layer.weight._scale = quantized._scale
            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        from optimum.quanto import quantize_weight

        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self.lora_A.keys():
                continue

            base_layer = self.get_base_layer()
            orig_weight = base_layer.weight
            new_weight_data = orig_weight - self.get_delta_weight(active_adapter)
            quantized = quantize_weight(new_weight_data, qtype=base_layer.qweight.qtype, axis=base_layer.qweight.axis)
            base_layer.weight._data = quantized._data
            base_layer.weight._scale = quantized._scale

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class QuantoLoraConv2d(torch.nn.Module, LoraLayer):
    """LoRA layer implementation for quanto QConv2d"""

    def __init__(
        self,
        base_layer,
        adapter_name,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ):
        if use_dora:
            raise ValueError(f"{self.__class__.__name__} does not support DoRA yet, please set it to False")

        super().__init__()
        LoraLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora)

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora):
        # same as lora.layer.Conv2d
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        base_layer = self.get_base_layer()
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        self.lora_A[adapter_name] = nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)
        self.lora_B[adapter_name] = nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            # TODO: Implement DoRA
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        result = self.base_layer(x)
        adapter_names = kwargs.pop("adapter_names", None)
        if adapter_names is not None:
            raise ValueError(f"{self.__class__.__name__} does not support mixed_batch_forward yet.")

        if self.disable_adapters:
            return result

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    x = x.to(lora_A.weight.dtype)

                output = lora_B(lora_A(dropout(x)))
                if requires_conversion:
                    output = output.to(expected_dtype)
                output = output * scaling
                result = result + output

        return result

    def get_delta_weight(self, adapter):
        # same as lora.layer.Conv2d
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_A[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/lora.py#L117
        if self.get_base_layer().weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(
                3
            ) * self.scaling[adapter]
        else:
            # conv2d 3x3
            output_tensor = (
                F.conv2d(
                    weight_A.permute(1, 0, 2, 3),
                    weight_B,
                ).permute(1, 0, 2, 3)
                * self.scaling[adapter]
            )

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        # same as lora.quanto.QuantoLoraLinear
        from optimum.quanto import quantize_weight

        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        base_layer = self.get_base_layer()
        orig_weight = base_layer.weight

        for active_adapter in adapter_names:
            delta_weight = self.get_delta_weight(active_adapter)
            # note: no in-place for safe_merge=False
            new_weight_data = orig_weight + delta_weight
            if safe_merge:
                if torch.isfinite(new_weight_data).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )
            quantized = quantize_weight(new_weight_data, qtype=orig_weight.qtype, axis=orig_weight.axis)
            base_layer.weight._data = quantized._data
            base_layer.weight._scale = quantized._scale
            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        # same as lora.quanto.QuantoLoraLinear
        from optimum.quanto import quantize_weight

        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self.lora_A.keys():
                continue

            base_layer = self.get_base_layer()
            orig_weight = base_layer.weight
            new_weight_data = orig_weight - self.get_delta_weight(active_adapter)
            quantized = quantize_weight(new_weight_data, qtype=orig_weight.qtype, axis=orig_weight.axis)
            base_layer.weight._data = quantized._data
            base_layer.weight._scale = quantized._scale

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


def dispatch_quanto(
    target: torch.nn.Module,
    adapter_name: str,
    **kwargs: Any,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if is_quanto_available() and isinstance(target_base_layer, QLinear):
        new_module = QuantoLoraLinear(target, adapter_name, **kwargs)
        target.weight = target_base_layer.weight

        if hasattr(target, "bias"):
            target.bias = target_base_layer.bias
    elif is_quanto_available() and isinstance(target_base_layer, QConv2d):
        new_module = QuantoLoraConv2d(target, adapter_name, **kwargs)
        target.weight = target_base_layer.weight

        if hasattr(target, "bias"):
            target.bias = target_base_layer.bias

    return new_module
