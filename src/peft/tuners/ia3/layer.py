# coding=utf-8
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

import warnings
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import transpose


class IA3Layer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ["ia3_l"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_feedforward: bool,
    ):
        self.scaling = {}
        self.ia3_l = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.in_features = in_features
        self.out_features = out_features
        self.is_feedforward = is_feedforward

    def update_layer(self, adapter_name, init_ia3_weights):
        # Actual trainable parameters
        if self.is_feedforward:
            weight = torch.randn((1, self.in_features))
        else:
            weight = torch.randn((self.out_features, 1))
        self.ia3_l[adapter_name] = nn.Parameter(weight)
        if init_ia3_weights:
            self.reset_ia3_parameters(adapter_name)
        self.to(self.weight.device)
        self.set_adapter(self.active_adapters)

    def reset_ia3_parameters(self, adapter_name):
        if adapter_name in self.ia3_l.keys():
            # initialize learned vector with torch.ones
            nn.init.constant_(self.ia3_l[adapter_name], 1.0)


class Linear(nn.Linear, IA3Layer):
    # (IA)^3 implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_feedforward: bool = False,  # Set to True if the layer is treated as a feedforward layer
        is_target_conv_1d_layer: bool = False,  # whether target module is a conv1d layer. useful while unloading later
        **kwargs,
    ) -> None:
        init_ia3_weights = kwargs.pop("init_ia3_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        IA3Layer.__init__(self, in_features=in_features, out_features=out_features, is_feedforward=is_feedforward)
        self.is_feedforward = is_feedforward
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        self.is_target_conv_1d_layer = is_target_conv_1d_layer

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, init_ia3_weights)
        self.set_adapter(adapter_name)

    def update_layer(self, adapter_name, init_ia3_weights):
        # Actual trainable parameters
        if self.is_feedforward:
            weight = torch.randn((1, self.in_features))
        else:
            weight = torch.randn((self.out_features, 1))
        self.ia3_l[adapter_name] = nn.Parameter(weight)
        if init_ia3_weights:
            self.reset_ia3_parameters(adapter_name)
        self.to(self.weight.device)
        self.set_adapter(self.active_adapters)

    def merge(self, safe_merge: bool = False) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )

        for active_adapter in self.active_adapters:
            if active_adapter in self.ia3_l.keys():
                if safe_merge:
                    orig_weights = transpose(self.weight, self.fan_in_fan_out).clone()
                    orig_weights = torch.mul(orig_weights.data, self.ia3_l[active_adapter].data)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    self.weight.data = orig_weights
                    self.weight = transpose(self.weight, self.fan_in_fan_out)
                else:
                    self.weight = transpose(self.weight, self.fan_in_fan_out)
                    self.weight.data = torch.mul(self.weight.data, self.ia3_l[active_adapter].data)
                    self.weight = transpose(self.weight, self.fan_in_fan_out)

                if not self.is_feedforward and (self.bias is not None):
                    scaling = self.ia3_l[active_adapter].reshape(self.bias.shape)
                    self.bias.data = torch.mul(self.bias.data, scaling.data)

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        warnings.warn("Unmerge result can be inaccurate for (IA)^3.")
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.ia3_l.keys():
                self.weight = transpose(self.weight, self.fan_in_fan_out)
                # divide by (IA)^3 vector. Add tolerace to avoid division by zero
                self.weight.data = torch.div(self.weight.data, self.ia3_l[active_adapter].data + 1e-8)
                self.weight = transpose(self.weight, self.fan_in_fan_out)

                if not self.is_feedforward and (self.bias is not None):
                    scaling = self.ia3_l[active_adapter].reshape(self.bias.shape)
                    self.bias.data = torch.div(self.bias.data, scaling.data + 1e-8)

    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._linear(x)
        elif self.merged:
            result = self._linear(x)
        else:
            ia3_scaling = 1
            for active_adapter in self.active_adapters:
                if active_adapter not in self.ia3_l.keys():
                    continue
                dtype = self.ia3_l[active_adapter].dtype
                ia3_scaling *= self.ia3_l[active_adapter].flatten()

            if self.is_feedforward:
                x = x.to(dtype)
                # TODO: self.weight.dtype can be != self.ia3_l[self.active_adapters].dtype
                # e.g. bf16 vs fp32. Is that okay?
                interm = (x * ia3_scaling).to(self.weight.dtype)
                result = self._linear(interm)
            else:
                result = self._linear(x)
                result = result.to(dtype) * ia3_scaling

        result = result.to(previous_dtype)
        return result


class Conv2d(nn.Conv2d, IA3Layer):
    def __init__(
        self,
        adapter_name: str,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_feedforward: bool = False,  # Set to True if the layer is treated as a feedforward layer
        **kwargs,
    ) -> None:
        init_ia3_weights = kwargs.pop("init_ia3_weights", True)

        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        IA3Layer.__init__(self, in_features=in_channels, out_features=out_channels, is_feedforward=is_feedforward)
        self.is_feedforward = is_feedforward
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Conv2d.reset_parameters(self)
        self.update_layer(adapter_name, init_ia3_weights)
        self.set_adapter(adapter_name)

    def update_layer(self, adapter_name, init_ia3_weights):
        # Actual trainable parameters
        if self.is_feedforward:
            weight = torch.randn((1, self.in_features, 1, 1))
        else:
            weight = torch.randn((1, self.out_features, 1, 1))
        self.ia3_l[adapter_name] = nn.Parameter(weight)
        if init_ia3_weights:
            self.reset_ia3_parameters(adapter_name)
        self.to(self.weight.device)
        self.set_adapter(self.active_adapters)

    def merge(self, safe_merge: bool = False) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )

        for active_adapter in self.active_adapters:
            if active_adapter in self.ia3_l.keys():
                ia3_scaling = self.ia3_l[active_adapter].data
                if not self.is_feedforward:
                    ia3_scaling = ia3_scaling.permute(1, 0, 2, 3)

                if safe_merge:
                    output_weight = torch.mul(self.weight.data, ia3_scaling).clone()

                    if not torch.isfinite(output_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    self.weight.data = output_weight
                else:
                    self.weight.data = torch.mul(self.weight.data, ia3_scaling)

                if not self.is_feedforward and (self.bias is not None):
                    scaling = self.ia3_l[active_adapter].reshape(self.bias.shape)
                    self.bias.data = torch.mul(self.bias.data, scaling.data)

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        warnings.warn("Unmerge result can be inaccurate for (IA)^3.")
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.ia3_l.keys():
                # divide by (IA)^3 vector. Add tolerace to avoid division by zero
                ia3_scaling = self.ia3_l[active_adapter].data
                if not self.is_feedforward:
                    ia3_scaling = ia3_scaling.permute(1, 0, 2, 3)
                self.weight.data = torch.div(self.weight.data, ia3_scaling + 1e-8)

                if not self.is_feedforward and (self.bias is not None):
                    scaling = self.ia3_l[active_adapter].reshape(self.bias.shape)
                    self.bias.data = torch.mul(self.bias.data, scaling.data)

    def _conv2d(self, input: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            input,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._conv2d(x)
        elif self.merged:
            result = self._conv2d(x)
        else:
            ia3_scaling = 1
            for active_adapter in self.active_adapters:
                if active_adapter not in self.ia3_l.keys():
                    continue
                dtype = self.ia3_l[active_adapter].dtype
                ia3_scaling *= self.ia3_l[active_adapter]

            if self.is_feedforward:
                x = x.to(dtype)
                # TODO: self.weight.dtype can be != self.ia3_l[self.active_adapters].dtype
                # e.g. bf16 vs fp32. Is that okay?
                interm = (x * ia3_scaling).to(self.weight.dtype)
                result = self._conv2d(interm)
            else:
                result = self._conv2d(x)
                result = result.to(dtype) * ia3_scaling

        result = result.to(previous_dtype)
        return result
