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
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import transpose


class VeraLayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = [
        "vera_lambda_b",
        "vera_lambda_d",
    ]

    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.r = {}
        self.vera_dropout = nn.ModuleDict({})

        # For storing vector scale
        self.vera_lambda_b = nn.ParameterDict({})
        self.vera_lambda_d = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def _init_empty_weights(self, cls, *args, **kwargs) -> None:
        # A helper method that allows to initialize the layer of the given class without spending time to initialize the
        # model weights. The implementation is inspired by
        # https://pytorch.org/docs/stable/generated/torch.nn.utils.skip_init.html but this function cannot be used
        # directly.
        # Instead of this approach, it would be possible to bypass the __init__ of the class but that runs the risk of
        # omitting important logic inside that __init__.
        kwargs = kwargs.copy()
        final_device = kwargs.pop("device", "cpu")

        cls.__init__(self, *args, device="meta", **kwargs)
        self.to_empty(device=final_device)

    def update_layer(self, adapter_name, r, vera_dropout, init_vera_weights, d_initial: float = 1.0):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        if vera_dropout > 0.0:
            vera_dropout_layer = nn.Dropout(p=vera_dropout)
        else:
            vera_dropout_layer = nn.Identity()

        self.vera_dropout.update(nn.ModuleDict({adapter_name: vera_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.vera_lambda_b[adapter_name] = nn.Parameter(torch.ones(self.out_features), requires_grad=True)
            self.vera_lambda_d[adapter_name] = nn.Parameter(torch.ones(r), requires_grad=True)

        if init_vera_weights:
            self.reset_vera_parameters(adapter_name, d_initial=d_initial)

        weight = getattr(self, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

    # def update_layer_conv2d(self, adapter_name, r, vera_dropout, init_vera_weights, d_initial: float = 1.0):
    #     raise NotImplementedError("Not implemented for new memory efficient implementation yet.")
    #     if r <= 0:
    #         raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
    #     self.r[adapter_name] = r
    #     if vera_dropout > 0.0:
    #         vera_dropout_layer = nn.Dropout(p=vera_dropout)
    #     else:
    #         vera_dropout_layer = nn.Identity()

    #     self.vera_dropout[adapter_name] = vera_dropout_layer
    #     # Actual trainable parameters
    #     if r > 0:
    #         kernel_size = self.kwargs["kernel_size"]
    #         stride = self.kwargs["stride"]
    #         padding = self.kwargs["padding"]

    #         self.vera_lambda_b[adapter_name] = nn.Parameter(torch.zeros(self.out_features), requires_grad=True)
    #         self.vera_lambda_d[adapter_name] = nn.Parameter(torch.zeros(r), requires_grad=True)

    #     if init_vera_weights:
    #         self.reset_vera_parameters(adapter_name, d_initial=d_initial)

    #     weight = getattr(self, "weight", None)
    #     if weight is not None:
    #         # the layer is already completely initialized, this is an update
    #         self.to(self.weight.device, dtype=weight.dtype)

    def update_layer_embedding(self, adapter_name, r, vera_dropout, init_vera_weights, d_initial: float = 1.0):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        if vera_dropout > 0.0:
            vera_dropout_layer = nn.Dropout(p=vera_dropout)
        else:
            vera_dropout_layer = nn.Identity()

        self.vera_dropout[adapter_name] = vera_dropout_layer
        # Actual trainable parameters
        if r > 0:
            self.vera_lambda_b[adapter_name] = nn.Parameter(torch.ones(self.out_features), requires_grad=True)
            self.vera_lambda_d[adapter_name] = nn.Parameter(torch.ones(r), requires_grad=True)

        if init_vera_weights:
            self.reset_vera_parameters(adapter_name, d_initial=d_initial)

        weight = getattr(self, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            self.to(self.weight.device, dtype=weight.dtype)

    def reset_vera_parameters(self, adapter_name, d_initial: float = 1.0):
        if adapter_name in self.vera_lambda_d.keys():
            with torch.no_grad():
                nn.init.zeros_(self.vera_lambda_d[adapter_name]).fill_(d_initial)
                nn.init.zeros_(self.vera_lambda_b[adapter_name])


# Below was based on 'src/peft/tuners/lora/layer.py
# Which was in turn based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Linear, VeraLayer):
    # Vera implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        vera_dropout: float = 0.0,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        d_initial: float = 1.0,
        **kwargs,
    ) -> None:
        init_vera_weights = kwargs.pop("init_vera_weights", True)
        # this gets the init from nn.Linear's super perspective, i.e.
        # nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        # Note that we don't use self._init_empty_weights() for Linear because it is a bit slower and the benefit of
        # added robustness is not big enough for Linear.

        VeraLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix

        self.fan_in_fan_out = fan_in_fan_out

        self.update_layer(adapter_name, r, vera_dropout, init_vera_weights, d_initial=d_initial)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self.set_adapter(adapter_name)

    def merge(self, vera_A: torch.Tensor, vera_B: torch.Tensor, safe_merge: bool = False) -> None:
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
            if active_adapter in self.vera_lambda_d.keys():
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = self.weight.data.clone()

                    orig_weights += self.get_delta_weight(active_adapter, vera_A, vera_B)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    self.weight.data = orig_weights
                else:
                    self.weight.data += self.get_delta_weight(active_adapter, vera_A, vera_B)
                self.merged_adapters.append(active_adapter)

    def unmerge(self, vera_A: torch.Tensor, vera_B: torch.Tensor) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.vera_lambda_d.keys():

                self.weight.data -= self.get_delta_weight(active_adapter, vera_A, vera_B)

    def get_delta_weight(self, adapter, vera_A: torch.Tensor, vera_B: torch.Tensor) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = vera_B.device
        dtype = vera_B.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        lambda_d = self.vera_lambda_d[adapter]
        lambda_b = self.vera_lambda_b[adapter]

        if cast_to_fp32:
            vera_A = vera_A.float()
            vera_B = vera_B.float()
            lambda_d = lambda_d.float()
            lambda_b = lambda_b.float()

        lambda_b = lambda_b.unsqueeze(-1)
        lambda_d = lambda_d.unsqueeze(-1)
        output_tensor = transpose((lambda_b * vera_B) @ (lambda_d * vera_A), self.fan_in_fan_out)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            # TODO: why?
            self.vera_lambda_d[adapter].data = lambda_d.to(dtype)
            self.vera_lambda_b[adapter].data = lambda_b.to(dtype)

        return output_tensor

    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    # we use a forward hook to populate vera_A and vera_B
    # this indirectly enforces the constraint of only supporting a single vera shape for now.
    def forward(self, x: torch.Tensor, vera_A: torch.Tensor, vera_B: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._linear(x)
        elif self.merged:
            result = self._linear(x)
        else:
            result = self._linear(x)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.vera_lambda_d.keys():
                    continue

                lambda_d = self.vera_lambda_d[active_adapter]
                lambda_b = self.vera_lambda_b[active_adapter]

                dropout = self.vera_dropout[active_adapter]
                x = x.to(lambda_d.dtype)
                result += lambda_b * F.linear(lambda_d * F.linear(dropout(x), vera_A), vera_B)

        result = result.to(previous_dtype)
        return result


class Embedding(nn.Embedding, VeraLayer):
    # Vera implemented in a Embedding layer
    def __init__(
        self,
        adapter_name: str,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        vera_dropout: float = 0.0,
        d_initial: float = 1.0,
        **kwargs,
    ) -> None:
        init_vera_weights = kwargs.pop("init_vera_weights", True)
        self._init_empty_weights(nn.Embedding, num_embeddings, embedding_dim, **kwargs)
        VeraLayer.__init__(self, in_features=num_embeddings, out_features=embedding_dim)
        self.update_layer_embedding(adapter_name, r, vera_dropout, init_vera_weights, d_initial=d_initial)
        self.set_adapter(adapter_name)

    def merge(self, vera_A: torch.Tensor, vera_B: torch.Tensor, safe_merge: bool = False) -> None:
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
            if active_adapter in self.vera_lambda_d.keys():
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = self.weight.data.copy()
                    orig_weights += self.get_delta_weight(active_adapter, vera_A, vera_B)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    self.weight.data = orig_weights
                else:
                    self.weight.data += self.get_delta_weight(active_adapter, vera_A, vera_B)
                self.merged_adapters.append(active_adapter)

    def unmerge(self, vera_A: torch.Tensor, vera_B: torch.Tensor) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.vera_lambda_d.keys():
                self.weight.data -= self.get_delta_weight(active_adapter, vera_A, vera_B)

    def get_delta_weight(self, adapter, vera_A: torch.Tensor, vera_B: torch.Tensor) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = vera_A.device
        dtype = vera_A.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        lambda_d = self.vera_lambda_d[adapter]
        lambda_b = self.vera_lambda_b[adapter]

        if cast_to_fp32:
            vera_A = vera_A.float()
            vera_B = vera_B.float()
            lambda_d = lambda_d.float()
            lambda_b = lambda_b.float()

        lambda_b = lambda_b.unsqueeze(-1)
        lambda_d = lambda_d.unsqueeze(-1)
        output_tensor = transpose((lambda_b * vera_B) @ (lambda_d * vera_A), True)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.vera_lambda_d[adapter].data = lambda_d.to(dtype)
            self.vera_lambda_b[adapter].data = lambda_b.to(dtype)

        return output_tensor

    def _embed(self, input: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = self.weight if weight is None else weight
        return F.embedding(
            input,
            weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

    def forward(self, x: torch.Tensor, vera_A: torch.Tensor, vera_B: torch.Tensor) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._embed(x)
        elif self.merged:
            result = self._embed(x)
        else:
            result = self._embed(x)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.vera_lambda_d:
                    continue
                lambda_d = self.vera_lambda_d[active_adapter]
                lambda_b = self.vera_lambda_b[active_adapter]

                after_A = lambda_d * self._embed(x, vera_A.T)
                result += lambda_b * (after_A @ vera_B.T)

        return result


class Conv2d(nn.Conv2d, VeraLayer):
    # Vera implemented in a conv2d layer
    def __init__(
        self,
        adapter_name: str,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        r: int = 0,
        vera_dropout: float = 0.0,
        d_initial: float = 1.0,
        **kwargs,
    ) -> None:
        raise NotImplementedError("Not implemented for new memory efficient implementation yet.")
        init_vera_weights = kwargs.pop("init_vera_weights", True)
        self._init_empty_weights(nn.Conv2d, in_channels, out_channels, kernel_size, stride=stride, padding=padding)

        VeraLayer.__init__(
            self,
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.update_layer_conv2d(adapter_name, r, vera_dropout, init_vera_weights, d_initial=d_initial)
        self.set_adapter(adapter_name)

    def merge(self, vera_A: torch.Tensor, vera_B: torch.Tensor, safe_merge: bool = False) -> None:
        """
Merge the active adapter weights inside the base weights

Args: # safe_merge (`bool`, *optional*): # If True, the merge operation will be performed in a copy of the original
weights and check for NaNs # before merging the weights. This is useful if you want to check if the merge operation
will produce # NaNs. Defaults to `False`. #"""
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )
        for active_adapter in self.active_adapters:
            if active_adapter in self.vera_A.keys():
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = self.weight.data.copy()
                    orig_weights += self.get_delta_weight(active_adapter, vera_A, vera_B)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    self.weight.data = orig_weights
                else:
                    self.weight.data += self.get_delta_weight(active_adapter, vera_A, vera_B)
                self.merged_adapters.append(active_adapter)

    def unmerge(self, vera_A: torch.Tensor, vera_B: torch.Tensor) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.vera_A.keys():
                self.weight.data -= self.get_delta_weight(active_adapter, vera_A, vera_B)

    def get_delta_weight(self, adapter, vera_A: torch.Tensor, vera_B: torch.Tensor) -> torch.Tensor:
        """
Compute the delta weight for the given adapter.

Args: # adapter (str): # The name of the adapter for which the delta weight should be computed. #"""
        device = vera_B.device
        dtype = vera_B.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        lambda_d = self.vera_lambda_d[adapter]
        lambda_b = self.vera_lambda_b[adapter]

        if cast_to_fp32:
            lambda_d = lambda_d.float()
            lambda_b = lambda_b.float()

        # lambda_b = lambda_b.unsqueeze(-1)
        # lambda_d = lambda_d.unsqueeze(-1)

        # lambda_d = torch.diag(lambda_d)
        # lambda_b = torch.diag(lambda_b)

        # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/lora.py#L117
        if self.weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            output_tensor = (lambda_b.unsqueeze(-1) * vera_B.squeeze(3).squeeze(2)) @ (
                lambda_d.unsqueeze(-1) * weight_A.squeeze(3).squeeze(2)
            ).unsqueeze(2).unsqueeze(3)
        else:
            # conv2d 3x3
            output_tensor = F.conv2d(
                lambda_d.unsqueeze(0) * weight_A.permute(1, 0, 2, 3),
                lambda_b.unsqueeze(-1) * weight_B,
            ).permute(1, 0, 2, 3)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.vera_A[adapter].weight.data = weight_A.to(dtype)
            self.vera_B[adapter].weight.data = weight_B.to(dtype)
            self.vera_lambda_d[adapter].data = torch.diag(lambda_d).to(dtype)
            self.vera_lambda_b[adapter].data = torch.diag(lambda_b).to(dtype)

        return output_tensor

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

    def forward(self, x: torch.Tensor, vera_A: torch.Tensor, vera_B: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._conv2d(x)
        elif self.merged:
            result = self._conv2d(x)
        else:
            result = self._conv2d(x)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.vera_lambda_d.keys():
                    continue

                lambda_d = self.vera_lambda_d[active_adapter]
                lambda_b = self.vera_lambda_b[active_adapter]

                dropout = self.vera_dropout[active_adapter]
                x = x.to(vera_A.dtype)

                conv_kwargs = dict(stride=self.stride, padding=self.padding)
                result += lambda_b * F.conv2d(
                    lambda_d * F.conv2d(dropout(x), vera_A.unsqueeze(-1).unsqueeze(-1), **conv_kwargs),
                    vera_B.unsqueeze(-1).unsqueeze(-1),
                    **conv_kwargs,
                )

        result = result.to(previous_dtype)
        return result
