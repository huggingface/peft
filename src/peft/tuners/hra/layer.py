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

import math
import warnings
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


class HRALayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("hra_u",)
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("hra_r", "hra_apply_GS")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.hra_r = {}
        self.hra_apply_GS = {}
        self.hra_u = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            self.in_features, self.out_features = base_layer.in_channels, base_layer.out_channels
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        apply_GS: bool,
        init_weights: bool,
        **kwargs,
    ) -> None:
        """Internal function to create hra adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            init_weights (`bool`): Whether to initialize weights.
            apply_GS (`bool`): Whether to apply Gram-Schmidt orthogonalization or not.
        """
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.hra_r[adapter_name] = r
        self.hra_apply_GS[adapter_name] = apply_GS

        # Determine shape of HRA weights
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.hra_u[adapter_name] = nn.Parameter(torch.empty(self.in_features, r), requires_grad=True)
        elif isinstance(base_layer, nn.Conv2d):
            self.hra_u[adapter_name] = nn.Parameter(
                torch.empty(self.in_features * base_layer.kernel_size[0] * base_layer.kernel_size[0], r),
                requires_grad=True,
            )
        else:
            raise TypeError(f"HRA is not implemented for base layers of type {type(base_layer).__name__}")

        # Initialize weights
        if init_weights:
            self.reset_hra_parameters(adapter_name)
        else:
            self.reset_hra_parameters_random(adapter_name)

        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_hra_parameters(self, adapter_name: str):
        if self.hra_r[adapter_name] % 2 != 0:
            warnings.warn("The symmetric initialization can NOT be performed when r is odd!")
            nn.init.kaiming_uniform_(self.hra_u[adapter_name], a=math.sqrt(5))
        else:
            shape = self.hra_u[adapter_name].shape
            half_u = torch.zeros(shape[0], shape[1] // 2)
            nn.init.kaiming_uniform_(half_u, a=math.sqrt(5))
            self.hra_u[adapter_name] = nn.Parameter(torch.repeat_interleave(half_u, 2, dim=1))

    def reset_hra_parameters_random(self, adapter_name: str):
        nn.init.kaiming_uniform_(self.hra_u[adapter_name], a=math.sqrt(5))

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.hra_u.keys():
                continue

            warnings.warn("Scaling operation for HRA not supported! Automatically set scale to 1.")

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.hra_u.keys():
                continue

            warnings.warn("Unscaling operation for HRA not supported! Keeping scale at 1.")


class HRALinear(nn.Module, HRALayer):
    """
    HRA implemented in a dense layer.
    """

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        apply_GS: bool = False,
        init_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        HRALayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, apply_GS, init_weights, **kwargs)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If `True`, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If `None`, all active adapters will be merged.
                Defaults to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.hra_u.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weight = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weight = torch.mm(orig_weight, delta_weight)

                    if not torch.isfinite(orig_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    self.base_layer.weight.data = orig_weight
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    self.base_layer.weight.data = torch.mm(self.base_layer.weight.data, delta_weight)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.hra_u.keys():
                orig_weight = self.get_base_layer().weight.data.clone()
                delta_weight = self.get_delta_weight(active_adapter, reverse=True)
                self.get_base_layer().weight.data = torch.mm(orig_weight, delta_weight)

    def get_delta_weight(self, adapter_name: str, reverse: bool = False) -> torch.Tensor:
        rank = self.hra_r[adapter_name]
        apply_GS = self.hra_apply_GS[adapter_name]
        opt_u = self.hra_u[adapter_name]
        shape = opt_u.shape

        if apply_GS:
            weight = [(opt_u[:, 0] / opt_u[:, 0].norm()).view(-1, 1)]
            for i in range(1, rank):
                ui = opt_u[:, i].view(-1, 1)
                for j in range(i):
                    ui = ui - (weight[j].t() @ ui) * weight[j]
                weight.append((ui / ui.norm()).view(-1, 1))
            weight = torch.cat(weight, dim=1)
            weight = torch.eye(shape[0], device=opt_u.device, dtype=opt_u.dtype) - 2 * weight @ weight.t()

        else:
            opt_u = opt_u / opt_u.norm(dim=0)
            weight = torch.eye(shape[0], device=opt_u.device, dtype=opt_u.dtype)
            if reverse:
                indices = range(rank - 1, -1, -1)
            else:
                indices = range(rank)

            for i in indices:
                ui = opt_u[:, i].view(-1, 1)
                weight = weight @ (torch.eye(shape[0], device=opt_u.device, dtype=opt_u.dtype) - 2 * ui @ ui.t())

        return weight

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            new_weight = torch.eye(self.in_features, device=x.device)

            for active_adapter in self.active_adapters:
                if active_adapter not in self.hra_u.keys():
                    continue
                delta_weight = self.get_delta_weight(active_adapter)
                new_weight = torch.mm(new_weight, delta_weight)

            x = x.to(self.get_base_layer().weight.data.dtype)
            orig_weight = self.get_base_layer().weight.data
            new_weight = torch.mm(orig_weight, new_weight)

            result = F.linear(input=x, weight=new_weight, bias=self.base_layer.bias)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "hra." + rep


class HRAConv2d(nn.Module, HRALayer):
    """HRA implemented in Conv2d layer"""

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        apply_GS: bool = False,
        init_weights: Union[bool, str] = True,
        **kwargs,
    ):
        super().__init__()
        HRALayer.__init__(self, base_layer)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, apply_GS, init_weights, **kwargs)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If `True`, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If `None`, all active adapters will be merged.
                Defaults to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.hra_u.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weight = base_layer.weight.data.clone()
                    orig_weight = orig_weight.view(
                        self.out_features,
                        self.in_features * self.base_layer.kernel_size[0] * self.base_layer.kernel_size[0],
                    )
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weight = torch.mm(orig_weight, delta_weight)
                    orig_weight = orig_weight.view(
                        self.out_features,
                        self.in_features,
                        self.base_layer.kernel_size[0],
                        self.base_layer.kernel_size[0],
                    )

                    if not torch.isfinite(orig_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    self.base_layer.weight.data = orig_weight
                else:
                    orig_weight = base_layer.weight.data
                    orig_weight = orig_weight.view(
                        self.out_features,
                        self.in_features * self.base_layer.kernel_size[0] * self.base_layer.kernel_size[0],
                    )
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weight = torch.mm(orig_weight, delta_weight)
                    orig_weight = orig_weight.view(
                        self.out_features,
                        self.in_features,
                        self.base_layer.kernel_size[0],
                        self.base_layer.kernel_size[0],
                    )

                    self.base_layer.weight.data = orig_weight
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.hra_u.keys():
                orig_weight = self.get_base_layer().weight.data.clone()
                orig_weight = orig_weight.view(
                    self.out_features,
                    self.in_features * self.base_layer.kernel_size[0] * self.base_layer.kernel_size[0],
                )
                delta_weight = self.get_delta_weight(active_adapter, reverse=True)
                orig_weight = torch.mm(orig_weight, delta_weight)
                orig_weight = orig_weight.view(
                    self.out_features, self.in_features, self.base_layer.kernel_size[0], self.base_layer.kernel_size[0]
                )

                self.get_base_layer().weight.data = orig_weight

    def get_delta_weight(self, adapter_name: str, reverse: bool = False) -> torch.Tensor:
        rank = self.hra_r[adapter_name]
        apply_GS = self.hra_apply_GS[adapter_name]
        opt_u = self.hra_u[adapter_name]
        shape = opt_u.shape

        if apply_GS:
            weight = [(opt_u[:, 0] / opt_u[:, 0].norm()).view(-1, 1)]
            for i in range(1, rank):
                ui = opt_u[:, i].view(-1, 1)
                for j in range(i):
                    ui = ui - (weight[j].t() @ ui) * weight[j]
                weight.append((ui / ui.norm()).view(-1, 1))
            weight = torch.cat(weight, dim=1)
            weight = torch.eye(shape[0], device=opt_u.device, dtype=opt_u.dtype) - 2 * weight @ weight.t()

        else:
            opt_u = opt_u / opt_u.norm(dim=0)
            weight = torch.eye(shape[0], device=opt_u.device, dtype=opt_u.dtype)
            if reverse:
                indices = range(rank - 1, -1, -1)
            else:
                indices = range(rank)

            for i in indices:
                ui = opt_u[:, i].view(-1, 1)
                weight = weight @ (torch.eye(shape[0], device=opt_u.device, dtype=opt_u.dtype) - 2 * ui @ ui.t())

        return weight

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            new_weight = torch.eye(
                self.in_features * self.base_layer.kernel_size[0] * self.base_layer.kernel_size[0], device=x.device
            )
            for active_adapter in self.active_adapters:
                if active_adapter not in self.hra_u.keys():
                    continue
                delta_weight = self.get_delta_weight(active_adapter)
                new_weight = torch.mm(new_weight, delta_weight)

            x = x.to(self.base_layer.weight.data.dtype)

            orig_weight = self.base_layer.weight.data
            orig_weight = orig_weight.view(
                self.out_features,
                self.in_features * self.base_layer.kernel_size[0] * self.base_layer.kernel_size[0],
            )
            new_weight = torch.mm(orig_weight, new_weight)
            new_weight = new_weight.view(
                self.out_features, self.in_features, self.base_layer.kernel_size[0], self.base_layer.kernel_size[0]
            )

            result = F.conv2d(
                input=x,
                weight=new_weight,
                bias=self.base_layer.bias,
                padding=self.base_layer.padding[0],
                stride=self.base_layer.stride[0],
            )

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "hra." + rep
