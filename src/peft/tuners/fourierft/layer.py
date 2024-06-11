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
from typing import Any, List, Optional, Union
import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTunerLayer


class FourierLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ["spectrum"]
    # All names of other parameters that may contain adapter-related parameters
    # other_param_names = ("rank", "dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.n_frequency = {}
        self.scaling = {}
        self.spectrum = nn.ParameterDict({})
        self.indices = {}
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.random_loc_seed = kwargs.pop("random_loc_seed")
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")


    def update_layer(self, adapter_name, n_frequency, scaling, init_fourier_weights):
        if n_frequency <= 0:
            raise ValueError(f"`n_frequency` should be a positive integer value but the value passed is {n_frequency}")
        self.n_frequency[adapter_name] = n_frequency

        weight = getattr(self.get_base_layer(), "weight", None)
        # Actual trainable parameters
        self.indices[adapter_name] = torch.randperm(self.out_features * self.in_features, generator=torch.Generator().manual_seed(self.random_loc_seed))[:n_frequency]
        self.indices[adapter_name] = torch.stack([self.indices[adapter_name] // self.in_features, self.indices[adapter_name] % self.in_features], dim=0)
        self.spectrum[adapter_name] = nn.Parameter(torch.randn(n_frequency), requires_grad=True)
        self.scaling[adapter_name] = scaling
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        
        self.reset_fourier_parameters(adapter_name, init_fourier_weights)
        self.set_adapter(self.active_adapters)



    @torch.no_grad()
    def reset_fourier_parameters(self, adapter_name, init_fourier_weights):
        if init_fourier_weights is False:
            return

        nn.init.normal_(self.spectrum[adapter_name])

        if adapter_name in self.spectrum.keys():
            if init_fourier_weights.lower() == "xavier_dense":
                dense_weight = torch.zeros_like(self.get_base_layer().weight.data)
                nn.init.xavier_uniform_(dense_weight)
                scale = torch.norm(dense_weight) / torch.norm(self.get_delta_weight(adapter_name))
                self.spectrum[adapter_name].data *= scale
            elif init_fourier_weights.lower() == "gaussian":
                pass
            else:
                raise ValueError(f"Unknown initialization {init_fourier_weights=}")


    def get_delta_weight(self, adapter) -> torch.Tensor:
        spectrum = self.spectrum[adapter]
        indices = self.indices[adapter].to(spectrum.device)

        dense_spectrum = torch.zeros(self.out_features, self.in_features, device=spectrum.device, dtype=spectrum.dtype)
        dense_spectrum[indices[0,:], indices[1,:]] = spectrum
        # delta_weight = torch.fft.ifft2(dense_spectrum, norm='ortho').real
        delta_weight = torch.fft.ifft2(dense_spectrum).real * self.scaling[adapter]
        return delta_weight

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class FourierLinear(nn.Module, FourierLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        n_frequency: int = 1000,
        scaling: float = 150.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        init_fourier_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        FourierLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, n_frequency, scaling, init_fourier_weights)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )

        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in adapter_names:
            if active_adapter in self.spectrum.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
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
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.spectrum.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        return super().get_delta_weight(adapter)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
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
                if active_adapter not in self.spectrum.keys():
                    continue

                delta_w = self.get_delta_weight(active_adapter)
                x = x.to(delta_w.dtype)
                
                result += torch.einsum('ijk,kl->ijl', x, delta_w)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "fourier." + rep