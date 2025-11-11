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
import warnings
from typing import Any, Optional

import torch
import torch.nn as nn

from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


class DeloraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = (
        "delora_A",
        "delora_B",
        "delora_lambda",
    )
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = (
        "r",
        "delora_dropout",
        "delora_w_norm",
    )

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.delora_dropout = nn.ModuleDict({})
        self.delora_A = nn.ParameterDict({})
        self.delora_B = nn.ParameterDict({})
        self.delora_lambda = nn.ParameterDict({})
        # Use persistent buffers so they are included in state_dict and saved.
        self.delora_w_norm = BufferDict({}, persistent=True)
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer_mod = self.get_base_layer()
        if isinstance(base_layer_mod, nn.Linear):
            self.in_features, self.out_features = base_layer_mod.in_features, base_layer_mod.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer_mod)}")

    @staticmethod
    def _compute_delta(
        A: torch.Tensor, B: torch.Tensor, delora_lambda: torch.Tensor, r: int, w_norm: torch.Tensor
    ) -> torch.Tensor:
        """Compute delta = B @ diag(delora_lambda/r / (||A_i||*||B^j||)) @ A, scaled by provided w_norm (per-input channel)"""
        An = torch.clamp(A.norm(dim=1), min=1e-4)
        Bn = torch.clamp(B.norm(dim=0), min=1e-4)
        diag = torch.diag_embed(delora_lambda / r / (An * Bn))
        delta = B @ diag @ A
        delta = delta * w_norm.unsqueeze(0)
        return delta

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        if adapter not in self.delora_A or adapter not in self.delora_B:
            raise ValueError(f"Adapter {adapter} not found.")

        delta = self._compute_delta(
            self.delora_A[adapter],
            self.delora_B[adapter],
            self.delora_lambda[adapter],
            self.r[adapter],
            self.delora_w_norm[adapter],
        )
        return delta

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        delora_lambda: float,
        module_dropout: float,
        init_weights: bool = True,
        inference_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        """Internal function to create delora adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            delora_lambda (`float`): Boundary for the adapter's norm.
            module_dropout (`float`): The dropout probability for disabling adapter during training.
            init_weights (`bool`): Whether to initialize weights.
        """
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.delora_A[adapter_name] = nn.Parameter(torch.empty(r, self.in_features))
        self.delora_B[adapter_name] = nn.Parameter(torch.empty(self.out_features, r))
        self.delora_lambda[adapter_name] = nn.Parameter(torch.empty(1))
        if module_dropout > 0.0:
            module_dropout_layer = nn.Dropout(p=module_dropout)
        else:
            module_dropout_layer = nn.Identity()
        self.delora_dropout.update(nn.ModuleDict({adapter_name: module_dropout_layer}))

        # Initialize weights
        self.reset_delora_parameters(adapter_name, init_weights, delora_lambda)

        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def reset_delora_parameters(
        self,
        adapter_name: str,
        init_weights: bool = True,
        delora_lambda: float = 15.0,
    ) -> None:
        if adapter_name not in self.delora_A.keys():
            return

        if init_weights is True:
            nn.init.kaiming_uniform_(self.delora_A[adapter_name], a=math.sqrt(5))
            nn.init.zeros_(self.delora_B[adapter_name])
        else:
            nn.init.kaiming_uniform_(self.delora_A[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.delora_B[adapter_name], a=math.sqrt(5))

        self.delora_lambda[adapter_name].data.fill_(float(delora_lambda))

        # capture a fixed norm for this adapter to use for future delta computations
        with torch.no_grad():
            w = self.get_base_layer().weight
            if w.device.type != "meta":
                w_norm = torch.norm(w.data, dim=0).detach()
            else:
                # For meta tensors, we can't compute the norm, so use a default value
                w_norm = torch.ones(w.shape[1], device=w.device)
            self.delora_w_norm[adapter_name] = w_norm


class DeloraLinear(nn.Module, DeloraLayer):
    # DeLoRA implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int,
        delora_lambda: float,
        module_dropout: float,
        init_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        DeloraLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, delora_lambda, module_dropout, init_weights)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.delora_A.keys():
                base_layer = self.get_base_layer()
                delta_weight = (
                    self.get_delta_weight(active_adapter)
                    .detach()
                    .to(dtype=base_layer.weight.dtype, device=base_layer.weight.device)
                )
                with torch.no_grad():
                    if safe_merge:
                        orig_weights = base_layer.weight.data.clone()
                        orig_weights = orig_weights + delta_weight

                        if not torch.isfinite(orig_weights).all():
                            raise ValueError(
                                f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                            )

                        base_layer.weight.data = orig_weights
                    else:
                        base_layer.weight.data.add_(delta_weight)

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        Unmerge all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.delora_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            if not self.active_adapters:
                return self.base_layer(x, *args, **kwargs).to(previous_dtype)

            base_out = self.base_layer(x, *args, **kwargs)
            add_out = torch.zeros_like(base_out)

            for adapter in self.active_adapters:
                if adapter not in self.delora_A:
                    continue

                x_d = self.delora_dropout[adapter](x)

                # Decomposed delta calculation
                # 1. (x * w_norm) @ A.T
                h = nn.functional.linear(x_d * self.delora_w_norm[adapter], self.delora_A[adapter])

                # 2. h @ diag
                An = torch.clamp(self.delora_A[adapter].norm(dim=1), min=1e-4)
                Bn = torch.clamp(self.delora_B[adapter].norm(dim=0), min=1e-4)
                scaling = (self.delora_lambda[adapter] / self.r[adapter]) / (An * Bn)

                h = h * scaling

                # 3. h @ B.T
                h = nn.functional.linear(h, self.delora_B[adapter])

                add_out += h

            result = base_out + add_out.to(base_out.dtype)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "delora." + rep
