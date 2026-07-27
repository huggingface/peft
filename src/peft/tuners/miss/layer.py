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
import torch.nn.functional as F
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer, _get_in_out_features, check_adapters_to_merge
from peft.utils import quantization_extra_repr, resolve_quantization_backend

from .config import MissConfig


class MissLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("miss_block",)
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("miss_r", "miss_dropout", "miss_mini_r")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.quantization_backend = resolve_quantization_backend(
            self.get_base_layer(), get_apply_tensor_subclass=kwargs.get("get_apply_tensor_subclass")
        )
        self.miss_r = {}
        self.miss_dropout = nn.ModuleDict({})
        self.miss_mini_r = {}
        self.miss_block = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        # flag to enable/disable casting of input to weight dtype during forward call
        self.cast_input_dtype_enabled = True
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        in_features, out_features = _get_in_out_features(base_layer)
        if (in_features is None) or (out_features is None):
            raise TypeError(f"Unsupported layer type {type(base_layer)}")
        self.in_features, self.out_features = in_features, out_features

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        config: MissConfig,
        inference_mode: bool = False,
        **kwargs,
    ) -> None:
        """Internal function to create miss adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            init_weights (`bool`): Whether to initialize weights.
        """
        mini_r = config.mini_r
        miss_dropout = config.miss_dropout
        init_weights = config.init_weights

        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.miss_r[adapter_name] = r
        self.miss_mini_r[adapter_name] = mini_r
        if miss_dropout > 0.0:
            miss_dropout_layer = nn.Dropout(p=miss_dropout)
        else:
            miss_dropout_layer = nn.Identity()

        self.miss_dropout[adapter_name] = miss_dropout_layer

        # Determine shape of MiSS weights; supportedness was already validated in __init__ and the model dispatcher.
        self.miss_block[adapter_name] = nn.Parameter(torch.zeros(r, self.out_features), requires_grad=True)

        # Initialize weights
        if init_weights == "bat":
            if self.in_features % r != 0 or self.out_features % r != 0:
                raise ValueError("The weight matrix must be fully divisible into [r, r] blocks.")
            self.reset_bat_parameters(adapter_name, r)
        elif init_weights == "mini":
            if self.out_features % mini_r != 0:
                raise ValueError(
                    "mini_r is divided along the out_features dimension. For optimal performance and implementation simplicity,"
                    "it is recommended that out_features be divisible by mini_r."
                    "Error: {self.out_features} % mini_r != 0"
                )
            self.reset_mini_parameters(adapter_name, r, mini_r)
        elif init_weights:
            self.reset_miss_parameters(adapter_name, r)
        else:
            self.reset_miss_parameters_random(adapter_name)
        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def reset_miss_parameters(self, adapter_name: str, r):
        self.miss_block[adapter_name] = nn.Parameter(torch.zeros(r, self.out_features), requires_grad=True)

    def reset_bat_parameters(self, adapter_name: str, r):
        self.miss_block[adapter_name] = nn.Parameter(torch.zeros(self.out_features // r, r, r), requires_grad=True)

    def reset_mini_parameters(self, adapter_name: str, r, mini_r):
        self.miss_block[adapter_name] = nn.Parameter(torch.zeros(r, mini_r), requires_grad=True)

    def reset_miss_parameters_random(self, adapter_name: str):
        nn.init.kaiming_uniform_(self.miss_block[adapter_name], a=math.sqrt(5))

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.miss_block.keys():
                continue

            warnings.warn("Scaling operation for MiSS not supported! Automatically set scale to 1.")

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.miss_block.keys():
                continue

            warnings.warn("Unscaling operation for MiSS not supported! Keeping scale at 1.")


class MissLinear(nn.Module, MissLayer):
    """
    MiSS implemented in a dense layer.
    """

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        config: MissConfig,
        r: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        MissLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, config=config, **kwargs)
        self.miss_fn = config.init_weights

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
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
            if active_adapter in self.miss_block.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    weight = self.get_base_weight().clone()
                    orig_dtype = weight.dtype
                    if self.miss_fn == "bat":
                        weight += self.get_delta_weight(active_adapter, weight)
                    else:
                        weight = self.get_delta_weight_miss(active_adapter, weight)

                    if not torch.isfinite(weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    self.set_base_weight(weight.to(orig_dtype))
                else:
                    weight = self.get_base_weight()
                    orig_dtype = weight.dtype
                    if self.miss_fn == "bat":
                        weight += self.get_delta_weight(active_adapter, weight)
                    else:
                        weight = self.get_delta_weight_miss(active_adapter, weight)
                    self.set_base_weight(weight.to(orig_dtype))
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
            base_layer = self.get_base_layer()
            if active_adapter in self.miss_block.keys():
                weight = self.get_base_weight()
                orig_dtype = weight.dtype
                if self.miss_fn == "bat":
                    weight = self.get_delta_weight(active_adapter, weight, reverse=True)
                elif self.miss_fn == "mini":
                    weight = self.get_delta_weight_miss(active_adapter, weight, reverse=True)
                else:
                    weight = self.get_delta_weight_miss(active_adapter, weight, reverse=True)

                self.set_base_weight(weight.to(orig_dtype))

    def get_delta_weight(self, adapter, orig_weight, reverse: bool = False) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (`str`):
                The name of the adapter for which the delta weight should be computed.
            reverse (bool):
                If True, reverse the merge (unmerge). If False, apply the merge (forward).
        """
        device = self.miss_block[adapter].device
        dtype = self.miss_block[adapter].dtype
        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        miss_B = self.miss_block[adapter]

        if cast_to_fp32:
            miss_B = miss_B.float()
        orig_weight = orig_weight.to(miss_B.dtype)

        r = miss_B.size(-1)
        W = orig_weight.reshape(orig_weight.size(0) // r, r, orig_weight.size(1) // r, r).permute(2, 0, 1, 3)

        if reverse:
            eye = torch.eye(r, device=miss_B.device, dtype=torch.float32)
            inv_I_plus_miss_B = torch.inverse(eye + miss_B.float()).to(miss_B.dtype)
            result = (W - miss_B) @ inv_I_plus_miss_B
        else:
            result = W @ miss_B + miss_B

        output_tensor = result.permute(1, 2, 0, 3).reshape(*orig_weight.shape)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)
            self.miss_block[adapter].data = miss_B.to(dtype)

        return output_tensor

    def get_delta_weight_miss(self, adapter, orig_weight, reverse: bool = False) -> torch.Tensor:
        """
        Compute the *full* weight for the given adapter (not the weight delta!)

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
            reverse (bool):
                If True, reverse the merge (unmerge). If False, apply the merge (forward).
        """
        device = self.miss_block[adapter].device
        dtype = self.miss_block[adapter].dtype
        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        miss_B = self.miss_block[adapter]

        if cast_to_fp32:
            miss_B = miss_B.float()

        in_features = orig_weight.size(-1)
        out_features = orig_weight.size(0)
        r = miss_B.size(0)
        if self.miss_fn == "mini":
            miss_B = miss_B.repeat(1, out_features // self.miss_mini_r[adapter])

        sign = -1 if reverse else 1

        if in_features % r != 0:
            remainder = in_features % r
            n_blocks = in_features // r
            aligned_size = n_blocks * r

            W_aligned = orig_weight[:, :aligned_size].reshape(-1, n_blocks, r).permute(1, 2, 0)
            orig_weight[:, :aligned_size] = (
                (W_aligned + sign * miss_B).permute(2, 0, 1).reshape(*orig_weight[:, :aligned_size].shape)
            )
            orig_weight[:, aligned_size:] = (
                orig_weight[:, aligned_size:] + sign * miss_B.transpose(0, 1)[:, :remainder]
            )
            output_tensor = orig_weight
        else:
            W_blocks = orig_weight.reshape(-1, orig_weight.size(1) // r, r).permute(1, 2, 0)
            output_tensor = (W_blocks + sign * miss_B).permute(2, 0, 1).reshape(*orig_weight.shape)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)
            self.miss_block[adapter].data = miss_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            if self.miss_fn == "bat":
                if (self.quantization_backend is not None) and (not self.quantization_backend.supports_merge):
                    raise ValueError(
                        "Using MiSS with `init_weights='bat'` is not supported because quantization backend "
                        f"{self.quantization_backend.backend_name} does not support dequantization. Use a different "
                        "quantization backend or a different MiSS initialization method."
                    )

                orig_weight = self.get_base_weight().clone()
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.miss_block.keys():
                        continue
                    delta_weight = self.get_delta_weight(active_adapter, orig_weight)
                    orig_weight = orig_weight + delta_weight

                x = self._cast_input_dtype(x, orig_weight.dtype)
                bias = self._cast_input_dtype(self.base_layer.bias, orig_weight.dtype)
                result = F.linear(input=x, weight=orig_weight, bias=bias)
            else:
                result = self.base_layer(x, *args, **kwargs)
                if self.quantization_backend is not None:
                    result = self.quantization_backend.maybe_clone_base_result(result)
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.miss_block.keys():
                        continue
                    miss = self.miss_block[active_adapter]
                    if self.miss_fn == "mini":
                        miss = miss.repeat(1, self.base_layer.out_features // self.miss_mini_r[active_adapter])

                    dropout = self.miss_dropout[active_adapter]
                    r = miss.size(0)
                    if x.size(-1) % r != 0:
                        padding_size = (r - x.size(-1) % r) % r
                        x = F.pad(x, (0, padding_size))
                    x = self._cast_input_dtype(x, miss.dtype)
                    result = result + torch.sum(dropout(x).reshape(*x.shape[:-1], x.size(-1) // r, r), dim=-2) @ miss

        result = result.to(previous_dtype)
        return result

    def supports_lora_conversion(self, adapter_name: str = "default") -> bool:
        return True

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "miss." + rep

    def extra_repr(self) -> str:
        return quantization_extra_repr(self)
