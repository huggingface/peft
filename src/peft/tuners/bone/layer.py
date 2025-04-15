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
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


class BoneLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("bone_block",)
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("bone_r",)

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.bone_r = {}
        self.bone_block = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        # flag to enable/disable casting of input to weight dtype during forward call
        self.cast_input_dtype_enabled = True
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        init_weights: bool,
        **kwargs,
    ) -> None:
        """Internal function to create bone adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            init_weights (`bool`): Whether to initialize weights.
        """
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.bone_r[adapter_name] = r

        # Determine shape of Bone weights
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.bone_block[adapter_name] = nn.Parameter(torch.zeros(r, self.out_features), requires_grad=True)

        else:
            raise TypeError(f"Bone is not implemented for base layers of type {type(base_layer).__name__}")

        # Initialize weights
        if init_weights == "bat":
            if self.in_features % r != 0 or self.out_features % r != 0:
                raise ValueError("The weight matrix must be fully divisible into [r, r] blocks.")
            self.reset_bat_parameters(adapter_name, r)
        elif init_weights:
            self.reset_bone_parameters(adapter_name, r)
        else:
            self.reset_bone_parameters_random(adapter_name)
        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_bone_parameters(self, adapter_name: str, r):
        self.bone_block[adapter_name] = nn.Parameter(torch.zeros(r, self.out_features), requires_grad=True)

    def reset_bat_parameters(self, adapter_name: str, r):
        self.bone_block[adapter_name] = nn.Parameter(torch.zeros(self.out_features // r, r, r), requires_grad=True)

    def reset_bone_parameters_random(self, adapter_name: str):
        nn.init.kaiming_uniform_(self.bone_block[adapter_name], a=math.sqrt(5))

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.bone_block.keys():
                continue

            warnings.warn("Scaling operation for Bone not supported! Automatically set scale to 1.")

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.bone_block.keys():
                continue

            warnings.warn("Unscaling operation for Bone not supported! Keeping scale at 1.")


class BoneLinear(nn.Module, BoneLayer):
    """
    Bone implemented in a dense layer.
    """

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        init_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        BoneLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, init_weights, **kwargs)
        self.bone_fn = init_weights

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
            if active_adapter in self.bone_block.keys():
                base_layer = self.get_base_layer()
                orig_dtype = base_layer.weight.dtype
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weight = base_layer.weight.data.clone()
                    if self.bone_fn == "bat":
                        delta_weight = self.get_delta_weight(active_adapter, orig_weight)
                        orig_weight += delta_weight
                    else:
                        delta_weight = self.get_delta_weight_bone(active_adapter, self.base_layer.weight.data)
                        orig_weight = delta_weight

                    if not torch.isfinite(orig_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weight.to(orig_dtype)
                else:
                    if self.bone_fn == "bat":
                        delta_weight = self.get_delta_weight(active_adapter, self.base_layer.weight.data)
                        base_layer.weight.data += delta_weight.to(orig_dtype)
                    else:
                        delta_weight = self.get_delta_weight_bone(active_adapter, self.base_layer.weight.data)
                        base_layer.weight.data = delta_weight.to(orig_dtype)
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
            orig_dtype = base_layer.weight.dtype
            if active_adapter in self.bone_block.keys():
                orig_weight = self.get_base_layer().weight.data.clone()
                if self.bone_fn == "bat":
                    delta_weight = self.get_delta_weight(active_adapter, orig_weight, re=True)
                else:
                    delta_weight = self.get_delta_weight_bone(active_adapter, orig_weight, re=True)

                base_layer.weight.data = delta_weight.to(orig_dtype)

    def get_delta_weight(self, adapter, orig_weight, re: bool = False) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.bone_block[adapter].device
        dtype = self.bone_block[adapter].dtype
        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_bone = self.bone_block[adapter]

        if cast_to_fp32:
            weight_bone = weight_bone.float()
        orig_weight = orig_weight.to(weight_bone.dtype)

        r = weight_bone.size(-1)
        if re:
            o = orig_weight.reshape(orig_weight.size(0) // r, r, orig_weight.size(1) // r, r).permute(2, 0, 1, 3)
            one = torch.eye(weight_bone.size(-1)).to(weight_bone.device)
            # inverse must be in float32, after that the dtype can be adjusted if needed
            inv_I_plus_b = torch.inverse(one + weight_bone)
            inv_I_plus_b = inv_I_plus_b.to(weight_bone.dtype)
            w = (o - weight_bone) @ inv_I_plus_b
            output_tensor = w.permute(1, 2, 0, 3).reshape(*orig_weight.shape)
        else:
            w = (
                orig_weight.reshape(orig_weight.size(0) // r, r, orig_weight.size(1) // r, r).permute(2, 0, 1, 3)
                @ weight_bone
                + weight_bone
            )
            output_tensor = w.permute(1, 2, 0, 3).reshape(*orig_weight.shape)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.bone_block[adapter].data = weight_bone.to(dtype)

        return output_tensor

    def get_delta_weight_bone(self, adapter, orig_weight, re: bool = False) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.bone_block[adapter].device
        dtype = self.bone_block[adapter].dtype
        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_bone = self.bone_block[adapter]

        if cast_to_fp32:
            weight_bone = weight_bone.float()

        in_features = orig_weight.size(-1)
        r = weight_bone.size(0)
        if in_features % r != 0:
            last_size = in_features % r
            n_block = in_features // r
            n_block_size = n_block * r

            if re:
                orig_weight[:, :n_block_size] = (
                    (orig_weight[:, :n_block_size].reshape(-1, n_block, r).permute(1, 2, 0) - weight_bone)
                    .permute(2, 0, 1)
                    .reshape(*orig_weight[:, :n_block_size].shape)
                )
                orig_weight[:, n_block_size:] = (
                    orig_weight[:, n_block_size:] - (weight_bone.transpose(0, 1))[:, :last_size]
                )
            else:
                orig_weight[:, :n_block_size] = (
                    (orig_weight[:, :n_block_size].reshape(-1, n_block, r).permute(1, 2, 0) + weight_bone)
                    .permute(2, 0, 1)
                    .reshape(*orig_weight[:, :n_block_size].shape)
                )
                orig_weight[:, n_block_size:] = (
                    orig_weight[:, n_block_size:] + (weight_bone.transpose(0, 1))[:, :last_size]
                )
            output_tensor = orig_weight

        else:
            if re:
                w = orig_weight.reshape(-1, orig_weight.size(1) // r, r).permute(1, 2, 0) - weight_bone
                output_tensor = w.permute(2, 0, 1).reshape(*orig_weight.shape)
            else:
                w = orig_weight.reshape(-1, orig_weight.size(1) // r, r).permute(1, 2, 0) + weight_bone
                output_tensor = w.permute(2, 0, 1).reshape(*orig_weight.shape)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.bone_block[adapter].data = weight_bone.to(dtype)

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
            if self.bone_fn == "bat":
                orig_weight = self.base_layer.weight.data.clone()
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.bone_block.keys():
                        continue
                    delta_weight = self.get_delta_weight(active_adapter, orig_weight)
                    orig_weight = orig_weight + delta_weight

                x = self._cast_input_dtype(x, orig_weight.dtype)
                bias = self._cast_input_dtype(self.base_layer.bias, orig_weight.dtype)
                result = F.linear(input=x, weight=orig_weight, bias=bias)
            else:
                result = self.base_layer(x, *args, **kwargs)
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.bone_block.keys():
                        continue
                    bone = self.bone_block[active_adapter]
                    r = bone.size(0)
                    if x.size(-1) % r != 0:
                        padding_size = (r - x.size(-1) % r) % r
                        x = F.pad(x, (0, padding_size))
                    x = self._cast_input_dtype(x, bone.dtype)
                    result = result + torch.sum(x.reshape(*x.shape[:-1], x.size(-1) // r, r), dim=-2) @ bone

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "bone." + rep
