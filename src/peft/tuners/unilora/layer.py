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

import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose
from .._buffer_dict import BufferDict

class UniLoraLayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    # unilora_theta_d is a shared parameter.
    #But it is referenced within individual layers.
    adapter_layer_names = ("unilora_theta_d",)

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.unilora_dropout = nn.ModuleDict({})
        device = next(self.base_layer.parameters()).device

        
        self.unilora_indices_A = BufferDict({}, persistent=False)
        self.unilora_indices_B = BufferDict({}, persistent=False)
        
       
        self.unilora_scales_A = BufferDict({}, persistent=False)
        self.unilora_scales_B = BufferDict({}, persistent=False)

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
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
        
        # Initialize indices and move to device
        self.reset_unilora_parameters(adapter_name, theta_d_length)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_unilora_parameters(self, adapter_name, theta_d_length):
        """
        Initializes the indices (pointers to theta_d) randomly.
        Renamed from reset_unilora_logits to ensure clarity.
        """
        if adapter_name in self.unilora_theta_d.keys():
            base_layer = self.get_base_layer()
            device = base_layer.weight.device
            dtype = base_layer.weight.dtype
            # Generate random indices pointing to the vector bank
            indices_A = torch.randint(0, theta_d_length, (self.r[adapter_name], self.in_features), dtype=torch.long)
            indices_B = torch.randint(0, theta_d_length, (self.out_features, self.r[adapter_name]), dtype=torch.long) 
            
            self.unilora_indices_A[adapter_name] = indices_A
            self.unilora_indices_B[adapter_name] = indices_B

            if adapter_name not in self.unilora_scales_A:
                self.unilora_scales_A[adapter_name] = torch.ones(
                    indices_A.shape, device=device, dtype=dtype
                )
            if adapter_name not in self.unilora_scales_B:
                self.unilora_scales_B[adapter_name] = torch.ones(
                    indices_B.shape, device=device, dtype=dtype
                )
              
    def update_scaling(
        self,
        adapter_name: str,
        unilora_scales_A,
        unilora_scales_B,
    ):   
        """
        Updates the scaling factors. 
        """
        if adapter_name in self.unilora_theta_d.keys():
            base_layer = self.get_base_layer()
            target_device = base_layer.weight.device
            target_dtype = base_layer.weight.dtype

            self.unilora_scales_A[adapter_name] = unilora_scales_A.to(
                device=target_device, dtype=target_dtype
            )
            self.unilora_scales_B[adapter_name] = unilora_scales_B.to(
                device=target_device, dtype=target_dtype
            )

    def _ensure_device(self, adapter):
        """
        Ensure all UniLoRA buffers/params for the given adapter are on the same device as base_layer.
        This is lazy-migration (only happens if device mismatch is detected).
        """
        # get target device from base_layer
        device = next(self.base_layer.parameters()).device

        # ---- indices ----
        if adapter in self.unilora_indices_A:
            t = self.unilora_indices_A[adapter]
            if t.device != device:
                self.unilora_indices_A[adapter] = t.to(device)

        if adapter in self.unilora_indices_B:
            t = self.unilora_indices_B[adapter]
            if t.device != device:
                self.unilora_indices_B[adapter] = t.to(device)

        # ---- scales ----
        if adapter in self.unilora_scales_A:
            t = self.unilora_scales_A[adapter]
            if t.device != device:
                self.unilora_scales_A[adapter] = t.to(device)

        if adapter in self.unilora_scales_B:
            t = self.unilora_scales_B[adapter]
            if t.device != device:
                self.unilora_scales_B[adapter] = t.to(device)

        # ---- theta_d ---- (ParameterDict, but ensure consistency)
        if adapter in self.unilora_theta_d:
            p = self.unilora_theta_d[adapter]
            if p.device != device:
                # Parameter migration: need .data to avoid creating new graph edges
                self.unilora_theta_d[adapter].data = p.data.to(device)

class Linear(nn.Linear, UniLoraLayer):
    # UniLora implemented in a dense layer
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
            adapter_name, unilora_theta_d, r, theta_d_length, unilora_dropout,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.unilora_indices_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
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

    def _get_lora_matrices(self, adapter, cast_to_fp32=False) -> Tuple[torch.Tensor, torch.Tensor]:
        # Changed: Accessing the renamed buffers
        self._ensure_device(adapter)
        
        unilora_indices_A = self.unilora_indices_A[adapter] 
        unilora_indices_B = self.unilora_indices_B[adapter] 

        
        
        unilora_theta_d = self.unilora_theta_d[adapter].to(unilora_indices_A.device)
        
        if cast_to_fp32:
            unilora_theta_d = unilora_theta_d.float()

        
        A = unilora_theta_d[unilora_indices_A.long()] * self.unilora_scales_A[adapter]
        B = unilora_theta_d[unilora_indices_B.long()] * self.unilora_scales_B[adapter]
        
        # Cast back if necessary (handled implicitly by torch usually, but good to be explicit if needed)
        if cast_to_fp32:
             A = A.float()
             B = B.float()

        return A, B

    def get_delta_weight(self, adapter) -> torch.Tensor:
        # Changed: Accessing the renamed buffer for device check
        self._ensure_device(adapter)
        device = self.unilora_indices_A[adapter].device
        # Note: indices are Long, we usually want the dtype of the output (which depends on theta_d/scales)
        # Using theta_d's dtype is safer for checking fp16/32 mismatch
        dtype = self.unilora_theta_d[adapter].dtype 
        
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)
        
        A, B = self._get_lora_matrices(adapter, cast_to_fp32)
        
        # B @ A computation
        output_tensor = transpose(B @ A, self.fan_in_fan_out)
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
                
                # Standard LoRA calculation: x @ A @ B
                result = result + F.linear(F.linear(dropout(x), A), B)
        
        result = result.to(previous_dtype)
        return result