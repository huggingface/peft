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

import math
from typing import Any, Optional, Set, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.lycoris_utils import LycorisLayer


class WeightLoraLayer(nn.Module, LycorisLayer):
    # All names of layers that may contain adapter weights
    adapter_layer_names = (
        "weight_lora_A",
        "weight_lora_B",
        "weight_lora_w",
    )
    # other_param_names is defined on parent class

    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__()
        LycorisLayer.__init__(self, base_layer)

        # Weight LoRA info
        self.weight_lora_A = nn.ParameterDict({})
        self.weight_lora_B = nn.ParameterDict({})
        self.weight_lora_w = nn.ParameterDict({})

    @property
    def _available_adapters(self) -> Set[str]:
        return {
            *self.weight_lora_A,
            *self.weight_lora_B,
            *self.weight_lora_w,
        }

    def create_adapter_parameters(
        self,
        adapter_name: str,
        r: int,
        shape
    ):
        self.weight_lora_A[adapter_name] = nn.Parameter(torch.empty(shape[0], r))
        self.weight_lora_B[adapter_name] = nn.Parameter(torch.empty(r, shape[1]))
        self.weight_lora_w[adapter_name] = nn.Parameter(torch.empty(1))

    def reset_adapter_parameters(self, adapter_name: str):
        # Vanilla LoRA initialization
        nn.init.kaiming_uniform_(self.weight_lora_A[adapter_name], a=math.sqrt(5))
        nn.init.zeros_(self.weight_lora_B[adapter_name])
        nn.init.ones_(self.weight_lora_w[adapter_name])

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: float,
        rank_dropout: float,
        module_dropout: float,
        **kwargs,
    ) -> None:
        """Internal function to create Weight LoRA adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            lora_alpha (`float`): Alpha for the added adapter.
            rank_dropout (`float`): The dropout probability for rank dimension during training
            module_dropout (`float`): The dropout probability for disabling adapter during training.
            init_weights (`bool`): Whether to initialize adapter weights.
        """
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = lora_alpha / r
        self.rank_dropout[adapter_name] = rank_dropout
        self.module_dropout[adapter_name] = module_dropout
        base_layer = self.get_base_layer()

        # Determine shape of Weight LoRA weights
        if isinstance(base_layer, nn.Linear):
            shape = (base_layer.in_features, base_layer.out_features)
        else:
            raise TypeError(f"WeightLoRA is not implemented for base layers of type {type(base_layer).__name__}")

        # Create weights with provided shape
        self.create_adapter_parameters(adapter_name, r, shape)

        # Initialize weights
        self.reset_adapter_parameters(adapter_name)

        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        device = self.weight_lora_B[adapter_name].device
        dtype = self.weight_lora_B[adapter_name].dtype
        w_A = self.weight_lora_A[adapter_name]
        w_B = self.weight_lora_B[adapter_name]
        w = self.weight_lora_w[adapter_name]

        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)
        if cast_to_fp32:
            w_A = w_A.float()
            w_B = w_B.float()

        # Combine marixes
        weight = w * w_A @ w_B * self.scaling[adapter_name]
        weight = weight.T
        if cast_to_fp32:
            weight = weight.to(dtype=dtype)

            self.lora_A[adapter_name].weight.data = w_A.to(dtype)
            self.lora_B[adapter_name].weight.data = w_B.to(dtype)

        # Perform rank dropout during training - drop rows of addition weights
        rank_dropout = self.rank_dropout[adapter_name]
        if self.training and rank_dropout:
            drop = (torch.rand(weight.size(0)) > rank_dropout).float()
            drop = drop.view(-1, *[1] * len(weight.shape[1:])).to(weight.device)
            drop /= drop.mean()
            weight *= drop

        return weight

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

            # Execute all the adapters
            for active_adapter in self.active_adapters:
                if active_adapter not in self._available_adapters:
                    continue

                module_dropout = self.module_dropout[active_adapter]

                # Modify current execution weights
                if (not self.training) or (self.training and torch.rand(1) > module_dropout):
                    result = result + self._get_delta_activations(active_adapter, x, *args, **kwargs)

        result = result.to(previous_dtype)
        return result


class Linear(WeightLoraLayer):
    """WeightLoRA implemented in Linear layer"""

    def __init__(
        self,
        base_layer: nn.Module,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        adapter_name: str = "default",
        r: int = 0,
        lora_alpha: float = 1.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(base_layer)

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, rank_dropout, module_dropout, **kwargs)

    def _get_delta_activations(
        self, adapter_name: str, input: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        delta_weight = self.get_delta_weight(adapter_name)
        # don't add bias here, because the bias is already included in the output of the base_layer
        delta_weight = delta_weight.to(input.dtype)
        return F.linear(input, delta_weight)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "weight_lora." + rep
