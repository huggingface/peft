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
from typing import Any, List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.lycoris_utils import LycorisLayer

def parse_positions(positions: str):
    # Code borrow from https://github.com/stanfordnlp/pyreft/pyreft/dataset.py
    first_n, last_n = 0, 0
    if "+" in positions:
        first_n = int(positions.split("+")[0].strip("f"))
        last_n = int(positions.split("+")[1].strip("l"))
    else:
        if "f" in positions:
            first_n = int(positions.strip("f"))
        elif "l" in positions:
            last_n = int(positions.strip("l"))
    return first_n, last_n

class ReftLayer(nn.Module, LycorisLayer):
    # All names of layers that may contain adapter weights
    adapter_layer_names = ("reft_A", "reft_R")
    # other_param_names is defined on parent class   

    def __init__(self, base_layer: nn.Module):
        super().__init__()
        LycorisLayer.__init__(self, base_layer)

        # ReFT info
        self.reft_A = nn.ModuleDict({})
        self.reft_R = nn.ModuleDict({})
        self.loc = dict()
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels

        self.in_features = in_features
        self.out_features = out_features


    @property
    def _available_adapters(self) -> Set[str]:
        return {*self.reft_A, *self.reft_R}

    def create_adapter_parameters(self, adapter_name: str, r: int):
        rotate_layer = torch.nn.Linear(self.out_features, r, bias=False)
        torch.nn.init.orthogonal_(rotate_layer.weight)
        self.reft_R[adapter_name] = torch.nn.utils.parametrizations.orthogonal(rotate_layer, orthogonal_map='cayley')
        self.reft_A[adapter_name] =  torch.nn.Linear(self.out_features, r)

    def reset_adapter_parameters(self, adapter_name: str):
        # Original implementation performs initialization with normal distribution
        # https://github.com/KohakuBlueleaf/LyCORIS/blob/3549fdef8f564761d68b695a08ef88b1122fdedc/lycoris/modules/loha.py#L158

        # FedPara paper proposes to perform He initialization, let's stick with it
        # It is enough to initialize only single matrix with zeros to make adapter do nothing after initialization
        if adapter_name in self.reft_A.keys():
            nn.init.kaiming_uniform_(self.reft_A[adapter_name].weight, a=math.sqrt(5))


    def update_layer(
        self,
        adapter_name: str,
        r: int,
        alpha: float,
        loc: Optional[List[int] | int],
        rank_dropout: float,
        module_dropout: float,
        init_weights: bool,
        use_effective_conv2d: bool = False,
        **kwargs,
    ) -> None:
        """Internal function to create loha adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            alpha (`float`): Alpha for the added adapter.
            rank_dropout (`float`): The dropout probability for rank dimension during training.
            module_dropout (`float`): The dropout probability for disabling adapter during training.
            init_weights (`bool`): Whether to initialize weights.
            use_effective_conv2d (`bool`, *optional*, defaults to `False`):
                Use parameter effective decomposition for Conv2d with ksize > 1.
        """
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        first_n, last_n = parse_positions(loc)
        self.first_n[adapter_name] = first_n
        self.last_n[adapter_name] = last_n
        self.alpha[adapter_name] = alpha
        self.scaling[adapter_name] = alpha / r
        self.rank_dropout[adapter_name] = rank_dropout
        self.module_dropout[adapter_name] = module_dropout

        # Create weights with provided shape
        self.create_adapter_parameters(adapter_name, r)

        # Initialize weights
        if init_weights:
            self.reset_adapter_parameters(adapter_name)

        # Move new weights to device
        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        return self.reft_R[adapter_name]

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

                rotate_layer = self.reft_R[active_adapter]
                learned_source = self.reft_A[active_adapter]
                module_dropout = self.module_dropout[active_adapter]
                if self.loc[active_adapter] is None:
                    rotated_base = rotate_layer(result)
                    offset = torch.matmul((learned_source(result) - rotated_base), rotate_layer.weight)
                    output = result + offset
                else:
                    loc = self.loc[active_adapter]
                    selected_results = torch.gather(result, 1, loc)
                    rotated_base = rotate_layer(selected_results)
                    offset = torch.matmul((learned_source(selected_results) - rotated_base), rotate_layer.weight)
                    output.scatter_(1, loc, offset)
                    output = module_dropout(output)

        output = output.to(previous_dtype)
        return output


class Linear(ReftLayer):
    """Reft implemented in Linear layer"""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str = "default",
        r: int = 0,
        alpha: float = 0.0,
        loc: str = None,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        init_weights: bool = True,
        **kwargs,
    ):
        super().__init__(base_layer)

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, alpha, loc, rank_dropout, module_dropout, init_weights, **kwargs)

    def _get_delta_activations(
        self, adapter_name: str, input: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        delta_weight = self.get_delta_weight(adapter_name)
        # don't add bias here, because the bias is already included in the output of the base_layer
        return F.linear(input, delta_weight)



    def __repr__(self) -> str:
        rep = super().__repr__()
        return "reft." + rep


class Conv2d(ReftLayer):
    """Reft implemented in Conv2d layer"""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str = "default",
        r: int = 0,
        alpha: float = 0.0,
        loc: str = None,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        use_effective_conv2d: bool = False,
        init_weights: bool = True,
        **kwargs,
    ):
        super().__init__(base_layer)

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, r, alpha, loc, rank_dropout, module_dropout, init_weights, use_effective_conv2d, **kwargs
        )

    def _get_delta_activations(
        self, adapter_name: str, input: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        delta_weight = self.get_delta_weight(adapter_name)
        # don't add bias here, because the bias is already included in the output of the base_layer
        base_layer = self.get_base_layer()
        return F.conv2d(
            input,
            delta_weight,
            stride=base_layer.stride,
            padding=base_layer.padding,
            dilation=base_layer.dilation,
            groups=base_layer.groups,
        )


    def __repr__(self) -> str:
        rep = super().__repr__()
        return "reft." + rep



