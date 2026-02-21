# Copyright 2026-present the HuggingFace Inc. team.
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
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from peft.tuners.tuners_utils import BaseTunerLayer
from transformers.pytorch_utils import Conv1D
import warnings

class LilyLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names: tuple[str, ...] = ("lily_A", "lily_B", "lily_router")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names: tuple[str, ...] = ("r", "scaling", "num_A", "num_B")
    def __init__(
        self,
        base_layer: nn.Module, 
        **kwargs
    ) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.scaling = {}
        self.num_A = {}
        self.num_B = {}
        self.lily_A = nn.ModuleDict({})
        self.lily_B = nn.ModuleDict({})
        self.lily_router = nn.ModuleDict({})
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features
    
    def update_layer(
        self,
        adapter_name,
        r,
        scaling,
        num_A,
        num_B,
        lily_A=None,
        lily_B=None,
    ):
        # collect the kwargs
        kwargs = locals().copy()
        del kwargs["self"]

        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        
        self.r[adapter_name] = r
        self.scaling[adapter_name] = scaling
        
        # Actual trainble parameters
        self.lily_A[adapter_name] = lily_A if lily_A is not None else nn.Linear(self.in_features, r, bias=False)
        self.lily_B[adapter_name] = lily_B if lily_B is not None else nn.Linear(r, self.out_features, bias=False)
        self.lily_router[adapter_name] = nn.Linear(r, num_B, bias=False)
        self.num_A[adapter_name] = num_A  
        self.num_B[adapter_name] = num_B  
        self.reset_lily_parameters(adapter_name) # initialize the parameters
        self.set_adapter(self.active_adapters)

    def reset_lily_parameters(self, adapter_name):
        if adapter_name in self.lily_A.keys():
            nn.init.kaiming_uniform_(self.lily_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lily_router[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lily_B[adapter_name].weight)
    

class Linear(nn.Module, LilyLayer):
    # Lily implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        scaling: float = 1.0,
        num_A: int = 4,
        num_B: int = 4,
        lily_A: nn.Linear = None,
        lily_B: nn.Linear = None,
        **kwargs,
    ) -> None:
        super().__init__()
        LilyLayer.__init__(self, base_layer, **kwargs)

        self._active_adapter = adapter_name

        self.update_layer(
            adapter_name,
            r,
            scaling=scaling,
            lily_A=lily_A,
            lily_B=lily_B,
            num_A=num_A,
            num_B=num_B,
        )

    def get_delta_weight(self, adapter, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        raise NotImplementedError("This method is not supported for Lily.")
    
    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError("This method is not supported for Lily.")

    def unmerge(self) -> None:
        raise NotImplementedError("This method is not supported for Lily.")

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        lily_A_keys = self.lily_A.keys()

        for active_adapter in self.active_adapters:
            if active_adapter not in lily_A_keys:
                continue

            lily_A = self.lily_A[active_adapter]
            lily_B = self.lily_B[active_adapter]
            router = self.lily_router[active_adapter]
            B = einops.rearrange(lily_B.weight, "(e i) o -> e i o", e=self.num_B)
            scaling = self.scaling[active_adapter]
            x = x.to(lily_A.weight.dtype)
            hidden = lily_A(x)
            router_logits = router(hidden) # [B, N, num_of_experts]
            router_probability = F.softmax(router_logits, dim=-1) # [B, N, num_of_experts]
            expert_probabilities = router_probability.mean(dim=(0, 1)) 
            combined_B = torch.einsum("e,eio->io", expert_probabilities, B)
            delta = torch.matmul(hidden, combined_B)
            result = result + (delta * scaling)
            result = result.to(torch_result_dtype)

        return result
    
    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lily." + rep