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
from typing import Any

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
    other_param_names: tuple[str, ...] = ("r", "lily_scaling", "lily_dropout", "num_A", "num_B")
    def __init__(
        self,
        base_layer: nn.Module, 
        **kwargs
    ) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lily_scaling = {}
        self.lily_dropout = nn.ModuleDict({})
        self.num_A = {}
        self.num_B = {}
        self.lily_A = nn.ModuleDict({})
        self.lily_B = nn.ModuleDict({})
        self.lily_router = nn.ModuleDict({})
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv1d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv3d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif isinstance(base_layer, nn.MultiheadAttention):
            if not base_layer._qkv_same_embed_dim:
                raise ValueError(f"Only same dim for query/key/value is supported as of now for {self.__class__}.")
            in_features, out_features = base_layer.embed_dim, 3 * base_layer.embed_dim
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "EetqLinear":
            # Eetq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear":
            # HQQ layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "PatchedLinear":
            # INC layers
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
        lily_scaling,
        lily_dropout,
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
        self.lily_scaling[adapter_name] = lily_scaling
        if lily_dropout > 0.0:
            lily_dropout_layer = nn.Dropout(p=lily_dropout)
        else:
            lily_dropout_layer = nn.Identity()
        
        self.lily_dropout.update(nn.ModuleDict({adapter_name: lily_dropout_layer}))
        # Actual trainble parameters
        self.lily_A[adapter_name] = lily_A if lily_A is not None else nn.Linear(self.in_features, r, bias=False)
        self.lily_B[adapter_name] = lily_B if lily_B is not None else nn.Linear(r, self.out_features, bias=False)
        self.lily_router[adapter_name] = nn.Linear(r, num_B, bias=False)
        self.num_A = num_A
        self.num_B = num_B
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
        lily_scaling: float = 1.0,
        lily_dropout: float = 0.0,
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
            lily_scaling=lily_scaling,
            lily_dropout=lily_dropout,
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
        device = self.lily_B[adapter].weight.device
        dtype = self.lily_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lily_A[adapter].weight
        weight_B = self.lily_B[adapter].weight
        router = self.lily_router[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()
        
        # the delta weight is dynamically computed based on the input tensor
        hidden = self.lily_A(x)
        router_logits = torch.matmul(hidden, router.t())
        router_probability = F.softmax(router_logits, dim=-1) 
        expert_probabilities = router_probability.mean(dim=(0, 1)) 
        weight_B = einops.rearrange(weight_B, "(e i) o -> e i o", e=self.num_B)
        combined_B = torch.einsum("e,eio->io", expert_probabilities, weight_B)
        output_tensor = torch.matmul(weight_A.t(), combined_B) * self.lily_scaling # get the approximated weight update

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lily_A[adapter].weight.data = weight_A.to(dtype)
            self.lily_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

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
            dropout = self.lily_dropout
            scaling = self.lily_scaling
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