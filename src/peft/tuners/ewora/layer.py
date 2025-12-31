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
from typing import List, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose


class EworaLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("ewora_As", "ewora_Bs", "ewora_weighting")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "ewora_dropout", "num_experts")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.num_experts = {}
        self.ewora_alpha = {}
        # self.scaling = {}
        self.ewora_dropout = nn.ModuleDict({})

        self.ewora_As = nn.ParameterDict({})
        self.ewora_Bs = nn.ParameterDict({})

        # self.ewora_As = nn.ModuleList()
        # self.ewora_Bs = nn.ModuleList()

        self.ewora_weighting = nn.ModuleDict({})

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        # TODO look at this more closely
        self.use_dora: dict[str, bool] = {}
        # self.lora_magnitude_vector = torch.nn.ModuleDict()  # for DoRA
        # self._caches: dict[str, Any] = {}

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
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
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
            self,
            adapter_name,
            r,
            ewora_alpha,
            ewora_dropout,
            init_weights,
            use_rslora,
            use_dora,
            num_experts
    ):
        if r <= 0:
            raise ValueError(f"`r` should not be 0 or non-negative")

        self.r[adapter_name] = r
        self.num_experts[adapter_name] = num_experts

        if ewora_dropout > 0.0:
            ewora_dropout_layer = nn.Dropout(p=ewora_dropout)
        else:
            ewora_dropout_layer = nn.Identity()

        self.ewora_dropout.update(nn.ModuleDict({adapter_name: ewora_dropout_layer}))
        # Actual trainable parameters

        self.ewora_As[adapter_name] = nn.Parameter(torch.Tensor(num_experts, self.in_features, r), requires_grad=True)
        self.ewora_Bs[adapter_name] = nn.Parameter(torch.Tensor(num_experts, r, self.out_features), requires_grad=True)

        self.ewora_weighting[adapter_name] = nn.Linear(r*num_experts, num_experts, bias=True)

        self.reset_ewora_parameters(adapter_name, init_weights)

        # # implement these init functions later - pissa_init, olora_init, loftq_init
        # if isinstance(init_weights, str) and init_weights.startswith("pissa"):
        #     self.pissa_init(adapter_name, init_weights)
        # elif isinstance(init_weights, str) and init_weights.lower() == "olora":
        #     self.olora_init(adapter_name)
        # elif init_weights == "loftq":
        #     self.loftq_init(adapter_name)
        # elif init_weights:
        #     self.reset_ewora_parameters(adapter_name, init_weights)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        # implement dora init and then uncomment this

        # if use_dora:
        #     self.dora_init(adapter_name)
        #     self.use_dora[adapter_name] = True
        # else:
        #     self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

        # if init_weights:
        #     self.reset_vera_parameters(adapter_name, d_initial=d_initial)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_ewora_parameters(self, adapter_name, init_weights):
        if init_weights is False:
            return

        if adapter_name in self.ewora_As.keys():
            if init_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.ewora_As[adapter_name], a=math.sqrt(5))
                # nn.init.kaiming_uniform_(self.ewora_weighting[adapter_name].weight, a=math.sqrt(5))
                # nn.init.uniform_(self.ewora_weighting[adapter_name].weight, a=-0.05, b=0.05)
                nn.init.uniform_(self.ewora_weighting[adapter_name].weight, a=-1e-2, b=1e-2)
                # nn.init.zeros_(self.ewora_weighting[adapter_name].weight)

            elif init_weights.lower() == "gaussian":

                # TODO fix this
                for adapter_A in self.ewora_As[adapter_name]:
                    nn.init.normal_(adapter_A.weight, std=1 / self.r[adapter_name])
                nn.init.normal_(self.ewora_weighting[adapter_name].weight, std=1 / sum(self.r[adapter_name]))
            else:
                raise ValueError(f"Unknown initialization {init_weights=}")

            nn.init.zeros_(self.ewora_Bs[adapter_name])



class Linear(nn.Linear, EworaLayer):
    def __init__(
            self,
            base_layer,
            adapter_name: str,
            r: int = 8,
            ewora_alpha: int = 1,
            ewora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            is_target_conv_1d_layer: bool = False,
            init_weights: bool = True,
            use_rslora: bool = False,
            use_dora: bool = False,
            num_experts: int = 16,
            **kwargs,
    ) -> None:
        # this gets the init from nn.Linear's super perspective, i.e. nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        EworaLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, ewora_alpha, ewora_dropout, init_weights, use_rslora, use_dora, num_experts)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

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
                if active_adapter not in self.ewora_As.keys():
                    continue

                ewora_As = self.ewora_As[active_adapter]
                ewora_Bs = self.ewora_Bs[active_adapter]
                dropout = self.ewora_dropout[active_adapter]
                weighting = self.ewora_weighting[active_adapter]
                num_experts = self.num_experts[active_adapter]


                bs, seq_len, _ = x.size()
                x = x.to(ewora_As.dtype)
                x = x.unsqueeze(2).expand(-1, -1, num_experts, -1)

                intermediate = torch.einsum('beid, idj -> beij', dropout(x), ewora_As)

                scores = weighting(F.relu(intermediate.reshape(bs, seq_len, -1)))
                # scores = F.softmax(weighting(F.relu(intermediate.reshape(bs, seq_len, -1))))
                # scores = weighting(intermediate.reshape(bs, seq_len, -1))

                final = torch.einsum('beij, ijk -> beik', intermediate, ewora_Bs)
                del intermediate

                final = final * scores.unsqueeze(-1)
                result.add_(final.sum(dim=2))

                del final, scores

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "ewora." + rep
