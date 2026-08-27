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
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer, _get_in_out_features

from .config import EworaConfig


class EworaLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("ewora_As", "ewora_Bs", "ewora_weighting")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "ewora_dropout", "num_experts")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.num_experts = {}
        self.ewora_dropout = nn.ModuleDict({})
        self.ewora_As = nn.ParameterDict({})
        self.ewora_Bs = nn.ParameterDict({})
        self.ewora_weighting = nn.ModuleDict({})

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        in_features, out_features = _get_in_out_features(base_layer)
        if in_features is None:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(self, adapter_name, r, config: EworaConfig):
        if r <= 0:
            raise ValueError("`r` should be a positive integer")

        num_experts = config.num_experts
        self.r[adapter_name] = r
        self.num_experts[adapter_name] = num_experts

        if config.ewora_dropout > 0.0:
            ewora_dropout_layer = nn.Dropout(p=config.ewora_dropout)
        else:
            ewora_dropout_layer = nn.Identity()
        self.ewora_dropout[adapter_name] = ewora_dropout_layer

        # Actual trainable parameters
        self.ewora_As[adapter_name] = nn.Parameter(torch.Tensor(num_experts, self.in_features, r), requires_grad=True)
        self.ewora_Bs[adapter_name] = nn.Parameter(torch.Tensor(num_experts, r, self.out_features), requires_grad=True)
        self.ewora_weighting[adapter_name] = nn.Linear(r * num_experts, num_experts, bias=True)

        self.reset_ewora_parameters(adapter_name, config.init_weights)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_ewora_parameters(self, adapter_name, init_weights=True):
        if adapter_name in self.ewora_As.keys():
            # initialize A the same way as the default for nn.Linear
            # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
            nn.init.kaiming_uniform_(self.ewora_As[adapter_name], a=math.sqrt(5))
            nn.init.uniform_(self.ewora_weighting[adapter_name].weight, a=-1e-2, b=1e-2)
            if init_weights:
                # B is zero, so a freshly-created adapter is an identity transform
                nn.init.zeros_(self.ewora_Bs[adapter_name])
            else:
                # small non-identity initialization, mainly used for testing (EWoRA applies no scaling and
                # sums over experts, so a large B would be unstable)
                nn.init.normal_(self.ewora_Bs[adapter_name], std=0.02)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        raise NotImplementedError(
            "EWoRA dynamically weights its experts at the forward pass, so it has no static delta weight."
        )

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError(
            "EWoRA cannot be merged into the base weights because its experts are dynamically weighted per input."
        )

    def unmerge(self) -> None:
        raise NotImplementedError("EWoRA does not support merging, so there is nothing to unmerge.")


class Linear(nn.Linear, EworaLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        config: EworaConfig,
        r: int,
        **kwargs,
    ) -> None:
        # this gets the init from nn.Linear's super perspective, i.e. nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        EworaLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, config)

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

                x = x.to(ewora_As.dtype)
                # broadcast the input over the experts: (..., in_features) -> (..., num_experts, in_features)
                x = x.unsqueeze(-2).expand(*x.shape[:-1], num_experts, x.shape[-1])

                # einsum indices: ... = leading dims (batch, sequence, ...), i = expert,
                # d = in_features, j = rank, k = out_features
                intermediate = torch.einsum("...id, idj -> ...ij", dropout(x), ewora_As)
                scores = weighting(F.relu(intermediate.reshape(*intermediate.shape[:-2], -1)))
                final = torch.einsum("...ij, ijk -> ...ik", intermediate, ewora_Bs)

                # the learned routing scores act as the input-dependent scaling of the expert outputs, taking the
                # place of LoRA's static alpha/r factor
                final = final * scores.unsqueeze(-1)
                result = result + final.sum(dim=-2)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "ewora." + rep
