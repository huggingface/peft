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
import warnings
from typing import Any, Optional

import torch
from torch import nn
from transformers.activations import ACT2FN

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .config import PeanutConfig


class PeanutLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names: tuple[str, ...] = ("peanut_A", "peanut_B", "peanut_encoders", "peanut_decoders")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names: tuple[str, ...] = ("r", "depth", "scaling", "act_fn", "res_num")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.depth = {}
        self.res_num = {}
        self.scaling = {}
        self.act_fn = {}
        self.peanut_A = nn.ModuleDict({})
        self.peanut_B = nn.ModuleDict({})
        self.peanut_encoders = nn.ModuleDict({})
        self.peanut_decoders = nn.ModuleDict({})
        self.kwargs = kwargs

        self._disable_adapters = False
        self.merged_adapters = []
        self._cached_delta_weights = {}

        base_layer = self.get_base_layer()

        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
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
        adapter_name: str,
        r: int,
        config: PeanutConfig,
    ) -> None:
        depth = config.depth
        scaling = config.scaling
        act_fn = config.act_fn
        init_weights = config.init_weights
        inference_mode = config.inference_mode

        self.r[adapter_name] = r
        self.depth[adapter_name] = depth
        self.res_num[adapter_name] = depth
        self.scaling[adapter_name] = scaling
        self.act_fn[adapter_name] = act_fn

        self.peanut_A[adapter_name] = nn.Linear(self.out_features, r, bias=False)
        self.peanut_encoders[adapter_name] = nn.ModuleList([nn.Linear(r, r, bias=False) for _ in range(depth)])
        self.peanut_decoders[adapter_name] = nn.ModuleList([nn.Linear(r, r, bias=False) for _ in range(depth)])

        self.peanut_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)

        self.reset_peanut_parameters(adapter_name, init_weights=init_weights)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def reset_peanut_parameters(self, adapter_name: str, init_weights: bool = True):
        if adapter_name not in self.peanut_A:
            return

        nn.init.kaiming_uniform_(self.peanut_A[adapter_name].weight, a=math.sqrt(5))
        for encoder in self.peanut_encoders[adapter_name]:
            nn.init.kaiming_uniform_(encoder.weight, a=math.sqrt(5))
        for decoder in self.peanut_decoders[adapter_name]:
            nn.init.kaiming_uniform_(decoder.weight, a=math.sqrt(5))

        if init_weights:
            nn.init.zeros_(self.peanut_B[adapter_name].weight)
        else:
            nn.init.kaiming_uniform_(self.peanut_B[adapter_name].weight, a=math.sqrt(5))


class Linear(nn.Module, PeanutLayer):
    # PEANuT implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int,
        config: PeanutConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        PeanutLayer.__init__(self, base_layer, **kwargs)

        self._active_adapter = adapter_name

        self.update_layer(adapter_name, r, config=config)

    def _compute_delta_weight(self, adapter: str, base_weight: torch.Tensor) -> torch.Tensor:
        if adapter not in self.peanut_A:
            raise ValueError(f"Adapter {adapter} not found.")

        peanut_A = self.peanut_A[adapter]
        peanut_B = self.peanut_B[adapter]
        non_linear = ACT2FN[self.act_fn[adapter]]
        scaling = self.scaling[adapter]
        res_num = self.res_num[adapter]
        peanut_encoders = self.peanut_encoders[adapter]
        peanut_decoders = self.peanut_decoders[adapter]

        base_weight_t = base_weight.transpose(0, 1).to(peanut_A.weight.dtype)
        delta_w = non_linear(torch.matmul(base_weight_t, peanut_A.weight.t()))

        residuals = []
        for i in range(res_num):
            residuals.append(delta_w)
            encoder = peanut_encoders[i]
            delta_w = non_linear(encoder(delta_w))

        for i in range(res_num):
            decoder = peanut_decoders[i]
            delta_w = non_linear(decoder(delta_w))
            delta_w = delta_w + residuals[res_num - 1 - i]

        delta_w = peanut_B(delta_w)
        return (delta_w * scaling).transpose(0, 1)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        base_weight = self.get_base_layer().weight
        return self._compute_delta_weight(adapter, base_weight)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        base_layer = self.get_base_layer()
        merge_base_weight = base_layer.weight.data.detach().clone()

        for active_adapter in adapter_names:
            if active_adapter not in self.peanut_A:
                continue

            with torch.no_grad():
                delta_weight = self._compute_delta_weight(active_adapter, merge_base_weight)
                delta_weight = delta_weight.to(dtype=base_layer.weight.dtype, device=base_layer.weight.device)

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

            if delta_weight.device.type != "cpu":
                cached_delta_weight = delta_weight.detach().to("cpu")
            else:
                cached_delta_weight = delta_weight.detach()
            self._cached_delta_weights[active_adapter] = cached_delta_weight
            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        base_layer = self.get_base_layer()
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self.peanut_A:
                continue

            delta_weight = self._cached_delta_weights.pop(active_adapter, None)
            if delta_weight is None:
                raise ValueError(f"Cached delta weight for adapter '{active_adapter}' is missing; cannot unmerge.")

            base_layer.weight.data.sub_(
                delta_weight.to(dtype=base_layer.weight.dtype, device=base_layer.weight.device)
            )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)

            if self.active_adapters:
                torch_result_dtype = result.dtype

                for active_adapter in self.active_adapters:
                    if active_adapter not in self.peanut_A:
                        continue

                    delta_weight = self.get_delta_weight(active_adapter)
                    x_cast = self._cast_input_dtype(x, delta_weight.dtype)
                    delta = torch.matmul(x_cast, delta_weight.transpose(0, 1))
                    result = result + delta.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "peanut." + rep
