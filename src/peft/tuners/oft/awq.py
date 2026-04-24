# Copyright 2025-present the HuggingFace Inc. team.
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
from typing import Any, Optional

import torch

from peft.import_utils import is_gptqmodel_available
from peft.tuners.oft.layer import OFTLayer
from peft.tuners.tuners_utils import BaseTunerLayer

from .config import OFTConfig


class AwqOFTLinear(torch.nn.Module, OFTLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        config: OFTConfig,
        r: int = 0,
        **kwargs,
    ):
        super().__init__()
        OFTLayer.__init__(self, base_layer)

        # self.base_layer and self.quant_linear_module are the same; we need the former for consistency and the latter
        # for backwards compatibility
        self.quant_linear_module = base_layer

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            config=config,
        )

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            result = self.quant_linear_module(x)
            return result

        for active_adapter in self.active_adapters:
            if active_adapter not in self.oft_R.keys():
                continue
            oft_R = self.oft_R[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = x.dtype
                x = self._cast_input_dtype(x, oft_R.weight.dtype)

            x = oft_R(x)
            if requires_conversion:
                x = x.to(expected_dtype)

        result = self.quant_linear_module(x)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "oft." + rep


def dispatch_awq(
    target: torch.nn.Module,
    adapter_name: str,
    **kwargs: Any,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if is_gptqmodel_available():
        from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMQuantLinear

        if isinstance(target_base_layer, AwqGEMMQuantLinear):
            new_module = AwqOFTLinear(target, adapter_name, **kwargs)
            target.qweight = target_base_layer.qweight

    return new_module
