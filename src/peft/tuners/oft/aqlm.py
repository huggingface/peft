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

from peft.import_utils import is_aqlm_available
from peft.tuners.oft.layer import OFTLayer
from peft.tuners.tuners_utils import BaseTunerLayer


if is_aqlm_available():
    from aqlm import QuantizedLinear


class AqlmOFTLinear(torch.nn.Module, OFTLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        oft_block_size: int = 32,
        module_dropout: float = 0.0,
        init_weights: bool = True,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        use_cayley_neumann: bool = False,
        num_cayley_neumann_terms: int = 5,
        **kwargs,
    ):
        super().__init__()
        OFTLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            oft_block_size=oft_block_size,
            module_dropout=module_dropout,
            init_weights=init_weights,
            coft=coft,
            eps=eps,
            block_share=block_share,
            use_cayley_neumann=use_cayley_neumann,
            num_cayley_neumann_terms=num_cayley_neumann_terms,
        )

    def forward(self, x: torch.Tensor):
        # note: logic differs from default Linear because merging is not supported
        if self.disable_adapters:
            return self.base_layer(x)

        for active_adapter in self.active_adapters:
            if active_adapter not in self.oft_R.keys():
                continue
            oft_R = self.oft_R[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = x.dtype
                x = self._cast_input_dtype(x, oft_R.weight.dtype)

            x = oft_R(x)

        result = self.base_layer(x)
        if requires_conversion:
            result = result.to(expected_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "oft." + rep


def dispatch_aqlm(
    target: torch.nn.Module,
    adapter_name: str,
    **kwargs: Any,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if is_aqlm_available() and isinstance(target_base_layer, QuantizedLinear):
        new_module = AqlmOFTLinear(target, adapter_name, **kwargs)
        target.qweight = target_base_layer.codes

    return new_module
