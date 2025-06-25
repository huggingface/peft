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
from typing import Any, Optional

import torch

from peft.import_utils import is_eetq_available
from peft.tuners.oft.layer import OFTLayer
from peft.tuners.tuners_utils import BaseTunerLayer


if is_eetq_available():
    from eetq import EetqLinear

    class EetqOFTLinear(torch.nn.Module, OFTLayer):
        def __init__(
            self,
            base_layer,
            adapter_name,
            r: int = 0,
            oft_block_size: int = 0,
            module_dropout: float = 0.0,
            init_weights: bool = True,
            coft: bool = False,
            eps: float = 6e-5,
            block_share: bool = False,
            use_cayley_neumann: bool = False,
            num_cayley_neumann_terms: int = 5,
            fan_in_fan_out: bool = False,
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
                oft_block_size=oft_block_size,
                module_dropout=module_dropout,
                init_weights=init_weights,
                coft=coft,
                eps=eps,
                block_share=block_share,
                fan_in_fan_out=fan_in_fan_out,
                use_cayley_neumann=use_cayley_neumann,
                num_cayley_neumann_terms=num_cayley_neumann_terms,
            )

        def forward(self, x: torch.Tensor):
            if self.disable_adapters:
                return self.quant_linear_module(x)

            for active_adapter in self.active_adapters:
                if active_adapter not in self.oft_R.keys():
                    continue
                oft_R = self.oft_R[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = x.dtype
                    x = self._cast_input_dtype(x, oft_R.weight.dtype)

                x = oft_R(x)

            result = self.quant_linear_module(x)
            if requires_conversion:
                result = result.to(expected_dtype)
            return result

        def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
            raise AttributeError("Merging LoRA layers is not supported for Eetq layers.")

        def unmerge(self) -> None:
            raise AttributeError("Unmerging LoRA layers is not supported for Eetq layers.")

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "oft." + rep


def dispatch_eetq(
    target: torch.nn.Module,
    adapter_name: str,
    **kwargs: Any,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if is_eetq_available() and isinstance(target_base_layer, EetqLinear):
        new_module = EetqOFTLinear(target, adapter_name, **kwargs)
        target.weight = target_base_layer.weight

        if hasattr(target, "bias"):
            target.bias = target_base_layer.bias

    return new_module
