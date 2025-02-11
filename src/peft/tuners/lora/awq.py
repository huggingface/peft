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
import importlib.metadata as importlib_metadata
from typing import Any, Optional

import packaging.version
import torch

from peft.import_utils import is_auto_awq_available
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer


class AwqLoraLinear(torch.nn.Module, LoraLayer):
    def __init__(
        self,
        base_layer,
        adapter_name,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ):
        if use_dora:
            raise ValueError(f"{self.__class__.__name__} does not support DoRA yet, please set it to False")

        super().__init__()
        LoraLayer.__init__(self, base_layer)

        # self.base_layer and self.quant_linear_module are the same; we need the former for consistency and the latter
        # for backwards compatibility
        self.quant_linear_module = base_layer

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )

    def forward(self, x: torch.Tensor):
        result = self.quant_linear_module(x)

        if self.disable_adapters:
            return result

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = self._cast_input_dtype(x, lora_A.weight.dtype)

            output = lora_B(lora_A(dropout(x)))
            if requires_conversion:
                output = output.to(expected_dtype)
            output = output * scaling
            result = result + output
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


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

    if is_auto_awq_available():
        from awq.modules.linear import WQLinear_GEMM

        if isinstance(target_base_layer, WQLinear_GEMM):
            # Raise the error only at the dispatch level
            AUTOAWQ_MINIMUM_VERSION = packaging.version.parse("0.2.0")
            version_autoawq = packaging.version.parse(importlib_metadata.version("autoawq"))

            if AUTOAWQ_MINIMUM_VERSION > version_autoawq:
                raise ImportError(
                    f"Found an incompatible version of auto-awq. Found version {version_autoawq}, "
                    f"but only versions above {AUTOAWQ_MINIMUM_VERSION} are supported for PEFT."
                )

            new_module = AwqLoraLinear(target, adapter_name, **kwargs)
            target.qweight = target_base_layer.qweight

    return new_module
