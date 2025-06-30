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
from typing import Any, Optional

import torch

from peft.import_utils import is_gptqmodel_available
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import get_auto_gptq_quant_linear

from .layer import LoraVariant


class GPTQLoraLinear(torch.nn.Module, LoraLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        use_qalora: bool = False,
        lora_bias: bool = False,
        qalora_group_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        LoraLayer.__init__(self, base_layer)

        if use_dora:
            raise ValueError(f"{self.__class__.__name__} does not support DoRA yet, please set it to False")

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
            use_qalora=use_qalora,
            lora_bias=lora_bias,
            qalora_group_size=qalora_group_size,
        )

    def resolve_lora_variant(self, *, use_dora: bool, use_qalora: bool, **kwargs) -> Optional[LoraVariant]:
        if use_dora and use_qalora:
            raise NotImplementedError(
                f"Dora and QA_lora at the same time is not supported for {self.__class__.__name__} (yet)."
            )
        elif use_dora:
            from .variants import DoraLinearVariant

            variant = DoraLinearVariant()
        elif use_qalora:
            from .variants import QALoraLinearVariant

            variant = QALoraLinearVariant()
        else:
            variant = None
        return variant

    def forward(self, x: torch.Tensor):
        # note: logic differs from default Linear because merging is not supported
        result = self.quant_linear_module(x)

        if self.disable_adapters:
            return result

        lora_A_keys = self.lora_A.keys()

        for active_adapter in self.active_adapters:
            if active_adapter not in lora_A_keys:
                continue
            torch_result_dtype = result.dtype

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            x = self._cast_input_dtype(x, lora_A.weight.dtype)

            if active_adapter not in self.lora_variant:  # vanilla LoRA
                result = result + lora_B(lora_A(dropout(x))) * scaling
            else:
                result = self.lora_variant[active_adapter].forward(
                    self,
                    active_adapter=active_adapter,
                    x=x,
                    result=result,
                )

            result = result.to(torch_result_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

    # TODO: Check if it is better as suggested by users https://github.com/PanQiWei/AutoGPTQ/pull/102
    # def reset_lora_parameters(self, adapter_name):
    #     if adapter_name in self.lora_A.keys():
    #         torch.nn.init.xavier_uniform_(self.lora_A[adapter_name].weight)
    #         torch.nn.init.zeros_(self.lora_B[adapter_name].weight)


def dispatch_gptq(
    target: torch.nn.Module,
    adapter_name: str,
    **kwargs: Any,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    cfg = kwargs.get("gptq_quantization_config", None)

    if is_gptqmodel_available():
        from gptqmodel.nn_modules.qlinear import BaseQuantLinear

        if isinstance(target_base_layer, BaseQuantLinear):
            new_module = GPTQLoraLinear(target, adapter_name, **kwargs)
            target.qweight = target_base_layer.qweight
    else:
        quant_linear = get_auto_gptq_quant_linear(cfg)

        if quant_linear is not None and isinstance(target_base_layer, quant_linear):
            new_module = GPTQLoraLinear(target, adapter_name, **kwargs)
            target.qweight = target_base_layer.qweight

    return new_module
