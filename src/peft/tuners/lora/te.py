# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
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

from peft.import_utils import is_te_available
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer

from .config import LoraConfig


if is_te_available():
    import transformer_engine as te


class TELoRA(torch.nn.Module, LoraLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        config: LoraConfig,
        r: int = 0,
        lora_alpha: int = 1,
        **kwargs,
    ):
        if config.use_dora:
            raise ValueError(f"{self.__class__.__name__} does not support DoRA yet, please set it to False")

        super().__init__()
        LoraLayer.__init__(self, base_layer=base_layer, **kwargs)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            config=config,
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            raise ValueError(f"{self.__class__.__name__} does not support mixed_batch_forward yet.")
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            lora_A_keys = self.lora_A.keys()
            for active_adapter in self.active_adapters:
                if active_adapter not in lora_A_keys:
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = self._cast_input_dtype(x, lora_A.weight.dtype)
                result = result + lora_B(lora_A(dropout(x))) * scaling

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


def dispatch_transformer_engine(
    target: torch.nn.Module,
    adapter_name: str,
    config: LoraConfig,
    **kwargs: Any,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if is_te_available() and isinstance(
        target_base_layer, (te.pytorch.LayerNormLinear, te.pytorch.LayerNormMLP, te.pytorch.Linear)
    ):
        new_module = TELoRA(target, adapter_name, config=config, **kwargs)

    return new_module
