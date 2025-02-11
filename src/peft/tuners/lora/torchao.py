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
from __future__ import annotations

import warnings
from typing import Any, Optional

import torch

# from torch import nn
from peft.import_utils import is_torchao_available
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .config import LoraConfig
from .layer import Linear


class TorchaoLoraLinear(Linear):
    """LoRA layer implementation for Linear layers using torchao data"""

    def __init__(self, *args, get_apply_tensor_subclass, **kwargs):
        # this is not strictly necessary, as kwargs are stored either way, but we want to error early if
        # get_apply_tensor_subclass is missing.
        if kwargs.get("lora_bias", False):
            raise ValueError(f"{self.__class__.__name__} does not support lora_bias yet, set it to False")

        super().__init__(*args, **kwargs)
        self.get_apply_tensor_subclass = get_apply_tensor_subclass
        self._check_dtype_supported()

    def _check_dtype_supported(self):
        # TODO: Not required once int4_weight_only is properly supported by torchao
        base_layer = self.get_base_layer()
        weight = base_layer.weight
        # pytest tests/test_gpu_examples.py::PeftTorchaoGPUTests::test_causal_lm_training_single_gpu_torchao_0_int8_weight_only
        if (
            # torchao 0.7.0+
            (hasattr(weight, "tensor_impl") and (weight.tensor_impl.data.dtype != torch.int8))
            or
            # torchao < 0.7.0
            (hasattr(weight, "layout_tensor") and (weight.layout_tensor.data.dtype != torch.int8))
        ):
            raise ValueError(f"{type(self).__name__} only supports int8 weights for now.")

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        from torchao import quantize_

        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        self._check_dtype_supported()

        base_layer = self.get_base_layer()
        weight = base_layer.weight

        for active_adapter in adapter_names:
            try:
                weight = weight.dequantize()
            except NotImplementedError as exc:
                msg = (
                    f"Weights of type {type(weight).__name__} do not support dequantization (yet), which is needed to "
                    "support merging."
                )
                raise NotImplementedError(msg) from exc

            if safe_merge and not torch.isfinite(weight).all():
                raise ValueError(
                    f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                )

            weight += self.get_delta_weight(active_adapter)
            # TODO: once (if) torchao supports directly mutating the data, use that instead.
            del base_layer.weight
            base_layer.weight = weight
            quantize_(base_layer, self.get_apply_tensor_subclass())
            del weight

            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        from torchao import quantize_

        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self.lora_A.keys():
                continue

            base_layer = self.get_base_layer()
            weight = base_layer.weight
            try:
                weight = weight.dequantize()
            except NotImplementedError as exc:
                msg = (
                    f"Weights of type {type(weight).__name__} do not support dequantization (yet), which is needed to "
                    "support unmerging."
                )
                raise NotImplementedError(msg) from exc

            weight -= self.get_delta_weight(active_adapter)
            # We go through a dummy module because overriding the weight.data does not work, the tensor retains the old
            # data. Therefore, we need to go through quantize_, which takes a module as input, and we need to delete and
            # re-assign the weight.
            # TODO: once (if) torchao supports directly mutating the data, use that instead.
            del base_layer.weight
            base_layer.weight = weight
            quantize_(base_layer, self.get_apply_tensor_subclass())
            del weight

    def __repr__(self) -> str:
        rep = super().__repr__()
        return rep.replace("lora.Linear", f"lora.{self.__class__.__name__}")


def dispatch_torchao(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs: Any,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if not hasattr(target_base_layer, "weight"):
        return new_module

    if not is_torchao_available():
        return new_module

    from torchao.dtypes import AffineQuantizedTensor
    from torchao.quantization import LinearActivationQuantizedTensor

    if isinstance(target_base_layer.weight, (AffineQuantizedTensor, LinearActivationQuantizedTensor)):
        new_module = TorchaoLoraLinear(target, adapter_name, **kwargs)

    return new_module
