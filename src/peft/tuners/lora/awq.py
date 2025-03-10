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

from awq.utils.packing_utils import unpack_awq, reverse_awq_order
from awq.modules.linear.gemm import WQLinear_GEMM


def dequantize(qweight, qzeros, scales, bits, group_size):
    # Unpack the qweight and qzeros tensors
    iweight, izeros = unpack_awq(qweight, qzeros, bits)
    # Reverse the order of the iweight and izeros tensors
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)

    # overflow checks
    iweight = torch.bitwise_and(iweight, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)

    # fp16 weights
    scales = scales.repeat_interleave(group_size, dim=0)
    izeros = izeros.repeat_interleave(group_size, dim=0)
    iweight = (iweight - izeros) * scales

    return iweight, izeros, scales


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

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        from peft.utils.other import transpose
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, True) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor



    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        from peft.tuners.tuners_utils import check_adapters_to_merge
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                delta_weight = self.get_delta_weight(active_adapter)
                original_linear_layer, zeros, scales = dequantize(
                    base_layer.qweight,
                    base_layer.qzeros,
                    base_layer.scales,
                    base_layer.w_bit,
                    base_layer.group_size,
                )
                unquantized_weights = delta_weight + original_linear_layer
                if safe_merge:
                    if not torch.isfinite(unquantized_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                l = torch.nn.Linear(
                    original_linear_layer.shape[0],
                    original_linear_layer.shape[1]
                )
                l.weight.data = unquantized_weights.T

                # re-quantize
                new_layer = WQLinear_GEMM.from_linear(
                    linear=l,
                    w_bit=base_layer.w_bit,
                    group_size=base_layer.group_size,
                    scales=scales,
                    zeros=zeros,
                )
                
                if not self.use_dora[active_adapter]:
                    base_layer.qweight.data = new_layer.qweight.data
                else:
                    raise NotImplementedError("Dora merge is not supported for DoRA yet")

                if self.lora_bias[active_adapter]:
                    raise NotImplementedError("Dora merge is not supported for bias yet")

                self.merged_adapters.append(active_adapter)

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
