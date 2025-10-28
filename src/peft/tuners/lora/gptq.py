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
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from peft.import_utils import is_gptqmodel_available
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import get_auto_gptq_quant_linear
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .layer import LoraVariant

# Try to import Triton dequant kernel from GPTQModel
try:
    from gptqmodel.nn_modules.triton_utils.dequant import QuantLinearFunction
except Exception:
    QuantLinearFunction = None


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
            **kwargs,
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
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weight = base_layer.weight.data.clone()
                    if active_adapter not in self.lora_variant:  # vanilla LoRA
                        orig_dtype = base_layer.weight.dtype
                        orig_weight += self.get_delta_weight(active_adapter).to(orig_dtype)
                    else:
                        orig_weight = self.lora_variant[active_adapter].merge_safe(self, active_adapter, orig_weight)

                    if not torch.isfinite(orig_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weight
                else:
                    if active_adapter not in self.lora_variant:  # vanilla LoRA
                        orig_dtype = base_layer.dequantize_weights().dtype
                        base_layer.weight.data += self.get_delta_weight(active_adapter).to(orig_dtype)
                    else:
                        self.lora_variant[active_adapter].merge_unsafe(self, active_adapter, base_layer)
                self.merged_adapters.append(active_adapter)


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

def _dequantize_tritonv2_weight_to_out_in(module: torch.nn.Module, *, dtype: torch.dtype) -> Optional[torch.Tensor]:
    """
    For a GPTQModel TritonV2QuantLinear, reconstruct the full-precision weight matrix
    and return it with shape (out_features, in_features).
    Uses the Triton dequant kernel with an identity input (memory efficient enough per-layer).
    """
    try:
        from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2QuantLinear
    except Exception:
        TritonV2QuantLinear = None

    base = module.get_base_layer() if isinstance(module, BaseTunerLayer) else module
    if TritonV2QuantLinear is None or not isinstance(base, TritonV2QuantLinear):
        return None
    if QuantLinearFunction is None:
        raise RuntimeError("QuantLinearFunction not available. Install GPTQModel with triton extras.")

    dev = base.qweight.device
    in_features, out_features = base.in_features, base.out_features

    # Identity trick: I @ W_deq = W_deq. This yields W_deq in shape (in_features, out_features).
    eye = torch.eye(in_features, device=dev, dtype=dtype)
    W_in_out = QuantLinearFunction.apply(
        eye, base.qweight, base.scales, base.qzeros, base.g_idx,
        base.bits, base.pack_dtype_bits, base.maxq
    )
    # Return transposed to match nn.Linear.weight shape (out, in)
    return W_in_out.T.contiguous()

def _tritonv2_to_float_linear(module: torch.nn.Module, *, dtype: torch.dtype) -> Optional[nn.Linear]:
    W_out_in = _dequantize_tritonv2_weight_to_out_in(module, dtype=dtype)
    if W_out_in is None:
        return None

    base = module.get_base_layer() if isinstance(module, BaseTunerLayer) else module
    lin = nn.Linear(base.in_features, base.out_features, bias=(base.bias is not None), device=W_out_in.device, dtype=dtype)
    lin.weight.data.copy_(W_out_in.to(dtype))
    if base.bias is not None:
        lin.bias.data.copy_(base.bias.detach().to(dtype))
    return lin

# ...existing code...
def _delta_weight_vanilla_lora(mod: "GPTQLoraLinear", adapter: str, dtype: torch.dtype) -> torch.Tensor:
    """
    Fallback delta computation for vanilla LoRA:
      deltaW = scaling * (B.weight @ A.weight)  -> shape (out, in)
    """
    A = mod.lora_A[adapter].weight   # (r, in)
    B = mod.lora_B[adapter].weight   # (out, r)
    scale = mod.scaling[adapter]
    delta = (B @ A) * scale
    return delta.to(dtype)

def merge_gptq_lora_to_linear(model: nn.Module, adapter_names: Optional[list[str]] = None, dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """
    Dequantize TritonV2QuantLinear to float nn.Linear and merge LoRA weights.
    Works even if LoraLayer.get_delta_weight is not available.
    """
    def get_parent(root: nn.Module, name: str) -> Tuple[nn.Module, str]:
        parts = name.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        return parent, parts[-1]

    for full_name, module in list(model.named_modules()):
        if isinstance(module, GPTQLoraLinear):
            float_lin = _tritonv2_to_float_linear(module, dtype=dtype)
            if float_lin is None:
                continue

            names_to_merge = check_adapters_to_merge(module, adapter_names)
            for an in names_to_merge:
                if an not in module.lora_A:
                    continue

                # If a variant (e.g. QALoRA) is active on this adapter, delegate to its merge_unsafe
                if hasattr(module, "lora_variant") and (an in module.lora_variant):
                    # Let the variant write into float_lin.weight in-place
                    module.lora_variant[an].merge_unsafe(module, an, float_lin)
                else:
                    # Vanilla LoRA: compute delta directly (no dependency on get_delta_weight)
                    delta = _delta_weight_vanilla_lora(module, an, dtype)
                    float_lin.weight.data.add_(delta)

            parent, child = get_parent(model, full_name)
            setattr(parent, child, float_lin)

    return model
