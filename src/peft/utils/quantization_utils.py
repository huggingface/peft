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

import copy
from abc import ABC

import torch
from torch import nn

from peft.import_utils import (
    is_aqlm_available,
    is_bnb_4bit_available,
    is_bnb_available,
    is_eetq_available,
    is_gptqmodel_available,
    is_hqq_available,
    is_inc_available,
    is_te_pytorch_available,
    is_torchao_available,
)
from peft.utils.integrations import dequantize_bnb_weight


class Quantizationbackend(ABC):
    """Base class for quantization backends used in PEFT layers.

    A `Quantizationbackend` encapsulates the quantization-specific logic for accessing and modifying a base layer's
    weights. By attaching an instance to a PEFT layer (`layer.quantization_backend = backend`), the PEFT layer's
    `merge` / `unmerge` methods can work with quantized weights without needing a dedicated subclass.

    Subclasses must implement `get_base_weight` and `set_base_weight` if they support merging.
    """

    # Whether this backend supports merge/unmerge operations
    supports_merge: bool = False
    # user readable name
    backend_name: str = ""

    def get_base_weight(self, base_layer: nn.Module) -> torch.Tensor:
        """Return the dequantized weight tensor from the base layer.

        The returned tensor is a plain float tensor that can be modified. The caller is responsible for cloning if the
        original must be preserved (e.g. for safe_merge).

        Args:
            base_layer: The quantized base layer module (e.g. `bnb.nn.Linear8bitLt`).

        Returns:
            Dequantized weight as a `torch.Tensor`.
        """
        raise NotImplementedError

    def set_base_weight(self, peft_layer, weight_data: torch.Tensor) -> None:
        """Store the (modified) weight back into the base layer, re-quantizing as needed.

        Some backends (e.g. HQQ) replace `peft_layer.base_layer` entirely with a new module.

        Args:
            peft_layer:
                The PEFT layer that owns the base layer. Provides access to `peft_layer.get_base_layer()` and allows
                replacing `peft_layer.base_layer` when needed (e.g. HQQ).
            weight_data:
                The modified weight tensor to store.
        """
        raise NotImplementedError

    def maybe_clone_base_result(self, x: torch.Tensor) -> torch.Tensor:
        # As per Tim Dettmers, for 4bit bnb, we need to defensively clone here. The reason is that in some cases, an
        # error can occur that backprop does not work on a manipulated view. This issue may be solved with newer PyTorch
        # versions but this would need extensive testing to be sure.
        # https://github.com/huggingface/peft/blob/58169b5e817e6053977ce82334d6c8d5ae3bed54/src/peft/tuners/lora/bnb.py#L524-L529
        # By default, don't clone.
        return x


class ForwardOnlyQuantizationbackend(Quantizationbackend):
    """Backend for quantization frameworks that do not support merging of weights.

    Used for GPTQ etc. The forward pass works without modification because `base_layer(x)` already dispatches to the
    quantized kernel. Merging and unmerging are either not implemented or not possible (e.g. no dequantize support).
    """

    supports_merge = False

    def __init__(self, backend_name: str = "") -> None:
        self.backend_name = backend_name

    def get_base_weight(self, base_layer: nn.Module) -> torch.Tensor:
        raise TypeError(
            f"Dequantization is not supported for {self.backend_name} layers. Merging/unmerging is not available."
        )

    def set_base_weight(self, peft_layer, weight_data: torch.Tensor) -> None:
        raise TypeError(
            f"Requantization is not supported for {self.backend_name} layers. Merging/unmerging is not available."
        )


class Bnb8bitBackend(Quantizationbackend):
    """Backend for bitsandbytes 8-bit quantized layers (`bnb.nn.Linear8bitLt`)."""

    supports_merge = True
    backend_name = "bnb 8bit"

    def get_base_weight(self, base_layer: nn.Module) -> torch.Tensor:
        weight = base_layer.weight
        state = base_layer.state
        if state.SCB is None:
            state.SCB = weight.SCB

        return dequantize_bnb_weight(weight, state=state)

    def set_base_weight(self, peft_layer, weight_data: torch.Tensor) -> None:
        import bitsandbytes as bnb

        base_layer = peft_layer.get_base_layer()
        weight = base_layer.weight
        base_layer.weight = bnb.nn.Int8Params(
            weight_data.to("cpu"), requires_grad=False, has_fp16_weights=weight.has_fp16_weights
        ).to(weight.device)
        base_layer.state.reset_grads()


class Bnb4bitBackend(Quantizationbackend):
    """Backend for bitsandbytes 4-bit quantized layers (`bnb.nn.Linear4bit`)."""

    supports_merge = True
    backend_name = "bnb 4bit"

    def get_base_weight(self, base_layer: nn.Module) -> torch.Tensor:
        weight = base_layer.weight
        return dequantize_bnb_weight(weight, state=weight.quant_state)

    def set_base_weight(self, peft_layer, weight_data: torch.Tensor) -> None:
        import bitsandbytes as bnb

        base_layer = peft_layer.get_base_layer()
        weight = base_layer.weight
        kwargs = weight.__dict__
        if "bnb_quantized" in kwargs:
            kwargs["bnb_quantized"] = False
        kwargs["requires_grad"] = False
        kwargs.pop("data", None)
        # torch.compile can introduce attributes preceded by '_', remove them
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        base_layer.weight = bnb.nn.Params4bit(weight_data.to("cpu"), **kwargs).to(weight.device)

    def maybe_clone_base_result(self, x: torch.Tensor) -> torch.Tensor:
        # As per Tim Dettmers, for 4bit bnb, we need to defensively clone here. The reason is that in some cases, an
        # error can occur that backprop does not work on a manipulated view. This issue may be solved with newer PyTorch
        # versions but this would need extensive testing to be sure.
        # https://github.com/huggingface/peft/blob/58169b5e817e6053977ce82334d6c8d5ae3bed54/src/peft/tuners/lora/bnb.py#L524-L529
        return x.clone()


class HqqBackend(Quantizationbackend):
    """Backend for HQQ quantized layers (`hqq.core.quantize.HQQLinear`)."""

    supports_merge = True
    backend_name = "hqq"

    def get_base_weight(self, base_layer: nn.Module) -> torch.Tensor:
        return base_layer.dequantize()

    def set_base_weight(self, peft_layer, weight_data: torch.Tensor) -> None:
        from hqq.core.quantize import HQQLinear

        base_layer = peft_layer.get_base_layer()
        quant_config = {**copy.deepcopy(base_layer.quant_config), "offload_meta": base_layer.offload_meta}

        new_hqq_layer = HQQLinear(None, quant_config, compute_dtype=base_layer.compute_dtype, device=base_layer.device)
        quant_config.pop("offload_meta", None)
        new_hqq_layer.quantize(weight_data, **quant_config)

        # HQQ requires replacing the entire base layer
        peft_layer.base_layer = new_hqq_layer


class TorchaoBackend(Quantizationbackend):
    """Backend for torchao quantized layers.

    Args:
        get_apply_tensor_subclass:
            A callable that returns the tensor subclass to use for re-quantization. This is stored from the original
            quantization config and is needed to re-quantize after merging.
    """

    supports_merge = True
    backend_name = "torchao"

    def __init__(self, get_apply_tensor_subclass):
        self.get_apply_tensor_subclass = get_apply_tensor_subclass

    def get_base_weight(self, base_layer: nn.Module) -> torch.Tensor:
        weight = base_layer.weight
        try:
            dequantized = weight.dequantize()
        except NotImplementedError as exc:
            msg = (
                f"Weights of type {type(weight).__name__} do not support dequantization (yet), which is needed to "
                "support merging."
            )
            raise NotImplementedError(msg) from exc
        return dequantized

    def set_base_weight(self, peft_layer, weight_data: torch.Tensor) -> None:
        from torchao import quantize_

        base_layer = peft_layer.get_base_layer()
        # torchao does not support directly mutating the data, so we need to delete and re-assign the weight, then
        # re-quantize through quantize_ which takes a module as input.
        del base_layer.weight
        base_layer.weight = weight_data
        quantize_(base_layer, self.get_apply_tensor_subclass())


def resolve_quantization_backend(base_layer: nn.Module, **kwargs) -> Quantizationbackend | None:
    """Determine the appropriate quantization backend for a given base layer.

    Inspects the type of `base_layer` and returns the matching `Quantizationbackend` instance, or `None` if the layer
    is not quantized.

    Args:
        base_layer:
            The base layer module to inspect.
        **kwargs:
            Additional keyword arguments needed by specific backends. Currently used:
            - `get_apply_tensor_subclass` (callable): Required for torchao layers.

    Returns:
        A `Quantizationbackend` instance, or `None` if the layer is not quantized.
    """
    # bitsandbytes
    if is_bnb_available():
        import bitsandbytes as bnb

        if isinstance(base_layer, bnb.nn.Linear8bitLt):
            return Bnb8bitBackend()

    if is_bnb_4bit_available():
        import bitsandbytes as bnb

        if isinstance(base_layer, bnb.nn.Linear4bit):
            return Bnb4bitBackend()

    # HQQ
    if is_hqq_available():
        from hqq.core.quantize import HQQLinear

        if isinstance(base_layer, HQQLinear):
            return HqqBackend()

    # torchao
    if is_torchao_available():
        from torchao.dtypes import AffineQuantizedTensor
        from torchao.quantization import LinearActivationQuantizedTensor

        if hasattr(base_layer, "weight") and isinstance(
            base_layer.weight, (AffineQuantizedTensor, LinearActivationQuantizedTensor)
        ):
            get_apply_tensor_subclass = kwargs.get("get_apply_tensor_subclass")
            if get_apply_tensor_subclass is None:
                raise ValueError(
                    "torchao quantized layer detected but 'get_apply_tensor_subclass' was not provided. "
                    "This is required for merge/unmerge support."
                )
            backend = TorchaoBackend(get_apply_tensor_subclass)
            if isinstance(base_layer.weight, LinearActivationQuantizedTensor):
                backend.supports_merge = False
            return backend

    # AQLM
    if is_aqlm_available():
        from aqlm import QuantizedLinear

        if isinstance(base_layer, QuantizedLinear):
            return ForwardOnlyQuantizationbackend("aqlm")

    # EETQ
    if is_eetq_available():
        from eetq import EetqLinear

        if isinstance(base_layer, EetqLinear):
            return ForwardOnlyQuantizationbackend("eetq")

    # GPTQ (via gptqmodel)
    if is_gptqmodel_available():
        from gptqmodel.nn_modules.qlinear import BaseQuantLinear

        if isinstance(base_layer, BaseQuantLinear):
            return ForwardOnlyQuantizationbackend("gptq")

    # AWQ (via gptqmodel)
    if is_gptqmodel_available():
        from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMQuantLinear

        if isinstance(base_layer, AwqGEMMQuantLinear):
            return ForwardOnlyQuantizationbackend("awq")

    # INC (Intel Neural Compressor)
    if is_inc_available():
        from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import PatchedLinear

        if isinstance(base_layer, PatchedLinear):
            return ForwardOnlyQuantizationbackend("inc")

    if is_te_pytorch_available():
        import transformer_engine as te

        if isinstance(base_layer, (te.pytorch.LayerNormLinear, te.pytorch.LayerNormMLP, te.pytorch.Linear)):
            return ForwardOnlyQuantizationbackend("te")

    # Not quantized
    return None


def quantization_extra_repr(module: nn.Module) -> str:
    """Extend the repr of the PEFT module.

    extra_repr is a method on nn.Modules that can be used to extend the repr of a module.
    """
    base = nn.Module.extra_repr(module)
    parts = []
    if base:
        parts.append(base)
    if getattr(module, "quantization_backend", None):
        parts.append(f"quantization_backend='{module.quantization_backend.backend_name}'")
    return ", ".join(parts)
