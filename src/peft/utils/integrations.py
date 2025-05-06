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

from __future__ import annotations

from contextlib import contextmanager
from typing import Literal

import packaging.version
import torch
import transformers


@contextmanager
def gather_params_ctx(param, modifier_rank: int = 0, fwd_module: torch.nn.Module = None):
    """Call DeepSpeed GatheredParameters context manager if DeepSpeed is enabled, otherwise do nothing."""
    if packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.33.0"):
        from transformers.integrations import is_deepspeed_zero3_enabled
    else:
        from transformers.deepspeed import is_deepspeed_zero3_enabled

    if not is_deepspeed_zero3_enabled():
        yield
        return

    import deepspeed

    with deepspeed.zero.GatheredParameters(param, modifier_rank=modifier_rank, fwd_module=fwd_module):
        yield
    return


def dequantize_module_weight(module: torch.nn.Module) -> torch.nn.Parameter:
    """
    Helper function to dequantize a quantized weight.

    This function should be extended if more quantization schemes are added to the library.

    If the weight is not quantized, it will be returned as is.
    """
    if hasattr(module, "W_q"):  # For handling HQQ quantized weight
        weight = module.dequantize()
        return weight

    weight = module.weight
    if not isinstance(weight, torch.nn.Parameter):
        if isinstance(weight, torch.Tensor):
            # this is an FSDP-specific edge case
            return weight  # type: ignore
        raise TypeError(f"Input weight should be of type nn.Parameter, got {type(weight)} instead")

    cls_name = weight.__class__.__name__
    if cls_name not in ("Params4bit", "Int8Params"):
        return weight

    quant_state = getattr(module, "state", None)
    device = weight.device
    is_cpu = device.type == torch.device("cpu").type
    weight = dequantize_bnb_weight(weight, state=quant_state)  # no-op if not bnb
    if is_cpu:
        # dequantize_bnb_weight for 8bit moves the device in-place, thus we need to move it back to CPU if necessary
        module.weight = module.weight.to(device)
    return weight


def dequantize_bnb_weight(weight: torch.nn.Parameter, state=None):
    """Helper function to dequantize 4bit or 8bit bnb weights.

    Since dequantization is not supported on CPU, the weight will be temporarily moved to CUDA if necessary.
    """
    import bitsandbytes as bnb

    # BNB requires CUDA weights
    device = weight.device
    is_cpu = device.type == torch.device("cpu").type
    if is_cpu:
        weight = weight.to(torch.device("cuda"))

    cls_name = weight.__class__.__name__
    if cls_name == "Params4bit":
        dequantized = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
        if is_cpu:
            dequantized = dequantized.to(device)
        return dequantized

    if state.SCB is None:
        state.SCB = weight.SCB

    im = torch.eye(weight.data.shape[-1]).contiguous().half().to(weight.device)
    im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
    im, Sim = bnb.functional.transform(im, "col32")
    if state.CxB is None:
        state.CxB, state.SB = bnb.functional.transform(weight.data, to_order=state.formatB)
    out32, Sout32 = bnb.functional.igemmlt(im, state.CxB, Sim, state.SB)
    dequantized = bnb.functional.mm_dequant(out32, Sout32, SCim, state.SCB, bias=None).t()
    if is_cpu:
        dequantized = dequantized.to(device)
    return dequantized


def get_bnb_param_type(param: torch.nn.Parameter) -> Literal[False, "4bit", "8bit"]:
    """Returns '4bit' or '8bit' if bitsandbytes parameter, else False"""
    if param.__class__.__name__ == "Params4bit":
        return "4bit"
    if param.__class__.__name__ == "Int8Params":
        return "8bit"
    return False
