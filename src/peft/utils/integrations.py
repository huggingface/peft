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
    elif type(module.weight).__module__.startswith("torchao."):
        # check for torchao without requiring any torchao imports
        weight = module.weight.dequantize()
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

    if hasattr(bnb.functional, "int8_vectorwise_dequant"):
        # Use bitsandbytes API if available (requires v0.45.0+)
        dequantized = bnb.functional.int8_vectorwise_dequant(weight.data, state.SCB)
    else:
        # Multiply by (scale/127) to dequantize.
        dequantized = weight.data * state.SCB.view(-1, 1) * 7.874015718698502e-3

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


# adapted from:
# https://github.com/huggingface/transformers/blob/eab6c491d439e83d5e31c660df6f7e36592eb0a2/src/transformers/generation/utils.py#L1617-L1643
def get_layer_device_map(model):
    """
    Derive the device map for the layers of the model.
    """
    main_device = [d for d in model.hf_device_map.values() if d not in ["cpu", "disk"]][0]

    execution_device_map = {
        name: main_device if device in ["cpu", "disk"] else device for name, device in model.hf_device_map.items()
    }

    if execution_device_map is None:
        return None

    if len(execution_device_map) == 1 and "" in execution_device_map:
        return {idx: execution_device_map[""] for idx in range(model.config.num_hidden_layers)}

    layer_device_map = {}
    for layer in execution_device_map:
        for idx in range(model.config.num_hidden_layers):
            if f".{idx}." in f"{layer}.":
                layer_device_map[idx] = execution_device_map[layer]
                break
    for idx in range(model.config.num_hidden_layers):
        if idx not in layer_device_map:
            raise RuntimeError(f"layer {idx} has not been mapped to a device.")
    return layer_device_map


# adapted from:
# https://github.com/huggingface/transformers/blob/eab6c491d439e83d5e31c660df6f7e36592eb0a2/src/transformers/cache_utils.py#L1159-L1179
def map_cache_to_layer_device_map(model, cache) -> None:
    """
    Ensure that the key and value cache of the model are on the same device as their corresponding layers.
    """
    if not (isinstance(cache, transformers.Cache) and hasattr(model, "hf_device_map")):
        return

    if isinstance(cache, transformers.EncoderDecoderCache):
        map_cache_to_layer_device_map(model, cache.self_attention_cache)
        return

    layer_device_map = get_layer_device_map(model)
    for idx in range(model.config.num_hidden_layers):
        layer_device = layer_device_map[idx]
        cache.key_cache[idx] = cache.key_cache[idx].to(layer_device)
        cache.value_cache[idx] = cache.value_cache[idx].to(layer_device)
