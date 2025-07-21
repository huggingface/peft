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

import functools
from contextlib import contextmanager
from typing import Literal, Optional

import packaging.version
import torch
import transformers
from torch import nn

from peft.import_utils import is_xpu_available


def check_deepspeed_zero3_enabled() -> bool:
    if packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.33.0"):
        from transformers.integrations import is_deepspeed_zero3_enabled
    else:
        from transformers.deepspeed import is_deepspeed_zero3_enabled
    return is_deepspeed_zero3_enabled()


@contextmanager
def gather_params_ctx(param, modifier_rank: Optional[int] = 0, fwd_module: torch.nn.Module = None):
    """Call DeepSpeed GatheredParameters context manager if DeepSpeed is enabled, otherwise do nothing."""

    if not check_deepspeed_zero3_enabled():
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
        if torch.cuda.is_available():
            weight = weight.to(torch.device("cuda"))
        elif is_xpu_available():
            weight = weight.to(torch.device("xpu"))

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


##################################
# START: ADAPTED FROM ACCELERATE #
##################################
#
# Modified to support explicitly skipping layer initialization for faster switching between layer states
# (necessary for supporting `nn.MultiHeadAttention` adapters)


@contextmanager
def init_empty_weights(include_buffers: bool = None):
    # adapted from accelerate.big_modeling.py
    with _init_on_device(torch.device("meta"), include_buffers=include_buffers) as f:
        yield f


@contextmanager
def _init_on_device(device: torch.device, include_buffers: bool = None):
    # adapted from accelerate.big_modeling.py
    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        # This works because torch first initializes the parameters with torch.empty, thus not assigning any new memory.
        # Then the parameter is moved to meta device before reset_parameters() is called, which then operates on the
        # meta device, making any subsequent calls to initialization methods no-ops.
        old_register_parameter(module, name, param)
        if (param is not None) and (getattr(_init_on_device, "_skip", False) is not True):
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    # Patch tensor creation
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


@contextmanager
def _skip_init_on_device():
    # context manager to skip the _init_on_device context manager
    old_val = getattr(_init_on_device, "_skip", False)
    try:
        _init_on_device._skip = True
        yield
    finally:
        _init_on_device._skip = old_val


def skip_init_on_device(func):
    """
    Ignore the init_on_device context manager when calling the decorated function.

    This is a narrow use decorator that allows us to avoid initializing on meta device even when we're inside the
    init_empty_weights context.

    """

    # The need for this functionality arose when working on MultiheadAttention, where we have to call _restore_weights
    # repeatedly as parametes are overwritten and need to be re-registered. When using low_cpu_mem_usage=True, as
    # register_parameter is patched inside of the init_empty_weights context, this would result in those parameters
    # suddenly being moved to meta device. Using this decorator allows us to avoid this.
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _skip_init_on_device():
            return func(*args, **kwargs)

    return wrapper


#######
# END #
#######
