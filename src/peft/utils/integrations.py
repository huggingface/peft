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

import copy
import functools
import re
from contextlib import contextmanager
from typing import Any, Literal, Optional

import packaging.version
import torch
import transformers
from torch import nn

from peft.import_utils import is_transformers_ge_v5


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
    """Helper function to dequantize 4bit or 8bit bnb weights."""
    import bitsandbytes as bnb

    device = weight.device

    cls_name = weight.__class__.__name__
    if cls_name == "Params4bit":
        dequantized = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
        return dequantized

    # 8bit case
    if state is None:
        raise ValueError(
            "No `state` was passed for bnb 8bit quantized weights. Please open an issue on the PEFT repository and "
            "report the error: https://github.com/huggingface/peft/issues"
        )

    if state.SCB is None:
        state.SCB = weight.SCB

    if hasattr(bnb.functional, "int8_vectorwise_dequant"):
        # Use bitsandbytes API if available (requires v0.45.0+)
        dequantized = bnb.functional.int8_vectorwise_dequant(weight.data, state.SCB)
    else:
        # Multiply by (scale/127) to dequantize.
        dequantized = weight.data * state.SCB.view(-1, 1) * 7.874015718698502e-3

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
        if hasattr(cache, "layers"):
            # new transformers uses cache.layers (>v4.55)
            layer = cache.layers[idx]
            layer.keys = layer.keys.to(layer_device)
            layer.values = layer.values.to(layer_device)
        else:
            # old transformers uses cache.{key,value}_cache (<=v4.55)
            # TODO: remove if we drop support for transformers <= 4.55
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


##################################
# TRANSFORMERS WEIGHT CONVERSION #
##################################


if is_transformers_ge_v5:
    # TODO: remove conditional when transformers < 5.0 is no longer supported
    from transformers.conversion_mapping import _MODEL_TO_CONVERSION_PATTERN, get_checkpoint_conversion_mapping, get_model_conversion_mapping
    from transformers.core_model_loading import WeightConverter, WeightRenaming, dot_natural_key, rename_source_key
    from .transformers_weight_conversion import build_peft_weight_mapping
 
    # The main reason we have to explicit this is because the conversion mapping
    # has the full layer name, while the config do not. We coould regex match but
    # this is more explicit and less error prone.
    # Note: this is used in PEFT, changing it requires coordiation.
    _MOE_TARGET_MODULE_MAPPING: dict[str, dict[str, str]] = {
        "mixtral": {
            "gate": "gate.weight",
            "w1": "gate_up_proj",
            "w3": "gate_up_proj",
            "w2": "down_proj",
        },
        "qwen2_moe": {
            "gate": "gate.weight",
            "gate_proj": "gate_up_proj",
            "up_proj": "gate_up_proj",
            "down_proj": "down_proj",
        },
    }

    # Note: this is used in PEFT, changing it requires coordiation.
    _MOE_FUSED_TARGETS: dict[str, dict[str, set[str]]] = {
        # use lists for dict values to ensure stable order
        "mixtral": {"gate_up_proj": ["w1", "w3"]},
        "qwen2_moe": {"gate_up_proj": ["gate_proj", "up_proj"]},
    }


    def _convert_peft_config_moe(peft_config, model_type: str):
        """
        Convert the PEFT config of MoE models whose architecture changed from transformers v4 to v5

        Since the model architecture changed, the targets have to updated accordingly. Moreover, when weights are
        fused, it requires updating the rank and alpha values of those parameters.
        """
        base_model_type = _MODEL_TO_CONVERSION_PATTERN.get(model_type, None)
        if base_model_type is None:
            return peft_config

        target_module_mapping = _MOE_TARGET_MODULE_MAPPING[base_model_type]
        fused_targets = _MOE_FUSED_TARGETS.get(base_model_type, {})

        peft_config.target_parameters = set(peft_config.target_parameters or [])
        peft_config.target_modules = set(peft_config.target_modules or [])
        if not hasattr(peft_config, "rank_pattern") or peft_config.rank_pattern is None:
            peft_config.rank_pattern = {}
        if not hasattr(peft_config, "alpha_pattern") or peft_config.alpha_pattern is None:
            peft_config.alpha_pattern = {}

        new_target_parameters = peft_config.target_parameters.copy()
        remaining_target_modules = set()
        matched_targets: dict[str, set[str]] = {new_name: set() for new_name in fused_targets}

        for target in peft_config.target_modules:
            mapped_new_name = None
            mapped_old_name = None
            for old_name, new_name in target_module_mapping.items():
                if (target == old_name) or target.endswith(f".{old_name}"):
                    mapped_new_name = new_name
                    mapped_old_name = old_name
                    break

            if mapped_new_name is None:
                remaining_target_modules.add(target)
                continue

            new_target_parameters.add(mapped_new_name)
            if mapped_new_name in fused_targets and mapped_old_name is not None:
                matched_targets.setdefault(mapped_new_name, set()).add(mapped_old_name)

        for new_name, required_old_targets in fused_targets.items():
            present_targets = matched_targets.get(new_name, set())
            if 0 < len(present_targets) < len(required_old_targets):
                missing = ", ".join(sorted(required_old_targets - present_targets))
                present = ", ".join(sorted(present_targets))
                raise ValueError(
                    f"Cannot convert PEFT target(s) {present} without also targeting {missing} because they are fused "
                    f"into {new_name}."
                )

            if len(present_targets) == len(required_old_targets) and len(required_old_targets) > 1:
                # TODO: if there is already a rank or alpha pattern for this module, we should update that instead, but
                # it's not trivial to detect a match here
                peft_config.rank_pattern[rf".*\.{re.escape(new_name)}"] = peft_config.r * len(required_old_targets)
                # Preserve per-branch LoRA scaling after fusion.
                # Example: w1 + w3 => r doubles, so alpha must also double to keep alpha/r unchanged.
                peft_config.alpha_pattern[rf".*\.{re.escape(new_name)}"] = peft_config.lora_alpha * len(
                    required_old_targets
                )

        peft_config.target_parameters = new_target_parameters
        peft_config.target_modules = remaining_target_modules

        return peft_config

    def convert_peft_config_for_transformers(peft_config, model: torch.nn.Module, conversions: list[Any] | None):
        """
        Convert the PEFT config of models whose architecture changed from transformers v4 to v5.

        For most models, this requires no changes, this mostly affects some MoE models like Mixtral.
        """
        # If, for any reason, we cannot apply conversion, we just return the PEFT config as is.
        from peft import PeftType  # avoid circular import

        if peft_config.peft_type != PeftType.LORA:
            # weight conversion is currently only supported for LoRA
            return peft_config
        if not hasattr(model, "config"):
            # not a transformer model
            return peft_config
        if not hasattr(model.config, "model_type"):
            # not a transformer model
            return peft_config

        peft_config = copy.deepcopy(peft_config)  # don't mutate the original config
        model_type = getattr(model.config, "model_type", None)
        if get_checkpoint_conversion_mapping(model_type) is not None:
            peft_config = _convert_peft_config_moe(peft_config, model_type)

        return peft_config

    def _convert_to_peft_serialized_keys(
        state_dict: dict[str, torch.Tensor],
        adapter_name: str,
        base_prefix: str = "base_model.model.",
    ) -> dict[str, torch.Tensor]:
        converted = {}
        adapter_suffix = f".{adapter_name}"

        for key, value in state_dict.items():
            # Return PEFT-serialized keys (prefixed), not model state_dict keys.
            if not key.startswith("base_model."):
                key = f"{base_prefix}{key}"

            # For module-backed params: ...lora_A.<adapter>.weight -> ...lora_A.weight
            # For parameter-backed entries: ...<something>.<adapter> -> ...<something>
            if key.endswith(adapter_suffix):
                key = key.removesuffix(adapter_suffix)
            else:
                key_no_suffix, dot, suffix = key.rpartition(".")
                if dot and key_no_suffix.endswith(adapter_suffix):
                    key_no_suffix = key_no_suffix.removesuffix(adapter_suffix)
                    key = f"{key_no_suffix}.{suffix}"
            converted[key] = value

        return converted

    def convert_peft_adapter_state_dict_for_transformers(
        model: torch.nn.Module,
        peft_config,
        adapter_state_dict: dict[str, torch.Tensor],
        adapter_name: str = "default",
    ) -> dict[str, torch.Tensor]:
        """Convert a PEFT adapter state dict to match a transformers v5 weight conversion.

        This function is intended for callers (e.g. `PeftModel.load_adapter`) that need transformers' conversion logic
        without going through `transformer_model.load_adapter`.

        Args:
            model:
                Base model on which the adapter will be loaded.
            peft_config:
                Adapter config.
            adapter_state_dict:
                Adapter weights as loaded from disk.
            adapter_name:
                Adapter name used for conversion of internal target patterns.

        Returns:
            The converted state dict.
        """
        weight_conversions = get_model_conversion_mapping(model)
        peft_weight_mapping = build_peft_weight_mapping(weight_conversions, adapter_name, peft_config=peft_config)

        if not peft_weight_mapping:
            return adapter_state_dict

        converted_state_dict = apply_peft_weight_mapping_to_state_dict(model, adapter_state_dict, peft_weight_mapping)
        converted_state_dict = _convert_to_peft_serialized_keys(converted_state_dict, adapter_name=adapter_name)
        return converted_state_dict


    # TODO remove once PEFT < 0.19 no longer supported
    def apply_peft_weight_mapping_to_state_dict(
        model: torch.nn.Module,
        state_dict: dict[str, torch.Tensor],
        weight_mapping: list[WeightConverter | WeightRenaming],
    ) -> dict[str, torch.Tensor]:
        """
        Function that exposes the weight conversion to the state dict. This is required to be called within PEFT to apply
        weight conversion there without having to duplicate the whole weight conversion logic.
        """
        renamings = [entry for entry in weight_mapping if isinstance(entry, WeightRenaming)]
        converters = [entry for entry in weight_mapping if isinstance(entry, WeightConverter)]
        pattern_to_converter = {k: converter for converter in converters for k in converter.source_patterns}

        param_name_to_load: dict[str, WeightRenaming | WeightConverter] = {}

        # 1) Rebuild the same "collect by target key + source pattern" structure used by core model loading.
        # We need this because some conversions are many-to-one (e.g. w1/w3 -> gate_up_proj) and must see all inputs.
        for original_key, tensor in sorted(state_dict.items(), key=lambda kv: dot_natural_key(kv[0])):
            renamed_key, source_pattern = rename_source_key(
                original_key,
                renamings,
                converters,
                prefix=None,
                meta_state_dict=None,
            )

            if source_pattern is not None:
                # Each destination key needs its own converter instance because converters keep internal collected state.
                new_converter = copy.deepcopy(pattern_to_converter[source_pattern])
                mapping = param_name_to_load.setdefault(renamed_key, new_converter)
            else:
                mapping = param_name_to_load.setdefault(renamed_key, WeightRenaming(original_key, renamed_key))
                source_pattern = original_key

            mapping.add_tensor(renamed_key, original_key, source_pattern, tensor)

        converted_state_dict = {}
        # 2) Materialize conversion ops (merge/concat/block-diag/permute/...) and emit final tensors.
        for first_param_name, mapping in param_name_to_load.items():
            realized_value = mapping.convert(
                first_param_name,
                model=model,
                config=model.config,
                hf_quantizer=None,
                loading_info=None,
            )
            for target_name, param in realized_value.items():
                converted_state_dict[target_name] = param[0] if isinstance(param, list) else param

        return converted_state_dict


#######
# END #
#######
