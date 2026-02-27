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
from transformers.conversion_mapping import _MODEL_TO_CONVERSION_PATTERN, get_checkpoint_conversion_mapping
from transformers.core_model_loading import (
    Concatenate,
    ConversionOps,
    MergeModulelist,
    Transpose,
    WeightConverter,
    WeightRenaming,
)


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


#######
# END #
#######

# Transformers weight conversion
#
# With transformers v5, we need to convert some weights to reflect updated model architectures. If users have trained
# PEFT adapters for these models, they also need to be updated. This may require updating the PEFT config too. The
# logic for this is found below. Right now, only LoRA is supported.

# The main reason we have to explicit this is because the conversion mapping
# has the full layer name, while the config do not. We coould regex match but
# this is more explicit and less error prone.
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

_MOE_FUSED_TARGETS: dict[str, dict[str, set[str]]] = {
    "mixtral": {"gate_up_proj": {"w1", "w3"}},
    "qwen2_moe": {"gate_up_proj": {"gate_proj", "up_proj"}},
}


def _convert_peft_config_moe(peft_config, model_type: str):
    base_model_type = _MODEL_TO_CONVERSION_PATTERN.get(model_type, None)
    if base_model_type is None:
        return peft_config

    target_module_mapping = _MOE_TARGET_MODULE_MAPPING[base_model_type]
    fused_targets = _MOE_FUSED_TARGETS.get(base_model_type, {})

    peft_config.target_parameters = set(peft_config.target_parameters or [])
    peft_config.target_modules = set(peft_config.target_modules or [])
    if not hasattr(peft_config, "rank_pattern") or peft_config.rank_pattern is None:
        peft_config.rank_pattern = {}

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
                f"Cannot convert PEFT target(s) {present} without also targeting {missing} because they are fused into {new_name}."
            )

        if len(present_targets) == len(required_old_targets) and len(required_old_targets) > 1:
            peft_config.rank_pattern[rf".*\.{re.escape(new_name)}"] = peft_config.r * len(required_old_targets)

    peft_config.target_parameters = new_target_parameters
    peft_config.target_modules = remaining_target_modules

    return peft_config


def convert_peft_config_for_transformers(peft_config, model: torch.nn.Module, conversions: list[Any] | None):
    # FIXME document this properly
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


def _block_diag_3d(*tensors):
    lora_b_block_diag = []
    for i in range(len(tensors[0])):
        lora_b_block_diag.append(torch.block_diag(tensors[0][i], tensors[1][i]))
    out = torch.stack(lora_b_block_diag, dim=0)
    return out


class PeftConcatenate(Concatenate):
    """Convert per-expert LoRA weights to merged weights.

    When the base weights are fused, e.g. W01 = [W0, W1], the LoRA weights also need to be fused. To achieve this
    correctly, concatenate the LoRA A weights along the r (rank) dimension. This doesn't require a new Operation. But
    for LoRA B, the weights need to be merged in a block diagonal fashion to achieve the correct result.

    To illustrate:

    Before W0' = W0 + A0 @ B0 W1' = W1 + A1 @ B1

    After W01' = W01 + A01 @ B01_bd
        where A01 = [A0, A1] B01_bd = [[B0, 0],
                  [0, B1]]

    This class is responsible for merging LoRA B in this block-diagonal fashion. Assuming that we fuse N weights, it
    should look like this:

    1. LoRA B is 2-dim
    Normal LoRA weight of shape (out_feat, rank), the output shape should be (N * out_feat, N * rank).

    2. LoRA B is 3-dim
    MoE LoRA weight of shape (experts, out_feat, rank), the output shape should be (experts, N * out_feat, N * rank).

    After this, the experts x rank dimension are flattened, as PEFT expects 2d tensors for LoRA.
    """

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        full_layer_name: str,
        **kwargs,
    ) -> dict[str, list[torch.Tensor]]:
        dims = [v.dim() for v in input_dict.values()]
        if set(dims) not in ({2}, {3}):
            raise ValueError(
                f"To convert this LoRA adapter, the LoRA weights all need to have either 2 or 3 dims, got {set(dims)}"
            )

        if set(dims) == {2}:
            output_dict = {full_layer_name: torch.block_diag(*input_dict.values())}
        else:
            out = _block_diag_3d(*input_dict.values())  # shape = experts, 2*out_feat, 2*r
            out = torch.permute(out, (2, 0, 1))  # shape = 2*r, experts, 2*out_feat
            out = out.flatten(0, 1)  # shape = 2*r * experts, 2*out_feat
            out = out.T
            output_dict = {full_layer_name: out}
        return output_dict

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("Reversing PEFT LoRA MoE conversions is not supported yet.")


class FlattenDims(ConversionOps):
    """
    Flatten the tensors along the given dimensions
    """

    def __init__(self, dims: int | tuple[int, ...]):
        if isinstance(dims, int):
            dims = (dims,)
        self.dims = dims

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        config,
        **kwargs,
    ) -> dict[str, list[torch.Tensor]]:
        output_dict = {k: v.flatten(*self.dims) for k, v in input_dict.items()}
        return output_dict

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("Reversing flatteing operatio is not supported.")

    def __repr__(self):
        return f"{self.__class__.__name__}(dims={self.dims})"


class PermuteDims(ConversionOps):
    """
    Permute the tensors along the given dimensions
    """

    def __init__(self, dims: tuple[int, ...]):
        self.dims = dims

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        config,
        **kwargs,
    ) -> dict[str, list[torch.Tensor]]:
        output_dict = {k: v.permute(*self.dims) for k, v in input_dict.items()}
        return output_dict

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("Reversing flatteing operatio is not supported yet.")

    def __repr__(self):
        return f"{self.__class__.__name__}(dims={self.dims})"


def build_peft_weight_mapping_for_transformers(
    weight_conversions: list[WeightConverter | WeightRenaming] | None, adapter_name: str, peft_config=None
) -> list[WeightConverter | WeightRenaming]:
    # We iterate over all the operations of the original model and simply edit them to apply to the PEFT adapter when
    # appropriate.
    if not weight_conversions:
        return []

    # strip "base_model.model" and add adapter name
    new_weight_conversions = [WeightRenaming("base_model.model.model.", "model.")]

    prefixes = set()
    from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING

    peft_type = getattr(peft_config, "peft_type", None)
    if peft_type in PEFT_TYPE_TO_PREFIX_MAPPING:
        prefixes.add(PEFT_TYPE_TO_PREFIX_MAPPING[peft_type])
    else:
        prefixes.update(PEFT_TYPE_TO_PREFIX_MAPPING.values())

    for prefix in sorted(prefixes):
        escaped_prefix = re.escape(prefix)
        new_weight_conversions.append(
            WeightRenaming(
                source_patterns=rf"({escaped_prefix}[^\.]*)",
                target_patterns=rf"\1.{adapter_name}",
            )
        )

    for orig_conversion in weight_conversions:
        if isinstance(orig_conversion, WeightRenaming):
            new_weight_conversions.append(orig_conversion)
            continue

        if orig_conversion.target_patterns == ["mlp.experts.gate_up_proj"]:
            # gate_up_proj requires both merging the experts and concatenating for the fusion of w1 and w3
            for lora in ("lora_A", "lora_B"):  # TODO: lora_embedding_A and lora_embedding_B
                # deal with operations
                peft_weight_operations = []
                for op in orig_conversion.operations:
                    if isinstance(op, Concatenate):
                        if lora == "lora_B":  # block diagonal concat
                            peft_weight_operations.append(PeftConcatenate(dim=op.dim))
                        else:  # normal concat + flatten
                            peft_weight_operations.append(op)
                            peft_weight_operations.append(FlattenDims(dims=(0, 1)))
                    elif isinstance(op, MergeModulelist):
                        peft_weight_operations.append(op)

                # TODO: this assumption may not hold for models != mixtral
                # For source, we capture the orignal weights + the lora weights
                new_source_patterns = []
                for pat in list(orig_conversion.source_patterns):
                    # we replace the weight pattern to colllect loras
                    pat = pat.rsplit(".", 1)[0]
                    # note: the source state_dict does *not* contain the adapter name
                    new_source_patterns.append(f"{pat}.{lora}.*")

                # the gate_up_proj is the innner PEFT ParamWrapper, so we need to use base_layer
                pat = orig_conversion.target_patterns[0]
                pat = pat.replace("gate_up_proj", "base_layer")
                # we make sure the target key is correct, add '.weight' because the parameter is targeted directly
                new_target_patterns = [f"{pat}.{lora}.{adapter_name}.weight"]

                # Instantiate a new object that correctly post process patterns if needed
                new_conversion = orig_conversion.__class__(
                    source_patterns=new_source_patterns,
                    target_patterns=new_target_patterns,
                    distributed_operation=orig_conversion.distributed_operation,
                    quantization_operation=orig_conversion.quantization_operation,
                    operations=new_weight_conversions,
                )
                new_weight_conversions.append(new_conversion)

        elif orig_conversion.target_patterns == ["mlp.experts.down_proj"]:
            # down_proj only requires merging of experts
            for lora in ("lora_A", "lora_B"):  # TODO: lora_embedding_A and lora_embedding_B
                peft_weight_operations = []
                for op in orig_conversion.operations:
                    if isinstance(op, MergeModulelist):
                        peft_weight_operations.append(op)
                        if lora == "lora_A":
                            peft_weight_operations.append(FlattenDims(dims=(0, 1)))
                        else:
                            peft_weight_operations.append(PermuteDims(dims=(2, 0, 1)))
                            peft_weight_operations.append(FlattenDims(dims=(0, 1)))
                            peft_weight_operations.append(Transpose(dim0=0, dim1=1))

                # TODO: this assumption may not hold for models != mixtral
                # For source, we capture the orignal weights + the lora weights
                new_source_patterns = []
                for pat in list(orig_conversion.source_patterns):
                    # we replace the weight pattern to colllect loras
                    pat = pat.rsplit(".", 1)[0]
                    # note: the source state_dict does *not* contain the adapter name
                    new_source_patterns.append(f"{pat}.{lora}.*")

                # the down_proj is the outer PEFT ParamWrapper, so we remove the prefix
                pat = orig_conversion.target_patterns[0]
                pat = pat.replace(".down_proj", "")
                # we make sure the target key is correct, add '.weight' because the parameter is targeted directly
                new_target_patterns = [f"{pat}.{lora}.{adapter_name}.weight"]

                # Instantiate a new object that correctly post process patterns if needed
                new_conversion = orig_conversion.__class__(
                    source_patterns=new_source_patterns,
                    target_patterns=new_target_patterns,
                    distributed_operation=orig_conversion.distributed_operation,
                    quantization_operation=orig_conversion.quantization_operation,
                    operations=new_weight_conversions,
                )
                new_weight_conversions.append(new_conversion)

    return new_weight_conversions
