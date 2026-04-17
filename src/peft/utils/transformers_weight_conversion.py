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

# NOTE: don't import from this module unless transformers v5+ is used
import copy
import re
from typing import Any

import torch
from transformers.conversion_mapping import (
    _MODEL_TO_CONVERSION_PATTERN,
    get_checkpoint_conversion_mapping,
    get_model_conversion_mapping,
)
from transformers.core_model_loading import (
    Concatenate,
    ConversionOps,
    MergeModulelist,
    Transpose,
    WeightConverter,
    WeightRenaming,
    dot_natural_key,
    rename_source_key,
)

from peft import PeftType


# https://github.com/huggingface/transformers/pull/45340#issuecomment-4222734042
_MODEL_TO_CONVERSION_PATTERN = _MODEL_TO_CONVERSION_PATTERN.copy()
_MODEL_TO_CONVERSION_PATTERN["mixtral"] = "mixtral"


def _block_diag_3d(tensors: list[torch.Tensor]) -> torch.Tensor:
    if len(tensors) < 2:
        raise ValueError(f"_block_diag_3d expects at least 2 tensors, got {len(tensors)}")

    if any(t.dim() != 3 for t in tensors):
        raise ValueError("_block_diag_3d expects all tensors to be 3d.")

    num_experts = tensors[0].shape[0]
    if any(t.shape[0] != num_experts for t in tensors):
        raise ValueError("All tensors passed to _block_diag_3d must have the same number of experts.")

    lora_b_block_diag = []
    for i in range(num_experts):
        lora_b_block_diag.append(torch.block_diag(*[tensor[i] for tensor in tensors]))
    return torch.stack(lora_b_block_diag, dim=0)


class PeftConcatenate(Concatenate):
    """Convert per-expert LoRA weights to merged weights.

    When the base weights are fused, e.g. W01 = [W0, W1], the LoRA weights also need to be fused. To achieve this
    correctly, concatenate the LoRA A weights along the r (rank) dimension. This doesn't require a new Operation. But
    for LoRA B, the weights need to be merged in a block diagonal fashion to achieve the correct result.

    To illustrate:

    Before:

    W0' = W0 + A0 @ B0

    W1' = W1 + A1 @ B1

    After:

    W01' = W01 + A01 @ B01_bd

    where:

    A01 = [A0, A1]

    B01_bd = [[B0, 0], [0, B1]]

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

        # Keep source order stable (e.g. w1 before w3 for Mixtral) to preserve gate/up semantics.
        ordered_tensors = [
            input_dict[source_pattern] for source_pattern in source_patterns if source_pattern in input_dict
        ]
        if len(ordered_tensors) != len(input_dict):
            missing = set(input_dict) - set(source_patterns)
            raise ValueError(
                "Collected tensors contain keys not present in source_patterns. "
                f"Unexpected keys: {sorted(missing)}; source_patterns={source_patterns}"
            )

        if set(dims) == {2}:
            output_dict = {full_layer_name: torch.block_diag(*ordered_tensors)}
        else:
            # with r being the LoRA rank and n being the number of fused weights:
            out = _block_diag_3d(ordered_tensors)  # shape = experts, n*out_feat, 2*r
            out = torch.permute(out, (2, 0, 1))  # shape = 2*r, experts, n*out_feat
            out = out.flatten(0, 1)  # shape = 2*r * experts, n*out_feat
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


def build_peft_weight_mapping(
    weight_conversions: list[WeightConverter | WeightRenaming] | None, adapter_name: str, peft_config=None
) -> list[WeightConverter | WeightRenaming]:
    # We iterate over all the operations of the original model and simply edit them to apply to the PEFT adapter when
    # appropriate.
    # Note: This function is used in PEFT, changing it requires coordination.
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

        if len(orig_conversion.target_patterns) == 1 and orig_conversion.target_patterns[0].endswith("gate_up_proj"):
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

                if not peft_weight_operations:
                    continue

                # TODO: this assumption may not hold for models != mixtral
                # For source, we capture the original weights + the lora weights
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
                    operations=peft_weight_operations,
                )
                new_weight_conversions.append(new_conversion)

        elif len(orig_conversion.target_patterns) == 1 and orig_conversion.target_patterns[0].endswith("down_proj"):
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

                if not peft_weight_operations:
                    continue

                # TODO: this assumption may not hold for models != mixtral
                # For source, we capture the original weights + the lora weights
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
                    operations=peft_weight_operations,
                )
                new_weight_conversions.append(new_conversion)

    return new_weight_conversions


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


def _convert_peft_config_moe(peft_config, model_type: str) -> None:
    """
    In-place convert the PEFT config of MoE models whose architecture changed from transformers v4 to v5.

    Since the model architecture changed, the targets have to updated accordingly. Moreover, when weights are fused, it
    requires updating the rank and alpha values of those parameters.
    """
    base_model_type = _MODEL_TO_CONVERSION_PATTERN.get(model_type, None)
    if base_model_type is None:
        return

    target_module_mapping = _MOE_TARGET_MODULE_MAPPING.get(base_model_type)
    if not target_module_mapping:
        return

    fused_targets = _MOE_FUSED_TARGETS.get(base_model_type, {})
    if not fused_targets:
        return

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
            missing = ", ".join(sorted(set(required_old_targets) - present_targets))
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


def convert_peft_config_for_transformers(peft_config, model: torch.nn.Module, conversions: list[Any] | None) -> None:
    """
    Convert the PEFT config of models whose architecture changed from transformers v4 to v5.

    For most models, this requires no changes, this mostly affects some MoE models like Mixtral.

    The conversion should be in-place to ensure that all references to this config stay up-to-date.
    """
    # If, for any reason, we cannot apply conversion, we just return the PEFT config as is.
    if peft_config.peft_type != PeftType.LORA:
        # weight conversion is currently only supported for LoRA
        return
    if not hasattr(model, "config"):
        # not a transformer model
        return
    if not hasattr(model.config, "model_type"):
        # not a transformer model
        return

    model_type = getattr(model.config, "model_type", None)
    if get_checkpoint_conversion_mapping(model_type) is not None:
        _convert_peft_config_moe(peft_config, model_type)


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
    if peft_config.peft_type != PeftType.LORA:
        # weight conversion is currently only supported for LoRA
        return adapter_state_dict

    if not hasattr(model, "config") or not hasattr(model.config, "model_type"):
        # not a transformers-like model
        return adapter_state_dict

    model_type = getattr(model.config, "model_type", None)
    if get_checkpoint_conversion_mapping(model_type) is None:
        # no architecture-level conversion registered for this model
        return adapter_state_dict

    weight_conversions = get_model_conversion_mapping(model)
    peft_weight_conversions = build_peft_weight_mapping(
        weight_conversions, adapter_name=adapter_name, peft_config=peft_config
    )
    if not peft_weight_conversions:
        return adapter_state_dict

    # Keep non-LoRA entries untouched (e.g. modules_to_save / auxiliary wrappers).
    # This avoids interfering with PEFT's later wrapper-specific key remapping.
    lora_like_state_dict = {}
    passthrough_state_dict = {}
    for key, value in adapter_state_dict.items():
        if ".lora_" in key:
            lora_like_state_dict[key] = value
        else:
            passthrough_state_dict[key] = value

    if not lora_like_state_dict:
        return adapter_state_dict

    renamings = [entry for entry in peft_weight_conversions if isinstance(entry, WeightRenaming)]
    converters = [entry for entry in peft_weight_conversions if isinstance(entry, WeightConverter)]
    pattern_to_converter = {k: converter for converter in converters for k in converter.source_patterns}

    param_name_to_load: dict[str, WeightRenaming | WeightConverter] = {}

    # Mirror transformers.core_model_loading flow: stable ordering + same rename logic.
    # https://github.com/huggingface/transformers/blob/1bd97f246318456c1b87cf8ef8dc043ec1a53fff/src/transformers/core_model_loading.py#L997
    state_items = sorted(lora_like_state_dict.items(), key=lambda kv: dot_natural_key(kv[0]))
    for original_key, tensor in state_items:
        renamed_key, source_pattern = rename_source_key(original_key, renamings, converters)

        if source_pattern is not None:
            new_converter = copy.deepcopy(pattern_to_converter[source_pattern])
            mapping = param_name_to_load.setdefault(renamed_key, new_converter)
        else:
            mapping = param_name_to_load.setdefault(renamed_key, WeightRenaming(original_key, renamed_key))
            source_pattern = original_key

        mapping.add_tensor(renamed_key, original_key, source_pattern, tensor)

    converted_lora_model_keys: dict[str, torch.Tensor] = {}
    consumed_source_keys: set[str] = set()

    for first_param_name, mapping in param_name_to_load.items():
        realized_values = mapping.convert(
            first_param_name,
            model=model,
            config=model.config,
            hf_quantizer=None,
            loading_info=None,
        )

        for source_keys in mapping.layer_targets.values():
            consumed_source_keys.update(source_keys)

        for target_name, param in realized_values.items():
            converted_lora_model_keys[target_name] = param[0] if isinstance(param, list) else param

    # Keep untouched LoRA keys, then overwrite with converted results.
    for key in consumed_source_keys:
        lora_like_state_dict.pop(key, None)
    lora_like_state_dict.update(converted_lora_model_keys)

    # Return PEFT-serialized keys (remove adapter-name insertion and keep base_model prefixing behavior).
    converted_lora_serialized = _convert_to_peft_serialized_keys(
        lora_like_state_dict, adapter_name=adapter_name, base_prefix="base_model.model."
    )

    out = {}
    out.update(passthrough_state_dict)
    out.update(converted_lora_serialized)
    return out
