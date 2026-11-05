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

"""Structural merging by concatenating neurons, including native export widths."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from transformers import PretrainedConfig

from peft.tuners.tuners_utils import _is_key_under_prefixes
from peft.utils.other import ModulesToSaveWrapper

from .runtime import check_no_offload
from .structure import (
    MlpStructure,
    describe_mlp,
    explicit_spec,
    is_ffn_candidate,
    is_moe,
    is_registered_layout,
    read_components,
    resolve_moe_spec,
    text_backbone_prefixes,
    unwrap,
    width_binding,
)


if TYPE_CHECKING:
    from .config import CortexFfnSpec, CortexMoeSpec
    from .layer import CortexLayer


@dataclass
class _ExportCandidate:
    """An FFN and its possible constructor-width bindings, before export validation."""

    path: str
    module: nn.Module
    config: PretrainedConfig | None
    fields: tuple[str, ...]
    multiplier: int


@dataclass
class _ExportMember:
    """A native FFN whose width must track its export group's config field."""

    module: nn.Module
    multiplier: int
    ffn_spec: CortexFfnSpec


@dataclass
class _ExportGroup:
    """The shared constructor width and all FFNs that must be padded to match it."""

    config: PretrainedConfig
    field: str
    width: int = 0
    members: list[_ExportMember] = field(default_factory=list)


def unmerge_state(module: nn.Module, spec: MlpStructure) -> dict[str, Any]:
    """Retain only the metadata changed by _apply_components, with its original values."""
    attributes = [
        (
            module,
            {name: getattr(module, name) for name in spec.ffn_spec.width_attributes if hasattr(module, name)},
        )
    ]
    if not spec.packed:
        kinds = ("gate_up", "down") if spec.fused else (("gate", "up", "down") if spec.has_gate else ("up", "down"))
        for kind in kinds:
            projection = getattr(module, getattr(spec.ffn_spec, kind + "_proj"))
            attributes.append(
                (projection, {name: getattr(projection, name) for name in ("in_features", "out_features")})
            )
    return {"width": spec.intermediate_dim, "attributes": attributes}


def original_components(
    components: dict[str, torch.Tensor], width: int, down_bias: torch.Tensor | None
) -> dict[str, torch.Tensor]:
    values = dict(components)
    for key, value in values.items():
        if key == "down_bias":
            values[key] = down_bias.clone()
        else:
            dim = -1 if key == "down" or key.endswith("_bias") else -2
            # Clone even contiguous slices to release the expanded storage.
            values[key] = value.narrow(dim, 0, width).clone()
    return values


@torch.no_grad()
def _apply_components(
    module: nn.Module,
    components: dict[str, torch.Tensor],
    *,
    spec: MlpStructure,
    attributes: Iterable[tuple[nn.Module, str, Any]] = (),
) -> None:
    """Prepare and commit one FFN and its explicit metadata assignments.

    Retain old values only until this FFN is committed. Rollback restores the fields written here, without replacing
    module dictionaries or invoking setters and registration hooks again. Other hook side effects are not restored.
    """
    values = dict(components)
    if spec.fused:
        for suffix, dim in (("", -2), ("_bias", -1)):
            if "gate" + suffix not in values:
                continue
            gate, up = values.pop("gate" + suffix), values.pop("up" + suffix)
            if spec.interleaved:
                value = torch.stack((gate, up), dim=dim).flatten(dim - 1, dim)
            else:
                value = torch.cat((gate, up), dim=dim)
            values["gate_up" + suffix] = value
    assignments = []
    for key, value in values.items():
        kind = key.removesuffix("_bias")
        bias = key.endswith("_bias")
        if spec.transposed and not bias:
            value = value.transpose(-1, -2)
        projection_name = getattr(spec.ffn_spec, kind + "_proj")
        if spec.packed:
            attr = projection_name + ("_bias" if bias else "")
            destination = module
        else:
            destination = getattr(module, projection_name)
            attr = "bias" if bias else "weight"
            if not bias:
                assignments.extend(
                    [(destination, "out_features", value.shape[0]), (destination, "in_features", value.shape[1])]
                )
        original = getattr(destination, attr)
        assignments.append((destination, attr, nn.Parameter(value.to(original).contiguous(), requires_grad=False)))
    width = components["down"].shape[-1]
    for attr in spec.ffn_spec.width_attributes:
        if hasattr(module, attr):
            assignments.append((module, attr, width))
    assignments.extend(attributes)
    missing = object()
    previous = []
    for child, attr, _ in assignments:
        if attr in child._parameters:
            storage = child._parameters
        elif attr in child._buffers:
            storage = child._buffers
        else:
            storage = vars(child)
        previous.append((storage, attr, storage.get(attr, missing)))
    # Finish all allocations before writing parameters, dimensions or adapter state.
    try:
        for child, attr, value in assignments:
            setattr(child, attr, value)
    except BaseException:
        for storage, attr, value in reversed(previous):
            if value is missing:
                storage.pop(attr, None)
            else:
                storage[attr] = value
        raise


def branch_components(layer: CortexLayer, name: str, prefix: str, public: bool) -> dict[str, torch.Tensor]:
    values = {}
    for kind in ("gate", "up", "down") if layer.structure.has_gate else ("up", "down"):
        projection = getattr(layer, f"cortex_{prefix}{kind}")[name]
        for suffix, tensor in (("", projection.weight), ("_bias", projection.bias)):
            if tensor is None:
                continue
            if public:
                tensor = tensor.unsqueeze(0).expand(layer.structure.num_experts, *tensor.shape)
                if kind == "down":
                    scale = 1.0 if layer.public_weighting[name] == "routing" else layer.public_scale
                    tensor = tensor / scale
            values[kind + suffix] = tensor
    return values


@torch.no_grad()
def _merge_components(
    layer: CortexLayer, names: list[str], safe_merge: bool, concatenate: bool
) -> tuple[MlpStructure, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    check_no_offload(layer)
    if (
        layer.role in {"public", "routed_public"}
        and layer.public_scale is None
        and any(layer.public_weighting[name] == "unweighted" for name in names)
    ):
        raise ValueError(
            "Cannot merge an unweighted public branch: routing weights must have a proven constant nonzero sum. "
            "Unnormalized top-k routing does not meet this condition; train with public_weighting='routing', "
            "use routed mode, or save the adapter."
        )
    if any(hasattr(param, "ds_id") or hasattr(param, "ds_shape") for param in layer.parameters()):
        raise ValueError("Structural merging changes parameter shapes. Reload an unpartitioned model before merging.")
    if any(param.is_meta for param in layer.parameters()):
        raise ValueError("Structural merging requires materialized parameters; load the weights before merging.")

    def check_finite(value):
        if safe_merge and not torch.isfinite(value).all():
            raise ValueError("Non-finite values detected during safe Cortex merge.")

    spec = describe_mlp(layer.base_layer, spec=layer.structure.ffn_spec)
    original = read_components(layer.base_layer, spec=spec)
    values = dict(original)
    for value in values.values():
        check_finite(value)
    for name in names:
        branches = [("", layer.role == "public")]
        if layer.role == "routed_public":
            branches.append(("shared_", True))
        for prefix, public in branches:
            branch = branch_components(layer, name, prefix, public)
            if any(key.endswith("_bias") and key not in values for key in branch):
                raise ValueError("Cannot export adapter biases into a bias-free native FFN; use adapter_bias=False.")
            for kind in ("gate", "up", "down") if layer.structure.has_gate else ("up", "down"):
                tensor = branch[kind].to(values[kind])
                check_finite(tensor)
                if concatenate:
                    values[kind] = torch.cat((values[kind], tensor), dim=-1 if kind == "down" else -2)
                bias_key = kind + "_bias"
                if bias_key not in values:
                    continue
                reference = values[bias_key]
                bias = branch.get(bias_key)
                if bias is None:
                    bias = reference.new_zeros(tensor.shape[:-1])
                bias = bias.to(reference)
                check_finite(bias)
                if kind == "down":
                    values[bias_key] = reference + bias
                    check_finite(values[bias_key])
                elif concatenate:
                    values[bias_key] = torch.cat((reference, bias), dim=-1)
    return spec, original, values


def validate_merge(layer: CortexLayer, names: list[str], safe_merge: bool = False) -> None:
    """Check casts, bias sums and base weights without constructing a wide FFN."""
    _merge_components(layer, names, safe_merge, concatenate=False)


def merged_components(
    layer: CortexLayer, names: list[str], safe_merge: bool = False
) -> tuple[MlpStructure, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return the current structure, original components and widened components."""
    return _merge_components(layer, names, safe_merge, concatenate=True)


@torch.no_grad()
def pad_width(module: nn.Module, width: int, *, ffn_spec: CortexFfnSpec | None = None) -> None:
    spec = describe_mlp(module, spec=ffn_spec)
    added = width - spec.intermediate_dim
    if added <= 0:
        return
    values = read_components(module, spec=spec)
    for key, value in values.items():
        if key == "down_bias":
            continue
        dim = -1 if key == "down" or key.endswith("_bias") else -2
        shape = list(value.shape)
        shape[dim] = added
        values[key] = torch.cat((value, value.new_zeros(shape)), dim=dim)
    _apply_components(module, values, spec=spec)


def export_plan(
    model: nn.Module,
    adapter_names: list[str] | None,
    *,
    ffn_specs: dict[str, CortexFfnSpec] | None = None,
    moe_specs: dict[str, CortexMoeSpec] | None = None,
) -> list[_ExportGroup]:
    """Preflight all config-coupled FFNs, including unadapted layers.

    Shared and routed FFNs sometimes use the same config field. Round up that field and zero-pad other members so
    native from_pretrained can reconstruct every tensor. The padding contributes zero to the output.
    """
    from .layer import CortexLayer

    def configs(config):
        if isinstance(config, PretrainedConfig):
            yield config
            for value in vars(config).values():
                if isinstance(value, PretrainedConfig):
                    yield from configs(value)

    root_config = getattr(model, "config", None)
    serialized = {id(config): config for config in configs(root_config)}

    def walk(module, path, owner, shared=False):
        if isinstance(module, ModulesToSaveWrapper):
            # Plan padding for the same copy that unloading will retain.
            selected = module.active_adapters if adapter_names is None else adapter_names
            adapter = module.check_set_adapter(selected) if selected else None
            module = module.modules_to_save[adapter] if adapter is not None else module.original_module
        local = getattr(unwrap(module), "config", None)
        if isinstance(local, PretrainedConfig):
            if id(local) in serialized:
                owner = local
            elif type(local) is not type(owner):
                candidates = [config for config in configs(owner) if type(config) is type(local)]
                if len(candidates) != 1:
                    raise ValueError(f"Cannot identify the serialized configuration owning {path}.")
                owner = candidates[0]
            # A modules_to_save deepcopy carries a private config. The native
            # constructor still obtains its dimensions from the serialized owner.
        yield path, module, owner, shared
        if isinstance(module, CortexLayer):
            return
        shared_modules = ("shared_expert", "shared_experts")
        if is_moe(module, moe_specs) and (
            is_registered_layout(module) or explicit_spec(module, moe_specs) is not None
        ):
            shared_modules = resolve_moe_spec(module, moe_specs).shared_modules or ()
        for name, child in module.named_children():
            yield from walk(child, f"{path}.{name}" if path else name, owner, name in shared_modules)

    candidates = []
    changed_fields = set()
    changed_configs = set()
    text_roots = text_backbone_prefixes(model)
    for path, module, owner, shared in walk(model, "", root_config):
        base = unwrap(module)
        if not is_ffn_candidate(base, ffn_specs):
            continue
        if (
            text_roots is not None
            and not is_registered_layout(base)
            and explicit_spec(base, ffn_specs) is None
            and not _is_key_under_prefixes(path, text_roots)
        ):
            continue
        config, fields, multiplier = width_binding(base, path, owner, ffn_specs, shared=shared)
        candidates.append(_ExportCandidate(path, module, config, fields, multiplier))
        if isinstance(module, CortexLayer) and (module.merged or module.merge_names(adapter_names)):
            changed_configs.add(id(config))
            changed_fields.update((id(config), field) for field in fields)

    groups: dict[tuple[int, str], _ExportGroup] = {}
    for candidate in candidates:
        path, module, config = candidate.path, candidate.module, candidate.config
        fields, multiplier = candidate.fields, candidate.multiplier
        if id(config) not in changed_configs:
            continue
        if not fields:
            # Even an untargeted sibling can depend on a width that is about to change.
            raise ValueError(
                f"Cannot determine export width for {path}: no conventional intermediate-width field on the "
                "owning config. Save the adapter checkpoint or add a dedicated export rule."
            )
        if not any((id(config), field) in changed_fields for field in fields):
            continue
        if len(fields) != 1:
            raise ValueError(
                f"Ambiguous export width for {path}: {', '.join(fields)} could set its intermediate width. "
                "Save the adapter checkpoint or add a dedicated export rule."
            )
        field = fields[0]
        base = unwrap(module)
        spec = describe_mlp(base, ffn_specs)
        if not is_registered_layout(base) or explicit_spec(base, ffn_specs) is not None:
            # A previous merge_adapter widens the tensor but leaves the serialized config unchanged.
            original = (
                module._merge_state["width"]
                if isinstance(module, CortexLayer) and module.merged
                else spec.intermediate_dim
            )
            if getattr(config, field, None) * multiplier != original:
                raise ValueError(
                    f"Cannot determine export width for {path}: expected config.{field} to match its original "
                    f"width {original}. Save the adapter checkpoint or add a dedicated export rule."
                )

        desired = spec.intermediate_dim
        if isinstance(module, CortexLayer):
            for name in module.merge_names(adapter_names):
                desired += module.cortex_up[name].weight.shape[-2]
                if module.role == "routed_public":
                    desired += module.cortex_shared_up[name].weight.shape[-2]

        # All FFNs built from this config field must fit when the native model is reloaded.
        group = groups.setdefault((id(config), field), _ExportGroup(config=config, field=field))
        group.width = max(group.width, math.ceil(desired / multiplier))
        group.members.append(_ExportMember(module=base, multiplier=multiplier, ffn_spec=spec.ffn_spec))
    return list(groups.values())


def apply_export_plan(plan: list[_ExportGroup]) -> None:
    for group in plan:
        for member in group.members:
            module = member.module
            pad_width(module, member.multiplier * group.width, ffn_spec=member.ffn_spec)
            local = getattr(module, "config", None)
            if type(local) is type(group.config):
                setattr(local, group.field, group.width)
        setattr(group.config, group.field, group.width)
