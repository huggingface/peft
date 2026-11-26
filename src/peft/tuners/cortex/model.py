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

from __future__ import annotations

import math
import warnings
from typing import Any, Optional

import torch
from torch import nn
from transformers import PreTrainedModel

from peft.tuners.tuners_utils import BaseTuner
from peft.utils import TRANSFORMERS_MODELS_TO_CORTEX_TARGET_MODULES_MAPPING

from .config import CortexConfig
from .layer import (
    CortexExperts,
    CortexLayer,
    CortexMLP,
    get_ffn_width,
    is_dense_ffn,
    is_packed_experts,
    pad_ffn,
    unpad_ffn,
)


class CortexModel(BaseTuner):
    """
    Creates a Cortex model from a pretrained transformers model.

    Cortex adds trainable channels to the FFNs of a frozen model. The method is described in [Cortex: Consolidating
    Context into Parametric Memory](https://arxiv.org/abs/2610.20399).

    Unlike most other PEFT methods, merging changes the architecture of the model: `merge_and_unload()` returns the
    base model with each adapted FFN widened by the added channels. It also updates the FFN widths in the config, so
    that the merged model can be saved and loaded with `from_pretrained`. Other FFNs whose width is determined by the
    same config field, e.g. FFNs of layers that were not adapted, are padded with zero channels, which don't change
    their output. If Cortex cannot infer or verify a config for the merged model, it warns and leaves the config
    unchanged instead.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`CortexConfig`]): The configuration of the Cortex model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The Cortex model.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import CortexConfig, get_peft_model

        >>> base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        >>> config = CortexConfig(k=16, init_weights="sample", task_type="CAUSAL_LM")
        >>> model = get_peft_model(base_model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`CortexConfig`]): The configuration of the Cortex model.
    """

    prefix: str = "cortex_"
    tuner_layer_cls = CortexLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_CORTEX_TARGET_MODULES_MAPPING

    def _prepare_adapter_config(self, peft_config: CortexConfig, model_config: dict) -> CortexConfig:
        # The default targets are the dense FFNs of dense models and the shared experts of MoE models, or their packed
        # experts if they have no shared experts. Routed channels are added to the packed experts, which are called
        # `experts` in Transformers MoE models.
        if peft_config.target_modules is None and peft_config.moe_target == "routed":
            peft_config.target_modules = {"experts"}
        elif peft_config.target_modules is None and peft_config.moe_target == "both":
            peft_config = super()._prepare_adapter_config(peft_config, model_config)
            peft_config.target_modules = set(peft_config.target_modules) | {"experts"}
        else:
            peft_config = super()._prepare_adapter_config(peft_config, model_config)
        return peft_config

    def _create_and_replace(
        self,
        cortex_config: CortexConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        **optional_kwargs,
    ) -> None:
        ffn = target.get_base_layer() if isinstance(target, CortexLayer) else target
        if is_packed_experts(ffn):
            branches = self._choose_expert_branches(cortex_config, current_key)
            layer_cls = CortexExperts
        elif is_dense_ffn(ffn):
            self._check_dense_ffn_target(cortex_config, current_key)
            # Dense FFNs have no branches.
            branches = {}
            layer_cls = CortexMLP
        else:
            # The name and the class are more helpful than the representation of a whole FFN or MoE block.
            raise TypeError(
                f"Target module {current_key} ({type(ffn).__name__}) is not supported. Cortex adapts whole FFNs: "
                "either dense FFNs with `gate_proj`, `up_proj` and `down_proj` (`nn.Linear`) and `act_fn`, or packed "
                "MoE experts following the Transformers experts interface. Target e.g. the `mlp` or `experts` "
                "modules, not their projections. FFNs with additional operations, submodules, or parameters, like the "
                "activation sparsity of Gemma3n, a normalization layer, or an output scale, are not supported."
            )

        if isinstance(target, CortexLayer):
            target.update_layer(adapter_name, cortex_config, **branches)
        else:
            new_module = layer_cls(target, adapter_name, cortex_config, **branches)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _choose_expert_branches(self, cortex_config: CortexConfig, experts_key: str) -> dict[str, bool]:
        """Choose the channels to add to packed experts, depending on `moe_target` and their native shared experts."""
        shared_experts = _find_shared_experts(self.model, experts_key)
        # Native shared experts are extended instead of adding channels common to all experts.
        branches = {
            "add_public": cortex_config.moe_target in ("shared", "both") and not shared_experts,
            "add_routed": cortex_config.moe_target in ("routed", "both"),
        }
        if not any(branches.values()):
            raise ValueError(
                f"Cortex adds no channels to {experts_key} with `moe_target={cortex_config.moe_target!r}`, since its "
                "MoE block has native shared experts, which are extended instead. Target the shared experts, or use "
                "`moe_target='routed'` or `moe_target='both'` to add channels to every expert."
            )

        # With explicit target modules, `moe_target="both"` may only be applied in part, which is likely unintended.
        # The warning does not name the experts, so that it is only shown once instead of for every layer.
        are_shared_experts_targeted = any(
            self._check_target_module_exists(cortex_config, key) for key in shared_experts
        )
        if cortex_config.moe_target == "both" and shared_experts and not are_shared_experts_targeted:
            warnings.warn(
                "With `moe_target='both'`, packed experts are targeted without their native shared experts, so only "
                "routed channels are added to them. Target the shared experts as well, or use `moe_target='routed'` "
                "if this is intended."
            )
        return branches

    def _check_dense_ffn_target(self, cortex_config: CortexConfig, ffn_key: str) -> None:
        """Check a dense FFN target against `moe_target` if the FFN is the native shared experts of a MoE block."""
        routed_experts = _find_routed_experts(self.model, ffn_key)
        if routed_experts and cortex_config.moe_target == "routed":
            raise ValueError(
                f"Cortex adds no channels to {ffn_key} with `moe_target='routed'`, since it is the native shared "
                "experts of a MoE block, which are not routed. Target the packed experts, or use "
                "`moe_target='shared'` or `moe_target='both'` to extend the shared experts."
            )

        # As for packed experts, warn if `moe_target="both"` is only applied in part.
        are_routed_experts_targeted = any(
            self._check_target_module_exists(cortex_config, key) for key in routed_experts
        )
        if cortex_config.moe_target == "both" and routed_experts and not are_routed_experts_targeted:
            warnings.warn(
                "With `moe_target='both'`, native shared experts are targeted without their packed experts, so only "
                "the shared experts are extended. Target the packed experts as well, or use `moe_target='shared'` if "
                "this is intended."
            )

    def _replace_module(self, parent: nn.Module, child_name: str, new_module: nn.Module, child: nn.Module) -> None:
        # Cortex layers wrap whole FFNs, which have no single `weight` as expected by `BaseTuner._replace_module`. The
        # adapter weights are already moved to the device of the base FFN when they are created.
        setattr(parent, child_name, new_module)

    def _unload_and_optionally_merge(
        self,
        merge: bool = True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ) -> nn.Module:
        # Adapters that were merged with `merge_adapter` before also remain merged when unloading without merging.
        is_merged = merge or any(isinstance(module, CortexLayer) and module.merged for module in self.model.modules())
        model = super()._unload_and_optionally_merge(
            merge=merge, progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )
        if is_merged:
            _update_config_for_merged_ffns(model)
        return model


def _get_named_children(model: nn.Module, key: str) -> dict[str, nn.Module]:
    """Return the children of the submodule with the given name by their full names.

    Children that are already wrapped by Cortex layers, e.g. by a previous adapter, are replaced by their base FFNs.
    """
    prefix = f"{key}." if key else ""
    return {
        prefix + name: child.get_base_layer() if isinstance(child, CortexLayer) else child
        for name, child in model.get_submodule(key).named_children()
    }


def _find_shared_experts(model: nn.Module, experts_key: str) -> list[str]:
    """Return the names of the native shared experts of the MoE block that holds the given packed experts.

    In Transformers MoE models, the shared experts are a dense FFN next to the packed experts, like the
    `shared_experts` of DeepSeek-V3 or the dense FFN of each layer of Gemma4, or next to the module that holds the
    packed experts, like the `shared_mlp` of GraniteMoeShared. Modules whose name contains `shared` are counted as
    well, so that shared experts that Cortex does not support are not mistaken for missing ones.
    """
    block_key = experts_key.rpartition(".")[0]
    candidates = _get_named_children(model, block_key)
    if block_key:
        candidates |= _get_named_children(model, block_key.rpartition(".")[0])
    return [
        key
        for key, module in candidates.items()
        if key not in (experts_key, block_key) and (is_dense_ffn(module) or "shared" in key.rpartition(".")[-1])
    ]


def _find_routed_experts(model: nn.Module, ffn_key: str) -> list[str]:
    """Return the names of the packed experts whose native shared experts are the given dense FFN, if any.

    This is the inverse of `_find_shared_experts`: the packed experts are next to the FFN, or held by a module next to
    it.
    """
    routed_experts = []
    for key, module in _get_named_children(model, ffn_key.rpartition(".")[0]).items():
        if key == ffn_key:
            continue
        if is_packed_experts(module):
            routed_experts.append(key)
        else:
            routed_experts.extend(
                child_key for child_key, child in _get_named_children(model, key).items() if is_packed_experts(child)
            )
    return routed_experts


def _get_ffn_widths(model: nn.Module) -> dict[str, int]:
    """Return the widths of the FFNs that Cortex supports, by module name."""
    widths = {}
    for name, module in model.named_modules():
        width = get_ffn_width(module)
        if width is not None:
            widths[name] = width
    return widths


def _find_width_fields(config_dict: dict, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """Find the fields of a config dict, including nested configs, that could determine the widths of FFNs.

    Following the naming convention of Transformers, these are the integer fields whose names contain `intermediate`,
    like `intermediate_size` or `moe_intermediate_size`.
    """
    fields = []
    for key, value in config_dict.items():
        if isinstance(value, dict):
            fields.extend(_find_width_fields(value, prefix + (key,)))
        # Some dicts in a config, like `id2label`, have integer keys. Booleans are excluded, although they are ints.
        elif isinstance(key, str) and "intermediate" in key and type(value) is int:
            fields.append(prefix + (key,))
    return fields


def _get_field(config_dict: dict, field: tuple[str, ...]) -> int:
    """Return a field of a config dict. The field may be nested in other dicts."""
    for key in field[:-1]:
        config_dict = config_dict[key]
    return config_dict[field[-1]]


def _set_field(config: Any, field: tuple[str, ...], value: int) -> None:
    """Set a field of a config or a config dict. The field may be nested in configs or plain dicts."""
    for key in field[:-1]:
        config = config[key] if isinstance(config, dict) else getattr(config, key)
    if isinstance(config, dict):
        config[field[-1]] = value
    else:
        setattr(config, field[-1], value)


def _build_meta_model(model: PreTrainedModel, changes: dict[tuple[str, ...], int]) -> Optional[PreTrainedModel]:
    """Instantiate the model on the meta device with the given changes to its config, or return `None` on failure.

    Like `from_pretrained`, the config is created from a dict, so the result matches the model that would be loaded
    from a saved checkpoint.
    """
    config_dict = model.config.to_dict()
    for field, value in changes.items():
        _set_field(config_dict, field, value)

    try:
        config = type(model.config).from_dict(config_dict)
        with torch.device("meta"):
            return type(model)(config)
    except Exception:
        # Not every model can be instantiated from its config alone, and a config may reject some values of a field.
        return None


def _warn_config_not_updated(reason: str) -> None:
    """Warn that the config of the merged model is left unchanged, and why."""
    warnings.warn(
        f"Cortex could not infer or verify a config for the merged model, because {reason}. The config is left "
        "unchanged, so the merged model cannot be loaded with `from_pretrained` after saving it. Instead, save the "
        "adapter before merging it, and load it with `PeftModel.from_pretrained` and merge it each time."
    )


def _update_config_for_merged_ffns(model: nn.Module) -> None:
    """Update the FFN widths in the config of a merged model, so that it can be loaded with `from_pretrained`.

    The widths of the FFNs are determined by config fields like `intermediate_size`. A field can determine the widths
    of several FFNs, e.g. of all layers, or of both the routed and the shared experts of DeepSeek-V4, and some FFNs are
    a multiple of a field wide, e.g. the shared experts of DeepSeek-V3. Instead of inferring this from the names of the
    fields, the model is instantiated on the meta device with each field incremented, which shows which FFNs the field
    determines. Then each field is increased to fit the widest merged FFN that it determines, and its other FFNs are
    padded with zero channels, which don't change their output.
    """
    if not isinstance(model, PreTrainedModel):
        return

    # Compare the widths of the merged FFNs with those of the model instantiated from the unchanged config.
    merged_widths = _get_ffn_widths(model)
    original_model = _build_meta_model(model, {})
    if original_model is None:
        _warn_config_not_updated("the model cannot be instantiated from its config")
        return
    original_widths = _get_ffn_widths(original_model)
    if merged_widths == original_widths:
        # Merging didn't widen any FFN, e.g. because no adapter was merged, so the config is still correct.
        return
    if merged_widths.keys() != original_widths.keys():
        _warn_config_not_updated(
            "the FFNs of the model differ from those of the model instantiated from its config, e.g. because the "
            "model was modified after its creation"
        )
        return

    # Find out by how much incrementing a field increases the width of each FFN.
    config_dict = model.config.to_dict()
    width_increases = {}
    for field in _find_width_fields(config_dict):
        perturbed_model = _build_meta_model(model, {field: _get_field(config_dict, field) + 1})
        if perturbed_model is None:
            continue
        perturbed_widths = _get_ffn_widths(perturbed_model)
        width_increases[field] = {
            name: perturbed_widths[name] - width
            for name, width in original_widths.items()
            if perturbed_widths.get(name, width) > width
        }

    # Increase each field to fit the widest merged FFN that it determines.
    field_increments = {}
    for name, merged_width in merged_widths.items():
        added_width = merged_width - original_widths[name]
        if added_width <= 0:
            continue
        fields = [field for field, increases in width_increases.items() if name in increases]
        if len(fields) != 1:
            _warn_config_not_updated(
                f"the width of `{name}` is not determined by a single integer config field whose name contains "
                "`intermediate`"
            )
            return
        field = fields[0]
        increment = math.ceil(added_width / width_increases[field][name])
        field_increments[field] = max(field_increments.get(field, 0), increment)

    target_widths = dict(original_widths)
    for field, increment in field_increments.items():
        for name, increase in width_increases[field].items():
            target_widths[name] += increase * increment

    changes = {field: _get_field(config_dict, field) + increment for field, increment in field_increments.items()}
    target_model = _build_meta_model(model, changes)
    if target_model is None or _get_ffn_widths(target_model) != target_widths:
        _warn_config_not_updated(
            "the config does not accept the increased fields, or the FFN widths do not increase linearly with them"
        )
        return

    padded_widths = {name: width for name, width in target_widths.items() if width > merged_widths[name]}

    # The changed fields may also determine the shapes of other parameters and buffers, e.g. of FFNs that Cortex does
    # not support, or the input features of FFNs. After padding, all shapes must match those of the target model, as
    # `from_pretrained` requires. To check this without modifying the model, remove the channels that padding would
    # add from the FFNs of the target model instead, which is on the meta device, and compare the shapes.
    for name in padded_widths:
        unpad_ffn(target_model.get_submodule(name), merged_widths[name])
    shapes = {key: value.shape for key, value in model.state_dict().items()}
    target_shapes = {key: value.shape for key, value in target_model.state_dict().items()}
    if shapes != target_shapes:
        _warn_config_not_updated(
            "the padded model would not have the parameter and buffer shapes of the model instantiated from the new "
            "config, e.g. because the increased fields also determine the shapes of other parameters"
        )
        return

    for name, width in padded_widths.items():
        pad_ffn(model.get_submodule(name), width)

    for field, value in changes.items():
        _set_field(model.config, field, value)
