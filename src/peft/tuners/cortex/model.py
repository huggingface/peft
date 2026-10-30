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

from copy import copy

from torch import Tensor, nn

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, _ExcludedModule, _is_key_under_prefixes
from peft.utils.other import AuxiliaryTrainingWrapper, _is_valid_match

from .config import CortexConfig
from .layer import CortexExperts, CortexLayer, CortexMLP
from .merge import apply_export_plan, export_plan, validate_merge
from .runtime import check_moe_forward, check_moe_forward_hook, check_no_offload
from .structure import (
    DENSE,
    EXPERTS,
    MlpStructure,
    describe_mlp,
    is_moe,
    native_name,
    public_merge_scale,
    shared_names,
)


class CortexModel(BaseTuner):
    """Adapt a frozen model with additional nonlinear FFN channels.

    Args:
        model (`torch.nn.Module`):
            Model containing supported dense FFNs or MoE blocks.
        peft_config ([`CortexConfig`] or `dict[str, CortexConfig]`):
            Adapter configuration, or a mapping from adapter names to configurations.
        adapter_name (`str`):
            Name of the adapter to inject.
        low_cpu_mem_usage (`bool`, *optional*, defaults to `False`):
            Create adapter weights on the meta device for subsequent checkpoint loading. Leave this disabled when
            creating a new adapter for training.
        state_dict (`dict[str, torch.Tensor]`, *optional*):
            Adapter checkpoint used by PEFT's loading and injection machinery.

    Example:

    ```py
    >>> from transformers import AutoModelForCausalLM
    >>> from peft import CortexConfig, get_peft_model

    >>> base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    >>> config = CortexConfig(k=16, init_weights="sample", task_type="CAUSAL_LM")
    >>> model = get_peft_model(base_model, config)
    ```
    """

    prefix = "cortex_"
    tuner_layer_cls = CortexLayer

    def inject_adapter(
        self,
        model: nn.Module,
        adapter_name: str,
        autocast_adapter_dtype: bool = True,
        low_cpu_mem_usage: bool = False,
        state_dict: dict[str, Tensor] | None = None,
    ) -> None:
        config = self.peft_config[adapter_name]
        original_targets = copy(config.target_modules)
        unresolved = not config.target_roles
        check_no_offload(model)
        self._resolved_structures = {}
        # Match BaseTuner's checkpoint-key priority and compiled-key normalization.
        self._checkpoint_targets = (
            {key.rsplit("." + self.prefix, 1)[0].removeprefix("_orig_mod.") for key in state_dict}
            if state_dict is not None
            else None
        )
        try:
            if state_dict is None and config.target_roles and hasattr(config, "_cortex_user_targets"):
                # Check the original selection without changing the configuration
                # owned by the previous model. Checkpoints retain explicit roles.
                probe = copy(config)
                probe.target_modules = config._cortex_user_targets
                probe.target_roles = {}
                resolved = self._resolve_targets(probe, model)
                if {path: role for path, (role, _) in resolved.items()} != config.target_roles:
                    raise ValueError(
                        "Reusing this CortexConfig would change the resolved target layers or roles. "
                        "Create a new CortexConfig for the new model instead of reusing a resolved configuration."
                    )
            super().inject_adapter(
                model, adapter_name, autocast_adapter_dtype, low_cpu_mem_usage, state_dict=state_dict
            )
            if state_dict is not None:
                # The checkpoint, rather than the original logical selection, now owns these targets.
                config.__dict__.pop("_cortex_user_targets", None)
            elif unresolved:
                # Runtime-only provenance is absent from dataclass serialization.
                config._cortex_user_targets = original_targets
            for path in config.target_roles:
                parent = model.get_submodule(path.rpartition(".")[0])
                if is_moe(parent) and check_moe_forward_hook not in parent._forward_pre_hooks.values():
                    parent.register_forward_pre_hook(check_moe_forward_hook)
        finally:
            # Structures describe this injection only, not future model mutations.
            del self._resolved_structures, self._checkpoint_targets

    def _prepare_adapter_config(self, peft_config: CortexConfig, model_config: dict[str, object]) -> CortexConfig:
        resolved = self._resolve_targets(peft_config, self.model, checkpoint_targets=self._checkpoint_targets)
        peft_config.target_roles = {path: role for path, (role, _) in resolved.items()}
        peft_config.target_modules = set(resolved)
        self._resolved_structures = {path: structure for path, (_, structure) in resolved.items()}
        return peft_config

    def _resolve_targets(
        self, peft_config: CortexConfig, model: nn.Module, checkpoint_targets: set[str] | None = None
    ) -> dict[str, tuple[str, MlpStructure]]:
        """Resolve and validate physical FFNs without changing the user configuration."""
        if isinstance(model, AuxiliaryTrainingWrapper):
            raise TypeError("Cortex cannot adapt an auxiliary training wrapper as the root model.")
        peft_config = copy(peft_config)
        automatic = checkpoint_targets is None and peft_config.target_modules is None and not peft_config.target_roles
        named_modules = list(model.named_modules())
        auxiliary_roots = {name for name, module in named_modules if isinstance(module, AuxiliaryTrainingWrapper)}
        if peft_config.modules_to_save:
            for name, module in named_modules:
                # Match PEFT's auxiliary wrapping step, before any adapter injection.
                if isinstance(module, AuxiliaryTrainingWrapper) or not any(
                    _is_valid_match(name, target) for target in peft_config.modules_to_save
                ):
                    continue
                if any(isinstance(child, CortexLayer) for child in module.modules()):
                    raise ValueError(
                        f"modules_to_save target {name} contains an existing Cortex layer. "
                        "Choose a module without Cortex layers; wrapping them would change their checkpoint ownership."
                    )

        def available(path):
            if checkpoint_targets is None and isinstance(
                self._check_target_module_exists(peft_config, path), _ExcludedModule
            ):
                return False
            if _is_key_under_prefixes(path, auxiliary_roots):
                if automatic:
                    return False
                raise ValueError(
                    f"Cortex target {path} is owned by an existing auxiliary training wrapper "
                    "(such as ModulesToSaveWrapper). Target an unwrapped FFN or exclude this target; "
                    "adapters inside auxiliary wrappers cannot be saved independently."
                )
            return True

        if checkpoint_targets is None and peft_config.target_modules is None:
            targets = []
            for name, module in named_modules:
                if (
                    not name
                    or _is_key_under_prefixes(name, auxiliary_roots)
                    or any(name.startswith(parent + ".") for parent in targets)
                ):
                    continue
                if isinstance(module, CortexLayer) or is_moe(module) or native_name(module) in DENSE | EXPERTS:
                    targets.append(name)
            if not targets:
                if auxiliary_roots:
                    raise ValueError("No eligible native FFNs remain outside auxiliary training wrappers.")
                raise ValueError("No supported native FFNs found; specify target_modules for an audited structure.")
            peft_config.target_modules = set(targets)
        if checkpoint_targets is None:
            selected = [
                name
                for name, _ in named_modules
                if name and self._check_target_module_exists(peft_config, name) and available(name)
            ]
        else:
            adapter_prefixes = tuple(
                name + "." for name, module in named_modules if isinstance(module, BaseTunerLayer)
            )
            selected = [
                name.removeprefix("_orig_mod.")
                for name, _ in named_modules
                if name
                and not name.startswith(adapter_prefixes)
                and name.removeprefix("_orig_mod.") in checkpoint_targets
                and available(name)
            ]
        if any(name.startswith(parent + ".") for name in selected for parent in selected if name != parent):
            raise ValueError("Select complete MLP blocks without also targeting their descendants.")
        checked_parents = set()
        if checkpoint_targets is not None:
            target_roles = {}
            for path in selected:
                role = peft_config.target_roles.get(path)
                if role is None:
                    parent_path, _, child = path.rpartition(".")
                    parent = model.get_submodule(parent_path)
                    if not describe_mlp(model.get_submodule(path)).packed:
                        role = "shared" if is_moe(parent) and child in shared_names(parent) else "dense"
                    elif not peft_config.target_roles:
                        # An explicit expert leaf means routed; a selected MoE parent uses moe_target.
                        logical_target = None
                        if peft_config.target_modules is not None and self._check_target_module_exists(
                            peft_config, path
                        ):
                            logical_target = path
                        elif is_moe(parent) and (
                            peft_config.target_modules is None
                            or self._check_target_module_exists(peft_config, parent_path)
                        ):
                            logical_target = parent_path
                        if logical_target is not None:
                            role = self._expand_target_roles(
                                peft_config, model, [logical_target], auxiliary_roots, checked_parents
                            ).get(path)
                    if role is None:
                        raise ValueError(
                            f"Cortex checkpoint injection requires target_roles for packed MoE target {path}. "
                            "Load the checkpoint's CortexConfig; this target's routed/public role cannot be "
                            "determined from the current configuration."
                        )
                target_roles[path] = role
        elif peft_config.target_roles:
            if set(selected) != set(peft_config.target_roles):
                raise ValueError("Resolved adapter targets do not match the current model.")
            target_roles = peft_config.target_roles
        else:
            target_roles = self._expand_target_roles(peft_config, model, selected, auxiliary_roots, checked_parents)
        # Parent selections may expand to excluded or separately owned FFNs.
        target_roles = {path: role for path, role in target_roles.items() if available(path)}
        resolved = self._validate_target_roles(model, target_roles, checked_parents)
        if automatic and auxiliary_roots and not resolved:
            raise ValueError("No eligible native FFNs remain outside auxiliary training wrappers and exclusions.")
        return resolved

    @staticmethod
    def _expand_target_roles(
        peft_config: CortexConfig,
        model: nn.Module,
        selected: list[str],
        auxiliary_roots: set[str],
        checked_parents: set[nn.Module],
    ) -> dict[str, str]:
        """Expand logical selections using the native structure, independently of existing adapter roles."""
        target_roles = {}
        targets = peft_config.moe_target
        targets = [targets] if isinstance(targets, str) else targets
        for path in selected:
            module = model.get_submodule(path)
            if is_moe(module):
                if module not in checked_parents:
                    check_moe_forward(module)
                    checked_parents.add(module)
                experts_path = path + ".experts"
                if (
                    not _is_key_under_prefixes(experts_path, auxiliary_roots)
                    and native_name(module.experts) not in EXPERTS
                ):
                    raise ValueError(
                        f"{path} requires a supported packed expert layout; older ModuleList experts are "
                        "unsupported. See the Cortex documentation for the validated Transformers versions."
                    )
                shared = shared_names(module)
                if shared:
                    if "routed" in targets:
                        target_roles[experts_path] = "routed"
                    if "shared" in targets:
                        for child in shared:
                            target_roles[path + "." + child] = "shared"
                else:
                    if len(targets) == 2:
                        role = "routed_public"
                    elif targets[0] == "shared":
                        role = "public"
                    else:
                        role = "routed"
                    target_roles[experts_path] = role
            elif native_name(module) in EXPERTS:
                # Selecting an expert leaf explicitly always selects routed experts.
                target_roles[path] = "routed"
            else:
                parent_path, _, child = path.rpartition(".")
                parent = model.get_submodule(parent_path)
                target_roles[path] = "shared" if is_moe(parent) and child in shared_names(parent) else "dense"
        return target_roles

    @staticmethod
    def _validate_target_roles(
        model: nn.Module, target_roles: dict[str, str], checked_parents: set[nn.Module]
    ) -> dict[str, tuple[str, MlpStructure]]:
        """Validate all physical targets before injection and retain their structure for this operation."""
        resolved = {}
        for path, role in target_roles.items():
            module = model.get_submodule(path)
            parent = model.get_submodule(path.rpartition(".")[0])
            if is_moe(parent) and parent not in checked_parents:
                check_moe_forward(parent)
                checked_parents.add(parent)
            if isinstance(module, BaseTunerLayer) and not isinstance(module, CortexLayer):
                raise TypeError("Mixing tuner types on the same target is not supported.")
            if isinstance(module, CortexLayer) and module.merged:
                raise ValueError(f"Unmerge before adding an adapter. Target {path} is merged.")
            structure = describe_mlp(module)
            if structure.packed != (role in {"routed", "public", "routed_public"}):
                raise ValueError(f"Target {path} does not have the saved adapter role {role}.")
            if isinstance(module, CortexLayer) and module.role != role:
                raise ValueError(
                    f"Target {path} already has a different adapter role: "
                    f"existing={module.role!r}, requested={role!r}."
                )
            if role in {"public", "routed_public"}:
                if not is_moe(parent) or shared_names(parent):
                    raise ValueError("Public branches require an audited MoE without native shared experts.")
            resolved[path] = (role, structure)
        return resolved

    def _create_and_replace(
        self,
        config: CortexConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        **kwargs,
    ) -> None:
        if isinstance(target, CortexLayer):
            target.update_layer(adapter_name, config)
            return
        structure = self._resolved_structures[current_key]
        role = config.target_roles[current_key]
        scale = public_merge_scale(parent) if role in {"public", "routed_public"} else None
        cls = CortexExperts if structure.packed else CortexMLP
        replacement = cls(target, adapter_name, config, structure, role=role, public_scale=scale)
        if adapter_name not in self.active_adapters:
            replacement.requires_grad_(False)
        self._replace_module(parent, target_name, replacement, target)

    def _replace_module(self, parent: nn.Module, child_name: str, new_module: nn.Module, child: nn.Module) -> None:
        # Whole FFNs do not expose the single weight assumed by BaseTuner.
        setattr(parent, child_name, new_module)

    def _preflight_merge(self, adapter_names: list[str] | None, safe_merge: bool) -> None:
        check_no_offload(self.model)
        if isinstance(adapter_names, str):
            raise TypeError("adapter_names must be a list of names.")
        if adapter_names is not None and set(adapter_names) - self.peft_config.keys():
            raise ValueError("Unknown adapter name requested for merging.")
        checked_parents = set()
        for path, module in self.model.named_modules():
            if isinstance(module, CortexLayer):
                parent = self.model.get_submodule(path.rpartition(".")[0])
                if is_moe(parent) and parent not in checked_parents:
                    check_moe_forward(parent)
                    checked_parents.add(parent)
                names = module.merge_names(adapter_names)
                if names:
                    if module.role in {"public", "routed_public"} and any(
                        module.public_weighting[name] == "unweighted" for name in names
                    ):
                        if public_merge_scale(parent) != module.public_scale:
                            raise ValueError(
                                "Routing normalization changed after adapter injection; reload the adapter."
                            )
                    validate_merge(module, names, safe_merge)

    def merge_adapter(self, adapter_names: list[str] | None = None, safe_merge: bool = False) -> None:
        """Merge adapter channels after checking common incompatibilities across the model.

        Execution is not atomic across layers, including with `safe_merge=True`. If execution fails, discard this model
        instance and reload the saved base model and adapter checkpoints.
        """
        self._preflight_merge(adapter_names, safe_merge)
        return super().merge_adapter(adapter_names=adapter_names, safe_merge=safe_merge)

    def unmerge_adapter(self) -> None:
        """Restore original FFN channels layer by layer.

        If execution fails, discard this model instance and reload the saved base model and adapter checkpoints.
        """
        check_no_offload(self.model)
        return super().unmerge_adapter()

    def delete_adapter(self, adapter_name: str) -> None:
        super().delete_adapter(adapter_name)
        for module in self.model.modules():
            if check_moe_forward_hook not in module._forward_pre_hooks.values():
                continue
            # Empty wrappers can remain after deletion; only live adapters need the check.
            if any(
                isinstance(child, CortexLayer) and child._all_available_adapter_names() for child in module.modules()
            ):
                continue
            for key, hook in list(module._forward_pre_hooks.items()):
                if hook is check_moe_forward_hook:
                    del module._forward_pre_hooks[key]

    def _unload_and_optionally_merge(
        self,
        merge: bool = True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: list[str] | None = None,
    ) -> nn.Module:
        plan = None
        if merge:
            self._preflight_merge(adapter_names, safe_merge)
            plan = export_plan(self.model, adapter_names)
        else:
            check_no_offload(self.model)
        result = super()._unload_and_optionally_merge(merge, progressbar, safe_merge, adapter_names)
        if plan is not None:
            apply_export_plan(plan)
        for module in result.modules():
            for key, hook in list(module._forward_pre_hooks.items()):
                if hook is check_moe_forward_hook:
                    del module._forward_pre_hooks[key]
        return result

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: list[str] | None = None
    ) -> nn.Module:
        """Merge selected channels into the base model and remove the adapter wrappers.

        Args:
            progressbar (`bool`, *optional*, defaults to `False`):
                Show progress while unloading adapter wrappers.
            safe_merge (`bool`, *optional*, defaults to `False`):
                Check merged tensors for non-finite values before modifying the model.
            adapter_names (`list[str]`, *optional*):
                Names of the adapters to merge. Defaults to the active adapters. Previously merged adapter channels
                remain in the base weights, even if their adapters are not selected here. To export only the selected
                adapters relative to the current wrapper's original base, call `unmerge_adapter()` first. All adapter
                wrappers are removed on export; save their checkpoints first if they are needed later.

        Returns:
            `torch.nn.Module`: The native model with wider FFNs. For registered Transformers structures, configuration
            widths are updated and other FFNs sharing those widths are zero-padded so the model can be saved and
            reloaded. Custom model constructors must handle their enlarged dimensions explicitly.

        Raises:
            ValueError:
                If an adapter name is unknown, a public branch cannot be merged under the native routing rules, adapter
                biases cannot be represented by the base FFN, or `safe_merge=True` detects non-finite values.
            TypeError:
                If `adapter_names` is a string instead of a list.

        Structural merging increases model parameters and computation. An unweighted public branch requires a proven
        constant nonzero routing mass; a routing-weighted public branch also supports unnormalized top-k weights.
        Public channels are copied into every routed expert. Reload exported MoE models with
        `experts_implementation="eager"` because expanded widths may not satisfy optimized kernel constraints.

        Save adapter checkpoints before merging. Merge a fully materialized, unpartitioned model, outside its training
        engine. The returned model no longer supports adapter switching or unmerging.

        Preflight checks reject common incompatibilities before modifying layers. Execution is not atomic across the
        model, including with `safe_merge=True`. If execution fails, discard this model instance and reload the saved
        base model and adapter checkpoints.
        """
        return self._unload_and_optionally_merge(True, progressbar, safe_merge, adapter_names)
