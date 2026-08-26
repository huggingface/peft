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

"""
Export strategies for `compress_and_unload`.

Each strategy takes the base model with the `MoeToDenseLayer`s still in place and replaces the MoE layers by dense
modules, adjusting the model config such that the resulting model can be saved with `save_pretrained` and loaded again
with `from_pretrained` without PEFT.
"""

import warnings
from collections.abc import Callable
from operator import attrgetter

import torch
import transformers
from torch import nn
from tqdm import tqdm

from peft.utils.other import _get_submodules

from .arch import MoeArchSpec, infer_layer_index, set_config_value
from .layer import MoeToDenseLayer


def _first_param(module: nn.Module) -> torch.Tensor | None:
    return next(iter(module.parameters()), None)


def _zero_router(router: nn.Module) -> None:
    """Zero the projection weights and biases of a router so that all experts get the same logit."""
    with torch.no_grad():
        for name, param in router.named_parameters():
            if (param.ndim >= 2) or ("bias" in name):
                param.zero_()


def _router_routes_to_single_expert(router: nn.Module, hidden_size: int) -> bool:
    """Check numerically that the router selects expert 0 with routing weight 1 for every token.

    This is used to heuristically determine if the router outputs are probabilities. If unsure, just return `False`.
    """
    param = _first_param(router)
    if param is None:
        # no param: can't be sure, return True and don't warn
        return True

    x = torch.randn(4, hidden_size, device=param.device, dtype=param.dtype)
    try:
        with torch.no_grad():
            output = router(x)
    except Exception:
        return False

    if not isinstance(output, (tuple, list)):
        return False

    tensors = [o for o in output if torch.is_tensor(o)]
    indices = next((o for o in tensors if not o.is_floating_point()), None)
    if indices is None:
        return False

    if not bool((indices == 0).all()):
        return False

    # The routing weights have the same shape as the indices. With a single expert, so do the logits/probabilities, so
    # accept if any of the candidates is all ones (the zeroed logits cannot be).
    candidates = [o.float() for o in tensors if o.is_floating_point() and o.shape == indices.shape]
    return any(torch.allclose(weights, torch.ones_like(weights)) for weights in candidates)


def _build_single_expert_router(
    old_router: nn.Module, config, hidden_size: int, fallback_param: torch.Tensor | None = None
) -> tuple[nn.Module | None, str | None]:
    """
    Build a router for a single expert from the given (single expert) config that always selects this expert with
    weight 1, i.e. a router of the same type as `old_router` with all projections zeroed.

    `fallback_param` provides the target device and dtype in case the old router's parameters are on the meta device
    (e.g. when parts of the base model were deliberately not materialized).

    Returns the new router (or `None` if it could not be constructed) and a description of the problem, if any.
    """
    try:
        new_router = type(old_router)(config)
    except Exception as exc:
        return None, (
            f"a router of type {type(old_router).__name__} with a single expert cannot be constructed "
            f"({type(exc).__name__}: {exc})"
        )

    old_param = _first_param(old_router)
    if ((old_param is None) or old_param.is_meta) and (fallback_param is not None) and not fallback_param.is_meta:
        old_param = fallback_param
    new_param = _first_param(new_router)
    if (new_param is not None) and (new_param.device.type == "meta"):
        # the router was constructed under `init_empty_weights` (low_cpu_mem_usage), there is nothing to check
        return new_router, None

    if old_param is not None:
        new_router.to(device=old_param.device, dtype=old_param.dtype)
    _zero_router(new_router)
    new_router.requires_grad_(False)
    if hasattr(new_router, "config"):
        new_router.config = config

    if not _router_routes_to_single_expert(new_router, hidden_size):
        return new_router, (
            f"the router of type {type(old_router).__name__} does not assign a routing weight of 1 to the single "
            "dense expert after zeroing its projections (e.g. because it does not use a softmax, or because it uses "
            "group-limited routing)"
        )
    return new_router, None


def one_expert_export_problem(layers: list, adapter_name: str) -> str | None:
    """Return a reason why the 'one expert' export would not produce an equivalent model, or `None` if it would."""
    checked_router_types = set()
    for _, layer in layers:
        if type(layer.router) in checked_router_types:
            continue

        checked_router_types.add(type(layer.router))
        # the config of the dense FFN already describes a single expert
        dense = layer.moe_to_dense_experts[adapter_name]
        _, problem = _build_single_expert_router(
            layer.router, dense.config, layer.layout.hidden_size, fallback_param=dense.down_proj
        )
        if problem is not None:
            return problem

    return None


def export_problem(model: nn.Module, layers: list, adapter_name: str, spec: MoeArchSpec) -> str | None:
    """
    Return a reason why `compress_and_unload` would most likely not produce a working standalone model for this
    architecture, or `None` if no problem is detected. Used to warn the user early, i.e. when the adapter is injected.
    """
    if spec.export == "dense_mlp" and (_dense_mlp_export_problem(model, layers, adapter_name, spec) is None):
        return None
    return one_expert_export_problem(layers, adapter_name)


def _try_forward(module: nn.Module, x: torch.Tensor) -> torch.Tensor | None:
    """Call the module with a single tensor input, return the first tensor of the output or `None` if not possible."""
    try:
        with torch.no_grad():
            output = module(x)
    except Exception:
        return None

    if isinstance(output, (tuple, list)):
        output = next((o for o in output if torch.is_tensor(o)), None)
    return output if torch.is_tensor(output) else None


def _prepare_dense_module(layer, adapter_name: str, config) -> nn.Module:
    """Take the dense FFN out of the tuner layer and make it a regular module of the base model."""
    dense = layer.moe_to_dense_experts[adapter_name]
    # remove the PEFT-specific buffer, it would otherwise end up in the state_dict of the exported model
    delattr(dense, "allocated")
    # share the (patched) model config instead of the copy, so that e.g. `set_experts_implementation` keeps working
    dense.config = config
    dense.requires_grad_(False)
    return dense


def _patch_moe_config(config, spec: MoeArchSpec, intermediate_size: int) -> None:
    set_config_value(config, spec.num_experts_fields, 1)
    set_config_value(config, spec.top_k_fields, 1)
    set_config_value(config, spec.expert_intermediate_fields, intermediate_size)


def export_one_expert(
    model: nn.Module, layers: list[tuple[str, MoeToDenseLayer]], adapter_name: str, progressbar: bool
) -> None:
    """
    Generic export: keep the MoE layer structure, but with a single expert (the dense FFN) and a router that always
    selects it with weight 1. The config is patched accordingly (1 expert, top-1 routing, bigger intermediate size).

    This works for any architecture whose router and experts modules can be constructed from the config alone and whose
    router assigns weight 1 to a single expert when all its projections are zero (e.g. softmax routers).
    """
    intermediate_sizes = {
        layer.num_experts_to_keep[adapter_name] * layer.layout.intermediate_size for _, layer in layers
    }
    if len(intermediate_sizes) != 1:
        raise ValueError(
            "All converted MoE layers must result in the same intermediate size, as they share the same config."
        )

    # build the new routers before modifying the model, so that the model is left untouched if this fails
    new_routers = []
    problems = set()
    for _, layer in layers:
        dense = layer.moe_to_dense_experts[adapter_name]
        new_router, problem = _build_single_expert_router(
            layer.router, dense.config, layer.layout.hidden_size, fallback_param=dense.down_proj
        )
        if new_router is None:
            raise RuntimeError(
                f"Cannot export the dense model because {problem}. This architecture is not supported by "
                "`compress_and_unload()`, consider saving the adapter with `save_pretrained` instead."
            )

        if problem is not None:
            problems.add(problem)
        new_routers.append(new_router)

    for problem in problems:
        warnings.warn(
            f"The exported model will probably not be equivalent to the PEFT model because {problem}. This router "
            "type is not supported by the 'one_expert' export."
        )

    intermediate_size = intermediate_sizes.pop()
    configs_patched = set()
    verified = False
    for (key, layer), new_router in zip(
        tqdm(layers, disable=not progressbar, desc="Exporting dense model"), new_routers
    ):
        parent, _, target_name = _get_submodules(model, key)
        config = layer.base_layer.config
        if id(config) not in configs_patched:
            _patch_moe_config(config, layer.spec, intermediate_size)
            configs_patched.add(id(config))

        if not verified:
            # Compare the output of the whole MoE block before and after the replacement (once, as all layers share
            # the same structure). This catches MoE state that lives outside of the router and the experts.
            dense_param = layer.moe_to_dense_experts[adapter_name].down_proj
            x = torch.randn(1, 4, layer.layout.hidden_size, device=dense_param.device, dtype=dense_param.dtype)
            expected = _try_forward(parent, x)

        dense = _prepare_dense_module(layer, adapter_name, config)
        setattr(parent, target_name, dense)
        if hasattr(new_router, "config"):
            # share the patched model config instead of the copy
            new_router.config = config
        setattr(parent, layer.router_name, new_router)

        if not verified and (expected is not None):
            verified = True
            actual = _try_forward(parent, x)
            if (actual is None) or not torch.allclose(actual, expected, atol=1e-3, rtol=1e-3):
                warnings.warn(
                    f"The MoE block of type {type(parent).__name__} does not give the same output after replacing the "
                    "experts and the router by the dense FFN. The exported model will probably not be equivalent to "
                    "the PEFT model; the block probably holds MoE-specific state besides the router and the experts. "
                    "Consider saving the adapter with `save_pretrained` instead."
                )


def _get_mlp_class_from_transformers(spec: MoeArchSpec) -> type[nn.Module]:
    if spec.dense_mlp_class_name is None:
        raise ValueError("This function should only be called when dense_mlp_class_name was specified.")

    try:
        mlp_cls = attrgetter(spec.dense_mlp_class_name.removeprefix("transformers."))(transformers)
    except AttributeError as exc:
        raise AttributeError(
            f"Specified {spec.dense_mlp_class_name=} class could not be found. This would prevent correct export of "
            "the model. Check your Transformers version if you think it should exist. If you are okay with using a "
            "fallback for the export, set `dense_mlp_class_name` to `None`."
        ) from exc
    return mlp_cls


def _dense_mlp_export_problem(model: nn.Module, layers: list, adapter_name: str, spec: MoeArchSpec) -> str | None:
    """Return a reason why the dense MLP export is not possible, or `None` if it is."""
    if spec.dense_mlp_class_name is None:
        return "No dense MLP class is configured for this architecture."

    intermediate_sizes = set()
    for key, layer in layers:
        layout = layer.layout
        if layout.is_transposed or not layout.is_concatenated or layout.has_bias or not layout.has_gate:
            return "the expert tensor layout is not the standard [gate; up] layout without biases"

        parent_key = key.rpartition(".")[0]
        try:
            parent, _, _ = _get_submodules(model, key)
            _get_submodules(model, parent_key)
        except AttributeError:
            return f"could not resolve the MoE block of '{key}'"

        children = {name for name, _ in parent.named_children()}
        if children != {key.rpartition(".")[2], layer.router_name}:
            return f"the MoE block of '{key}' contains modules besides the router and the experts: {children}"

        if infer_layer_index(key) is None:
            return f"could not infer the decoder layer index from '{key}'"

        intermediate_sizes.add(layer.num_experts_to_keep[adapter_name] * layout.intermediate_size)

    if len(intermediate_sizes) != 1:
        return "the converted MoE layers result in different intermediate sizes"

    intermediate_size = intermediate_sizes.pop()
    _, layer = layers[0]
    mlp_cls = _get_mlp_class_from_transformers(spec)
    for mod in model.modules():
        if isinstance(mod, mlp_cls) and (getattr(mod, "intermediate_size", intermediate_size) != intermediate_size):
            return (
                f"the model already contains dense MLPs with intermediate size {mod.intermediate_size}, which differs "
                f"from the intermediate size of the converted layers ({intermediate_size})"
            )
    return None


def export_dense_mlp(
    model: nn.Module, layers: list[tuple[str, MoeToDenseLayer]], adapter_name: str, progressbar: bool
) -> None:
    """
    Export to the architecture's dense MLP class, e.g. `Qwen3MoeMLP` for Qwen3-MoE: the whole MoE block (router +
    experts) is replaced by a dense MLP and the converted layers are registered as dense layers in the config (e.g.
    `mlp_only_layers`), which results in a model without any MoE overhead at inference time.

    Falls back to the generic "one expert" export if the architecture does not allow this.
    """
    _, first_layer = layers[0]
    spec0 = first_layer.spec
    problem = _dense_mlp_export_problem(model, layers, adapter_name, spec0)
    if problem is not None:
        warnings.warn(f"Cannot export to dense MLPs because {problem}. Falling back to the 'one_expert' export.")
        export_one_expert(model, layers, adapter_name, progressbar)
        return

    # at this point, we must be in the case that dense_layers_field and dense_intermediate_field are defined
    if not spec0.dense_layers_field or not spec0.dense_intermediate_field:
        raise ValueError("`dense_layers_field` and `dense_intermediate_field` must be defined for dense MLP export")

    mlp_cls = _get_mlp_class_from_transformers(spec0)
    config = first_layer.base_layer.config
    intermediate_size = first_layer.num_experts_to_keep[adapter_name] * first_layer.layout.intermediate_size
    dense_layer_indices = set(getattr(config, spec0.dense_layers_field, None) or [])

    for key, layer in tqdm(layers, disable=not progressbar, desc="Exporting dense model"):
        grandparent, _, block_name = _get_submodules(model, key.rpartition(".")[0])
        dense = layer.moe_to_dense_experts[adapter_name]
        gate_up = dense.gate_up_proj.detach()[0]  # [2I, H]
        down = dense.down_proj.detach()[0]  # [H, I]
        mlp = mlp_cls(config, intermediate_size=intermediate_size).to(device=gate_up.device, dtype=gate_up.dtype)

        with torch.no_grad():
            gate, up = gate_up.chunk(2, dim=0)
            mlp.gate_proj.weight.copy_(gate)
            mlp.up_proj.weight.copy_(up)
            mlp.down_proj.weight.copy_(down)

        mlp.requires_grad_(False)
        setattr(grandparent, block_name, mlp)
        dense_layer_indices.add(infer_layer_index(key))

    setattr(config, spec0.dense_layers_field, sorted(dense_layer_indices))
    setattr(config, spec0.dense_intermediate_field, intermediate_size)


EXPORT_STRATEGIES: dict[str, Callable[[nn.Module, list, str, bool], None]] = {
    "one_expert": export_one_expert,
    "dense_mlp": export_dense_mlp,
}
