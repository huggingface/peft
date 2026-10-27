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

import torch
from torch import nn
from torch.nn import functional as F

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .config import CortexConfig
from .initialization import initialize_branch
from .runtime import check_no_offload
from .structure import MlpStructure


class ExpertLinear(nn.Module):
    """One projection per expert, stored as [E, out, in].

    Expert slices belong to one Parameter. Unselected slices receive zero gradients when another slice participates in
    the graph; standard AdamW still applies decay and existing momentum and shares its step counter across slices.
    """

    def __init__(self, experts: int, in_features: int, out_features: int, bias: bool) -> None:
        super().__init__()
        # initialize_branch fills these after device placement and parameter gathering.
        self.weight = nn.Parameter(torch.empty(experts, out_features, in_features))
        self.bias = nn.Parameter(torch.empty(experts, out_features)) if bias else None

    def expert_views(self) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...] | None]:
        # One UnbindBackward assembles the packed gradient. Selecting the packed
        # parameter separately for every expert allocates a full gradient each time.
        return self.weight.unbind(0), None if self.bias is None else self.bias.unbind(0)

    def forward(
        self,
        x: torch.Tensor,
        expert: int,
        views: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...] | None] | None = None,
    ) -> torch.Tensor:
        weights, biases = self.expert_views() if views is None else views
        return F.linear(x, weights[expert], None if biases is None else biases[expert])


class CortexLayer(nn.Module, BaseTunerLayer):
    """Flat adapter storage compatible with PEFT checkpoint name handling."""

    adapter_layer_names = (
        "cortex_gate",
        "cortex_up",
        "cortex_down",
        "cortex_shared_gate",
        "cortex_shared_up",
        "cortex_shared_down",
    )
    other_param_names = ("public_weighting",)

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        config: CortexConfig,
        structure: MlpStructure,
        *,
        role: str = "dense",
        public_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.structure = structure
        self.in_features, self.out_features = structure.input_dim, structure.output_dim
        self.role = role
        self.per_expert = self.role in {"routed", "routed_public"}
        self.public_scale = public_scale
        self.public_weighting = {}
        for name in self.adapter_layer_names:
            setattr(self, name, nn.ModuleDict())
        self._active_adapter = adapter_name
        self._disable_adapters = False
        self.merged_adapters = []
        self._merge_state = None
        self.register_buffer("_original_down_bias", None, persistent=False)
        self.update_layer(adapter_name, config)

    def update_layer(self, adapter_name: str, config: CortexConfig) -> None:
        if adapter_name in self.cortex_up:
            raise ValueError(f"Adapter {adapter_name!r} already exists in this layer.")
        if self.merged:
            raise ValueError("Unmerge before adding an adapter.")
        self.public_weighting[adapter_name] = config.public_weighting
        width = config.k
        if self.per_expert:
            width = config.k_routed or width
        elif self.role in {"shared", "public"}:
            width = config.k_shared or width
        self._add_branch(adapter_name, config, width, "", self.per_expert)
        if self.role == "routed_public":
            self._add_branch(adapter_name, config, config.k_shared or config.k, "shared_", False)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        initialize_branch(self, adapter_name, config, per_expert=self.per_expert)
        if self.role == "routed_public":
            initialize_branch(self, adapter_name, config, prefix="shared_")
        self.set_adapter(self.active_adapters, inference_mode=config.inference_mode)

    def _add_branch(self, name: str, config: CortexConfig, width: int, prefix: str, per_expert: bool) -> None:
        def projection(input_dim, output_dim, bias):
            if per_expert:
                return ExpertLinear(self.structure.num_experts, input_dim, output_dim, bias)
            return nn.Linear(input_dim, output_dim, bias=bias)

        dimensions = {"up": (self.structure.input_dim, width), "down": (width, self.structure.output_dim)}
        if self.structure.has_gate:
            dimensions["gate"] = dimensions["up"]
        for kind, dims in dimensions.items():
            bias = config.adapter_bias
            if bias is None:
                # Inspect parameter presence only, including on meta/ZeRO-sharded base layers.
                source_kind = "gate_up" if self.structure.fused and kind in ("gate", "up") else kind
                source = self.get_base_layer()
                if self.structure.packed:
                    native_bias = getattr(source, source_kind + "_proj_bias", None)
                else:
                    native_bias = getattr(getattr(source, source_kind + "_proj"), "bias", None)
                bias = native_bias is not None
            layer = projection(*dims, bias=bias)
            getattr(self, f"cortex_{prefix}{kind}")[name] = layer

    def _get_base_layer_device_and_dtype(self, base_layer: nn.Module) -> tuple[torch.device, torch.dtype]:
        reference = next(base_layer.parameters())
        return reference.device, reference.dtype

    def _branch(
        self, x: torch.Tensor, name: str, expert: int | None = None, prefix: str = "", views=None
    ) -> torch.Tensor:
        up = getattr(self, f"cortex_{prefix}up")[name]
        down = getattr(self, f"cortex_{prefix}down")[name]
        x = self._cast_input_dtype(x, up.weight.dtype)

        def project(kind, projection, inputs):
            if expert is None:
                return projection(inputs)
            return projection(inputs, expert, None if views is None else views[kind])

        gate_value = None
        if self.structure.has_gate:
            gate_value = project("gate", getattr(self, f"cortex_{prefix}gate")[name], x)
        hidden = self.structure.activate(gate_value, project("up", up, x))
        return project("down", down, hidden)

    def merge_names(self, adapter_names: list[str] | None = None) -> list[str]:
        names = self.active_adapters if adapter_names is None else adapter_names
        if isinstance(names, str):
            raise TypeError("adapter_names must be a list of names.")
        return list(dict.fromkeys(n for n in names if n in self.cortex_up and n not in self.merged_adapters))

    def merge(self, safe_merge: bool = False, adapter_names: list[str] | None = None) -> None:
        from .merge import _apply_components, merged_components, unmerge_state

        names = self.merge_names(check_adapters_to_merge(self, adapter_names))
        if not names:
            return
        spec, original, components = merged_components(self, names, safe_merge)
        attributes = []
        if self._merge_state is None:
            bias = original.get("down_bias")
            attributes.extend(
                [
                    (self, "_merge_state", unmerge_state(self.base_layer, spec)),
                    (self, "_original_down_bias", None if bias is None else bias.detach().clone()),
                ]
            )
        attributes.append((self, "merged_adapters", self.merged_adapters + names))
        _apply_components(self.base_layer, components, spec=spec, attributes=attributes)

    def unmerge(self) -> None:
        from .merge import _apply_components, original_components
        from .structure import describe_mlp, read_components

        if self._merge_state is not None:
            check_no_offload(self)
            spec = describe_mlp(self.base_layer)
            components = original_components(
                read_components(self.base_layer, spec=spec), self._merge_state["width"], self._original_down_bias
            )
            attributes = [
                (module, name, value)
                for module, values in self._merge_state["attributes"]
                for name, value in values.items()
            ]
            attributes.extend(
                [(self, "_merge_state", None), (self, "_original_down_bias", None), (self, "merged_adapters", [])]
            )
            _apply_components(self.base_layer, components, spec=spec, attributes=attributes)

    def unload_and_optionally_merge_module(
        self, merge: bool, safe_merge: bool = False, adapter_names: list[str] | None = None
    ) -> nn.Module:
        if merge:
            self.merge(safe_merge=safe_merge, adapter_names=adapter_names)
        else:
            self.unmerge()
        return self.base_layer


class CortexMLP(CortexLayer):
    """Branch inside a dense or native shared FFN, preserving outer operations."""

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor | tuple[object, ...]:
        if self.disable_adapters and self.merged:
            self.unmerge()
        result = self.base_layer(hidden_states, *args, **kwargs)
        names = self.merge_names()
        if self.disable_adapters or not names:
            return result
        output = result[0] if isinstance(result, tuple) else result
        # The base layer already contains merged branches. Keep computing every
        # other active branch when only a subset of adapters has been merged.
        for name in names:
            output = output + self._branch(hidden_states, name).to(output.dtype)
        return (output, *result[1:]) if isinstance(result, tuple) else output


class CortexExperts(CortexLayer):
    """Reference backend using original routing without per-token weight copies.

    Each selected expert processes its token group with ordinary linear calls. This Python loop is a correctness
    backend, not a fused production kernel. The native base output and routed increments are accumulated separately;
    their mathematical sum need not reproduce the floating-point rounding of a legacy combined expert forward.
    """

    def _public_branch(
        self, hidden_states: torch.Tensor, name: str, weights: torch.Tensor, prefix: str = ""
    ) -> torch.Tensor:
        output = self._branch(hidden_states, name, prefix=prefix)
        if self.public_weighting[name] == "routing":
            # Compute shared parameters once, with the same contribution as
            # adding this branch inside every selected expert before routing.
            # Do not detach: routing and input gradients must also agree.
            output = output * weights.to(output.dtype).sum(dim=-1, keepdim=True)
        return output

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.disable_adapters and self.merged:
            self.unmerge()
        result = self.base_layer(hidden_states, *args, **kwargs)
        names = self.merge_names()
        if self.disable_adapters or not names:
            return result
        indices = args[0] if args else kwargs.get("top_k_index", kwargs.get("router_indices"))
        weights = args[1] if len(args) > 1 else kwargs.get("top_k_weights", kwargs.get("routing_weights"))
        if hidden_states.ndim != 2 or indices is None or weights is None or indices.shape != weights.shape:
            raise ValueError("Expected [tokens, hidden] inputs and [tokens, top_k] routing.")
        if indices.ndim != 2 or indices.shape[0] != hidden_states.shape[0]:
            raise ValueError("Routing indices and hidden states have inconsistent token counts.")
        for name in names:
            if self.per_expert:
                delta = torch.zeros_like(result)
                kinds = ("gate", "up", "down") if self.structure.has_gate else ("up", "down")
                views = {kind: getattr(self, f"cortex_{kind}")[name].expert_views() for kind in kinds}
                for expert in indices.unique().tolist():
                    if not 0 <= expert < self.structure.num_experts:
                        continue
                    token, slot = torch.where(indices == expert)
                    current = self._branch(hidden_states[token], name, expert, views=views)
                    current = current * weights[token, slot, None]
                    delta = delta.index_add(0, token, current.to(delta.dtype))
                result = result + delta
            else:
                result = result + self._public_branch(hidden_states, name, weights).to(result.dtype)
            if self.role == "routed_public":
                result = result + self._public_branch(hidden_states, name, weights, prefix="shared_").to(result.dtype)
        return result
