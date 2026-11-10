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

from .config import CortexConfig, CortexMoeSpec
from .initialization import initialize_branch
from .runtime import check_expert_forward, check_no_offload
from .structure import MlpStructure


class ExpertLinear(nn.Module):
    """One linear projection per expert, with weights stored as [E, out, in].

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
    """Shared base for Cortex adapters on dense FFNs and packed experts.

    Manages adapter projections, initialization, and merge state. Flat ModuleDicts keyed by adapter name keep parameter
    names compatible with PEFT checkpoints. The main branch uses cortex_*; routed_public stores its additional public
    branch in cortex_shared_*.
    """

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
        routing_args: tuple[str, str] | None = None,
        moe_spec: CortexMoeSpec | None = None,
    ) -> None:
        """Initialize the wrapper and its first adapter.

        Args:
            base_layer (`nn.Module`):
                Native FFN or packed experts module to wrap.
            adapter_name (`str`):
                Name of the first adapter added to this wrapper.
            config (`CortexConfig`):
                Configuration used to create and initialize the adapter.
            structure (`MlpStructure`):
                Resolved FFN dimensions, projection layout, activation, and field mappings.
            role (`str`, *optional*, defaults to `"dense"`):
                `dense` for an ordinary FFN, `shared` for a native shared FFN, `routed` for per-expert branches,
                `public` for a branch evaluated for every token, or `routed_public` for both routed and public
                branches.
            public_scale (`float`, *optional*):
                Proven constant sum of selected routing weights, used to rescale the down projection when merging an
                unweighted public branch. `None` means this sum has not been established.
            routing_args (`tuple[str, str]`, *optional*):
                Expert forward argument names in `(indices_name, weights_name)` order. For packed experts, `None`
                resolves these names from a supported native signature.
            moe_spec (`CortexMoeSpec`, *optional*):
                Resolved field mapping for the enclosing MoE block, when applicable.
        """
        super().__init__()
        self.base_layer = base_layer
        self.structure = structure
        self.in_features, self.out_features = structure.input_dim, structure.output_dim
        self.role = role
        self.per_expert = self.role in {"routed", "routed_public"}
        self.public_scale = public_scale
        self.routing_args = routing_args
        self.moe_spec = moe_spec
        if structure.packed and routing_args is None:
            self.routing_args = check_expert_forward(base_layer, type(base_layer).__name__)
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
            native_weight, native_bias = self._native_projection(kind)
            bias = config.adapter_bias
            if bias is None:
                bias = native_bias is not None
            layer = projection(*dims, bias=bias)
            # A meta placeholder has no storage yet, but its dtype still describes how to load the checkpoint.
            device = layer.weight.device if layer.weight.is_meta else native_weight.device
            layer.to(device=device, dtype=native_weight.dtype)
            getattr(self, f"cortex_{prefix}{kind}")[name] = layer

    def _native_projection(self, kind: str) -> tuple[nn.Parameter, nn.Parameter | None]:
        # Gate and up may share a single packed parameter. Inspect its metadata without gathering ZeRO shards.
        if self.structure.fused and kind in ("gate", "up"):
            kind = "gate_up"
        base = self.get_base_layer()
        name = getattr(self.structure.ffn_spec, kind + "_proj")
        projection = getattr(base, name)
        if self.structure.packed:
            return projection, getattr(base, name + "_bias", None)
        return projection.weight, projection.bias

    def _move_adapter_to_device_of_base_layer(self, adapter_name: str, device: torch.device | None = None) -> None:
        # Called after assign=True materializes the checkpoint. The prepared adapter dtype may be FP32
        # even when the native projection is BF16.
        for attr in self.adapter_layer_names:
            adapters = getattr(self, attr)
            if adapter_name not in adapters:
                continue
            adapter = adapters[adapter_name]
            if any(parameter.is_meta for parameter in adapter.parameters()):
                continue
            kind = attr.rsplit("_", 1)[-1]
            weight, _ = self._native_projection(kind)
            adapter.to(device=weight.device if device is None else device)

    def _branch(
        self, x: torch.Tensor, name: str, expert: int | None = None, prefix: str = "", views=None
    ) -> torch.Tensor:
        up = getattr(self, f"cortex_{prefix}up")[name]
        down = getattr(self, f"cortex_{prefix}down")[name]

        def project(kind, projection, inputs):
            if expert is None:
                return projection(inputs)
            return projection(inputs, expert, None if views is None else views[kind])

        up_input = self._cast_input_dtype(x, up.weight.dtype)
        gate_value = None
        if self.structure.has_gate:
            gate = getattr(self, f"cortex_{prefix}gate")[name]
            gate_input = up_input
            if gate.weight.dtype != up.weight.dtype:
                gate_input = self._cast_input_dtype(x, gate.weight.dtype)
            gate_value = project("gate", gate, gate_input)
        hidden = self.structure.activate(gate_value, project("up", up, up_input))
        return project("down", down, self._cast_input_dtype(hidden, down.weight.dtype))

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
            spec = describe_mlp(self.base_layer, spec=self.structure.ffn_spec)
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
    """Add Cortex branches to a dense or native shared FFN.

    The native forward supplies the base output; active, unmerged adapter branches are added to it. Any auxiliary tuple
    outputs are preserved.
    """

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
    """Add routed and public Cortex branches using the native expert selections and weights.

    Routed branches process each selected expert's token group and accumulate weighted increments. Public branches are
    evaluated once for all tokens, with optional weighting by the sum of routing weights.

    This reference backend uses a Python loop with ordinary linear calls, avoiding per-token weight copies. It has no
    fused kernel. The native base output and routed increments are accumulated separately, so floating-point rounding
    may differ from a combined expert forward.
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
        names = self.merge_names()
        if self.disable_adapters or not names:
            return self.base_layer(hidden_states, *args, **kwargs)

        index_name, weight_name = self.routing_args
        indices = args[0] if args else kwargs.get(index_name)
        weights = args[1] if len(args) > 1 else kwargs.get(weight_name)
        if not isinstance(indices, torch.Tensor) or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Cortex expects integer routing indices (int32 or int64).")
        if not isinstance(weights, torch.Tensor) or not weights.is_floating_point():
            raise ValueError("Cortex expects floating-point routing weights.")
        if (
            hidden_states.ndim != 2
            or indices.ndim != 2
            or indices.shape != weights.shape
            or indices.shape[0] != hidden_states.shape[0]
        ):
            raise ValueError(
                "Expected [tokens, hidden] inputs and matching [tokens, top_k] routing indices and weights."
            )

        result = self.base_layer(hidden_states, *args, **kwargs)
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
