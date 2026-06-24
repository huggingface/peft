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

from typing import Any, Optional

import torch
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer

from .config import ShadowConfig


class ShadowCarrier:
    """Couples the base `hidden_states` with the parallel `shadow_states` so the pair rides the base decoder loop."""

    def __init__(self, hidden: torch.Tensor, shadow: torch.Tensor) -> None:
        self.hidden = hidden
        self.shadow = shadow


class ShadowLayer(nn.Module, BaseTunerLayer):
    # Trainable, per-adapter sub-modules.
    adapter_layer_names = ("shadow_down", "shadow_up", "shadow_update_transform", "shadow_update_gate")
    # Non-tensor / non-trainable per-adapter bookkeeping.
    other_param_names = ("shadow_r", "shadow_alpha", "shadow_dropout")

    def __init__(self, base_layer: nn.Module, adapter_name: str, config: ShadowConfig, **kwargs) -> None:
        super().__init__()
        self.base_layer = base_layer

        self.shadow_r: dict[str, int] = {}
        self.shadow_alpha: dict[str, float] = {}
        self.shadow_down = nn.ModuleDict({})
        self.shadow_up = nn.ModuleDict({})
        self.shadow_dropout = nn.ModuleDict({})
        self.shadow_update_transform = nn.ModuleDict({})
        self.shadow_update_gate = nn.ModuleDict({})

        self._disable_adapters = False
        self.merged_adapters: list[str] = []
        self.kwargs = kwargs

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config)

    def update_layer(self, adapter_name: str, config: ShadowConfig) -> None:
        r = config.r
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        # `d` is the base block's hidden size. The shadow state always lives at this width so that the discrepancy
        # `h - s` in Eq. 2 is well defined; the shadow *backbone* may use a different width internally and project.
        d = self._base_hidden_size()
        update_hidden = config.update_hidden_size or r

        self.shadow_r[adapter_name] = r
        self.shadow_alpha[adapter_name] = config.shadow_alpha

        # Injection bottleneck (Eq. 3): W_down: d->r, W_up: r->d. W_up is zero-initialized so injection starts as a
        # no-op (Eq. 4 reduces to the identity), mirroring LoRA's B=0 convention.
        self.shadow_down[adapter_name] = nn.Linear(d, r, bias=False)
        self.shadow_up[adapter_name] = nn.Linear(r, d, bias=False)
        self.shadow_dropout[adapter_name] = (
            nn.Dropout(p=config.shadow_dropout) if config.shadow_dropout else nn.Identity()
        )
        if config.init_weights:
            nn.init.zeros_(self.shadow_up[adapter_name].weight)

        # Gated residual update (Eq. 5-6): lightweight two-layer MLPs producing the candidate `T(h)` and gate `G(h)`.
        self.shadow_update_transform[adapter_name] = nn.Sequential(
            nn.Linear(d, update_hidden), nn.GELU(), nn.Linear(update_hidden, d)
        )
        self.shadow_update_gate[adapter_name] = nn.Sequential(
            nn.Linear(d, update_hidden), nn.GELU(), nn.Linear(update_hidden, d)
        )

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def _base_hidden_size(self) -> int:
        """TODO: read the block's hidden size `d` (e.g. from the block's / model's config)."""

    def forward(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
        # `hidden_states` is a ShadowCarrier exactly when the shadow path is active for this pass (the entry pre-hook
        # wrapped it). Otherwise -- adapters disabled, or this block not adapted for the active adapter -- it is a plain
        # tensor and we behave exactly like the frozen block. This also makes the forward KV-cache-agnostic: a decode
        # step simply carries a (B, 1, d) hidden and its matching (B, 1, d) shadow.
        uses_shadow = isinstance(hidden_states, ShadowCarrier)
        if not uses_shadow:
            return self.base_layer(hidden_states, *args, **kwargs)

        adapter_name = self.active_adapters[0]  # there can only be one (enforced by ShadowModel)
        hidden, shadow = hidden_states.hidden, hidden_states.shadow

        # See Figure 2 and Sections 3.2-3.3 of the paper.
        injected = self.shadow_inject(hidden, shadow, adapter_name)  # Eq. 2-4 -> h^(l)
        out = self.base_layer(injected, *args, **kwargs)
        new_hidden = self._block_output_hidden_states(out)  # h_out^(l)
        new_shadow = self.shadow_update(new_hidden, shadow, adapter_name)  # Eq. 5-7 -> s^(l)
        # TODO: deal with more complex return values, e.g. CausalLMOutputWithPast
        return ShadowCarrier(new_hidden, new_shadow)

    def _block_output_hidden_states(self, out: Any) -> torch.Tensor:
        """TODO: pull the hidden-states tensor out of the block's return value."""

    def shadow_inject(
        self, hidden_states: torch.Tensor, shadow_states: torch.Tensor, adapter_name: str
    ) -> torch.Tensor:
        # Inject the shadow state into the block input (Section 3.2, Eq. 2-4).
        # delta       = h_out^(l-1) - s^(l-1)              (Eq. 2)
        # delta_tilde = Dropout(delta . W_down) . W_up     (Eq. 3)
        # h^(l)       = h_out^(l-1) + alpha * delta_tilde  (Eq. 4)
        delta = hidden_states - shadow_states  # Eq. 2
        delta = self.shadow_dropout[adapter_name](delta)
        delta = self.shadow_up[adapter_name](self.shadow_down[adapter_name](delta))  # Eq. 3
        return hidden_states + self.shadow_alpha[adapter_name] * delta  # Eq. 4

    def shadow_update(
        self, hidden_states: torch.Tensor, shadow_states: torch.Tensor, adapter_name: str
    ) -> torch.Tensor:
        # Advance the shadow state from the block output (Section 3.3, Eq. 5-7).
        # t^(l) = T^(l)(h_out^(l))                       (Eq. 5)
        # g^(l) = sigmoid(G^(l)(h_out^(l)))              (Eq. 6)
        # s^(l) = (1 - g^(l)) * s^(l-1) + g^(l) * t^(l)  (Eq. 7)
        candidate = self.shadow_update_transform[adapter_name](hidden_states)  # Eq. 5
        gate = torch.sigmoid(self.shadow_update_gate[adapter_name](hidden_states))  # Eq. 6
        return (1.0 - gate) * shadow_states + gate * candidate  # Eq. 7

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError(
            "ShadowPEFT cannot be merged into the base weights: the adaptation is an input-dependent, layer-space "
            "trajectory (the shadow state evolves with the data), not a static weight-space delta. Use "
            "`unload_shadow()` to obtain a standalone shadow model instead."
        )

    def unmerge(self) -> None:
        raise NotImplementedError("ShadowPEFT does not support merging, so there is nothing to unmerge.")

    def __repr__(self) -> str:
        return "shadow." + super().__repr__()
