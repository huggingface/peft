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

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer

from .config import ShadowConfig
from .layers import ShadowCarrier, ShadowLayer


class ShadowModel(BaseTuner):
    """TODO"""

    prefix: str = "shadow_"
    tuner_layer_cls = ShadowLayer
    target_module_mapping: dict = {}

    def _ensure_shadow_containers(self) -> None:
        """Create the model-level containers on first use (BaseTuner has no custom __init__ to hook into)."""
        if not hasattr(self, "shadow_backbone"):
            # Registered as submodules so their params are saved/loaded (names contain the "shadow_" prefix) and moved
            # with `.to(...)` alongside the rest of the model.
            self.shadow_backbone = nn.ModuleDict({})
            self.shadow_head = nn.ModuleDict({})
        if not hasattr(self, "_boundary_hook_handles"):
            self._boundary_hook_handles: list = []

    def _create_and_replace(
        self,
        shadow_config: ShadowConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        **optional_kwargs,
    ) -> None:
        self._ensure_shadow_containers()

        # Build the model-level shadow backbone + head for this adapter exactly once (before/at the first block).
        if adapter_name not in self.shadow_backbone:
            self._init_shadow_backbone(adapter_name, shadow_config)

        if isinstance(target, ShadowLayer):
            target.update_layer(adapter_name, shadow_config)
        else:
            new_module = self._create_new_module(shadow_config, adapter_name, target)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(shadow_config, adapter_name, target, **kwargs) -> ShadowLayer:
        target_base_layer = target.get_base_layer() if isinstance(target, BaseTunerLayer) else target
        return ShadowLayer(target_base_layer, adapter_name, config=shadow_config, **kwargs)

    def _init_shadow_backbone(self, adapter_name: str, config: ShadowConfig) -> None:
        if config.shadow_model == "auto":
            backbone = self._build_shadow_backbone(config)
        else:
            # TODO use AutoConfig / AutoModel to instantiate (or load a pretrained) shadow backbone from the id/type.
            backbone = self._load_shadow_backbone(config)
        self._check_backbone_compatible(backbone)
        self.shadow_backbone[adapter_name] = backbone
        head = self._build_shadow_head(config)
        if head is not None:
            self.shadow_head[adapter_name] = head

    def _build_shadow_backbone(self, config: ShadowConfig) -> nn.Module:
        """TODO"""

    def _load_shadow_backbone(self, config: ShadowConfig) -> nn.Module:
        """TODO"""

    def _build_shadow_head(self, config: ShadowConfig) -> Optional[nn.Module]:
        """TODO"""

    def _check_backbone_compatible(self, backbone: nn.Module) -> None:
        """TODO"""

    def _post_injection_hook(self, model: nn.Module, config: ShadowConfig, adapter_name: str) -> None:
        # The shadow state rides the base decoder loop; we only need to bridge the two ends. Register a pre-hook on the
        # first wrapped block (seed s^(0), wrap into a carrier) and a post-hook on the last (unwrap to a plain tensor).
        self._ensure_shadow_containers()
        self._register_boundary_hooks()

    def _register_boundary_hooks(self) -> None:
        for handle in self._boundary_hook_handles:
            handle.remove()
        self._boundary_hook_handles = []

        wrapped = [module for _, module in self.model.named_modules() if isinstance(module, ShadowLayer)]
        if not wrapped:
            return

        # TODO: assert the wrapped blocks are contiguous in the stack and form the modern "layer returns hidden tensor"
        # contract; otherwise the carrier would be handed to an unadapted block / unwrapped via `layer_outputs[0]`.
        entry, exit_ = wrapped[0], wrapped[-1]
        self._boundary_hook_handles.append(
            entry.register_forward_pre_hook(self._seed_entry_pre_hook, with_kwargs=True)
        )
        self._boundary_hook_handles.append(exit_.register_forward_hook(self._unwrap_exit_hook, with_kwargs=True))

    def _seed_entry_pre_hook(self, module: ShadowLayer, args: tuple, kwargs: dict):
        """Wrap the first block's input (the embeddings) into a [`ShadowCarrier`] seeded with ``s^(0)`` (Eq. 1)."""
        adapter_name = self.active_adapters[0]  # there can only be one
        if module.disable_adapters or adapter_name not in module.shadow_down:
            return  # shadow path off: leave the plain tensor; every block falls through to its frozen base
        hidden = args[0] if args else kwargs["hidden_states"]
        s0 = self._compute_initial_shadow_state(adapter_name, hidden, kwargs)
        carrier = ShadowCarrier(hidden, s0)
        if args:
            return (carrier, *args[1:]), kwargs
        return args, {**kwargs, "hidden_states": carrier}

    def _unwrap_exit_hook(self, module: ShadowLayer, args: tuple, kwargs: dict, output: Any):
        """Unwrap the last block's carrier back to a plain hidden-states tensor for the base model's final norm."""
        if not isinstance(output, ShadowCarrier):
            return output
        # TODO: if the classification aux loss (Eq. 9) needs the final shadow state s^(L), stash `output.shadow` here.
        return output.hidden

    def _compute_initial_shadow_state(
        self, adapter_name: str, hidden_states: torch.Tensor, kwargs: dict
    ) -> torch.Tensor:
        """TODO: run the shadow backbone to produce s^(0)."""

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        super()._mark_only_adapters_as_trainable(model)
        # The model-level backbones/heads live on the tuner, not on `model`; only the active adapter's pieces train.
        for adapter_name, backbone in self.shadow_backbone.items():
            backbone.requires_grad_(adapter_name in self.active_adapters)
        for adapter_name, head in self.shadow_head.items():
            head.requires_grad_(adapter_name in self.active_adapters)
        # TODO: re-freeze any parameters tied to `self.model.get_input_embeddings()` (shared embeddings stay frozen).

    @property
    def active_adapters(self) -> list[str]:
        """There can only be *one* active adapter: the shadow state is a single trajectory through the network."""
        adapter_names = super().active_adapters
        if len(adapter_names) > 1:
            raise ValueError(
                f"ShadowPEFT supports only one active adapter at a time, but {len(adapter_names)} are active."
            )
        return adapter_names

    def _check_merge_allowed(self):
        raise NotImplementedError(
            "ShadowPEFT cannot be merged into the base model. Use `unload()` to recover the plain base model, or "
            "`unload_shadow()` for a standalone shadow model."
        )

    def unload_shadow(self, adapter_name: Optional[str] = None) -> nn.Module:
        """Return the shadow backbone (+ head) as a standalone model, *without* the base model.

        The ShadowPEFT analogue of `merge_and_unload`: where that hands back the base model with the adaptation baked
        in, this hands back only the lightweight shadow network for high-efficiency / edge inference (Section 3.5). It
        runs `head(f_shadow(x))` only -- the per-block updates require the base outputs and so do not exist standalone.

        Assign the result to a variable; this is not an in-place operation.
        """
        self._ensure_shadow_containers()
        if adapter_name is None:
            adapter_name = self.active_adapters[0]
        if adapter_name not in self.shadow_backbone:
            raise ValueError(f"No shadow backbone found for adapter '{adapter_name}'.")
        # TODO: construct the detached model

    def delete_adapter(self, adapter_name: str) -> None:
        # Remove the per-block adapter weights via the base implementation, then drop the model-level pieces too.
        super().delete_adapter(adapter_name)
        for container in (self.shadow_backbone, self.shadow_head):
            if adapter_name in container:
                del container[adapter_name]
        # Coverage may have changed (the deleted adapter's blocks could be unwrapped now); rebind the boundary hooks.
        self._register_boundary_hooks()

    def shadow_auxiliary_loss(self, labels: torch.Tensor) -> torch.Tensor:
        """TODO: auxiliary shadow loss (Eq. 8-9), `lambda * CE(shadow_head(s), labels)`, added to the task loss so the"""
