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

import contextlib
import inspect
from copy import deepcopy
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TaskType

from .config import ShadowConfig
from .layers import DetachedShadowModel, ShadowCache, ShadowCarrier, ShadowLayer


# --------------------------------------------------------------------------------------------------- backbone helpers


def _get_backbone(model: nn.Module) -> nn.Module:
    """Return the module that holds the transformer decoder stack (e.g. `LlamaModel` inside `LlamaForCausalLM`)."""
    for attr in ("model", "transformer", "base_model", "decoder"):
        backbone = getattr(model, attr, None)
        if isinstance(backbone, nn.Module):
            return backbone
    if hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList):
        return model
    if hasattr(model, "h") and isinstance(model.h, nn.ModuleList):
        return model
    raise AttributeError("Unable to locate the transformer backbone inside the supplied model.")


def _get_decoder_layers(model: nn.Module) -> tuple[nn.ModuleList, str]:
    """Return `(layers, attr_name)` for the decoder-layer `nn.ModuleList` of a model."""
    backbone = _get_backbone(model)
    for attr in ("layers", "h"):
        candidate = getattr(backbone, attr, None)
        if isinstance(candidate, nn.ModuleList):
            return candidate, attr
    raise AttributeError(
        "Unsupported model: cannot find a `nn.ModuleList` of decoder layers (expected `.layers`/`.h`)."
    )


def _get_hidden_size(config: Any) -> int:
    for attr in ("hidden_size", "n_embd", "d_model"):
        if hasattr(config, attr):
            return int(getattr(config, attr))
    raise AttributeError("Unable to infer the hidden size from the model config.")


def _set_config_attr(config: Any, names: tuple[str, ...], value: int) -> bool:
    for attr in names:
        if hasattr(config, attr):
            setattr(config, attr, int(value))
            return True
    return False


def _normalize_layer_config(config: Any) -> Any:
    """Keep per-layer config lists consistent with a reduced `num_hidden_layers` (e.g. Qwen3 `layer_types`)."""
    try:
        num_layers = int(config.num_hidden_layers)
    except Exception:
        return config

    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        layer_types = list(layer_types)
        if not layer_types:
            layer_types = ["full_attention"] * num_layers
        elif len(layer_types) > num_layers:
            layer_types = layer_types[:num_layers]
        else:
            layer_types = layer_types + [layer_types[-1]] * (num_layers - len(layer_types))
        config.layer_types = layer_types

    max_window = getattr(config, "max_window_layers", None)
    if max_window is not None:
        with contextlib.suppress(Exception):
            if int(max_window) > num_layers:
                config.max_window_layers = num_layers
    return config


def _remove_embed_tokens(module: nn.Module) -> None:
    """Drop `embed_tokens` so a backbone can be driven purely via shared base `inputs_embeds`."""
    if hasattr(module, "embed_tokens") and isinstance(module.embed_tokens, nn.Module):
        try:
            module.embed_tokens = None
        except Exception:
            module._modules.pop("embed_tokens", None)


def _shifted_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)


def _pool_last_token(hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Pool the last non-padding token representation (used for sequence classification)."""
    if attention_mask is None:
        return hidden[:, -1, :]
    token_counts = (attention_mask.long().sum(dim=1) - 1).clamp(min=0)
    batch_idx = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch_idx, token_counts]


# ------------------------------------------------------------------------------------------------------------- tuner


class ShadowModel(BaseTuner):
    """
    Creates a ShadowPEFT model from a pretrained transformers model.

    ShadowPEFT augments a frozen base decoder-only model with a small, trainable parallel *shadow* network. A shadow
    backbone produces an initial shadow state that rides the base decoder loop; at every targeted block the
    discrepancy between the base hidden states and the shadow state is injected into the block, and the shadow state is
    advanced by a gated residual update. Only the shadow components are trained. See [`ShadowConfig`] for the
    configuration.

    The method cannot be merged into the base weights (the adaptation is an input-dependent trajectory, not a static
    weight delta); use [`ShadowModel.unload_shadow`] to obtain the standalone shadow network instead.
    """

    prefix: str = "shadow_"
    tuner_layer_cls = ShadowLayer
    target_module_mapping: dict = {}

    # ---------------------------------------------------------------------------------------------- config / setup

    def _prepare_adapter_config(self, peft_config: ShadowConfig, model_config: dict) -> ShadowConfig:
        if peft_config.target_modules is None:
            # Default to wrapping *every* decoder block (contiguous, which the shadow carrier requires).
            layers, attr = _get_decoder_layers(self.model)
            backbone = _get_backbone(self.model)
            prefix = None
            for name, module in self.model.named_modules():
                if module is backbone:
                    prefix = name
                    break
            base = f"{prefix}.{attr}" if prefix else attr
            peft_config.target_modules = {f"{base}.{i}" for i in range(len(layers))}
        return peft_config

    def _base_hidden_size(self) -> int:
        return _get_hidden_size(self.model.config)

    def _ensure_shadow_containers(self) -> None:
        """Create the model-level containers on first use (`BaseTuner` has no custom `__init__` to hook into)."""
        if not hasattr(self, "shadow_backbone"):
            # Registered as submodules so their params are saved/loaded (names contain the "shadow_" prefix) and moved
            # with `.to(...)` alongside the rest of the model.
            self.shadow_backbone = nn.ModuleDict({})
            self.shadow_projection = nn.ModuleDict({})
            self.shadow_head = nn.ModuleDict({})
        if not hasattr(self, "_boundary_hook_handles"):
            self._boundary_hook_handles: list = []
            # Held in a plain list so the entry/exit blocks are NOT re-registered as submodules of the tuner (they are
            # already registered under `self.model`); a direct attribute assignment would duplicate their parameters.
            self._boundary_layers: list = []
            self._seed_shadow_state: Optional[torch.Tensor] = None
            self._shadow_past_out: Any = None
            self._should_pack_shadow_cache: bool = False
            self._shadow_share_embeddings: dict[str, bool] = {}
            self._shadow_head_is_lm: dict[str, bool] = {}

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

        # Build the model-level shadow backbone + head for this adapter exactly once.
        if adapter_name not in self.shadow_backbone:
            self._init_shadow_backbone(adapter_name, shadow_config)

        if isinstance(target, ShadowLayer):
            target.update_layer(adapter_name, shadow_config)
        else:
            new_module = self._create_new_module(shadow_config, adapter_name, target)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _create_new_module(self, shadow_config: ShadowConfig, adapter_name: str, target: nn.Module) -> ShadowLayer:
        target_base_layer = target.get_base_layer() if isinstance(target, BaseTunerLayer) else target
        return ShadowLayer(target_base_layer, adapter_name, config=shadow_config, hidden_size=self._base_hidden_size())

    # ------------------------------------------------------------------------------------------ shadow backbone

    def _init_shadow_backbone(self, adapter_name: str, config: ShadowConfig) -> None:
        loaded_projection = None
        if config.shadow_model == "mirror":
            backbone = self._build_shadow_backbone(config)
        else:
            backbone, loaded_projection = self._load_shadow_backbone(config)

        base_hidden = self._base_hidden_size()
        shadow_hidden = _get_hidden_size(backbone.config)
        if loaded_projection is not None:
            # A pretrained (projected) shadow checkpoint carries a trained shadow_hidden -> base_hidden projection.
            if (loaded_projection.in_features, loaded_projection.out_features) != (shadow_hidden, base_hidden):
                raise ValueError(
                    f"The loaded shadow projection maps {loaded_projection.in_features}->"
                    f"{loaded_projection.out_features}, but the shadow/base hidden sizes are {shadow_hidden}/"
                    f"{base_hidden}."
                )
            projection = loaded_projection
        elif shadow_hidden != base_hidden:
            projection = nn.Linear(shadow_hidden, base_hidden, bias=False)
        else:
            projection = nn.Identity()

        # Whether we can (and should) drive the shadow backbone with the shared frozen base `inputs_embeds`.
        try:
            supports_inputs_embeds = "inputs_embeds" in inspect.signature(backbone.forward).parameters
        except (TypeError, ValueError):
            supports_inputs_embeds = False
        # Sharing feeds the frozen base `inputs_embeds` into the shadow backbone, which only works when the widths
        # match (a "mirror" backbone at the base hidden size).
        share = (
            config.share_embeddings
            and config.shadow_model == "mirror"
            and supports_inputs_embeds
            and shadow_hidden == base_hidden
        )
        if share:
            _remove_embed_tokens(backbone)

        head, head_is_lm = self._build_shadow_head(config)

        self.shadow_backbone[adapter_name] = backbone
        self.shadow_projection[adapter_name] = projection
        if head is not None:
            self.shadow_head[adapter_name] = head
        self._shadow_share_embeddings[adapter_name] = share
        self._shadow_head_is_lm[adapter_name] = head_is_lm

        # Match the freshly created shadow modules to the base model's device/dtype (the base is often loaded in
        # fp16/bf16 while new modules default to fp32).
        base_param = next(self.model.parameters(), None)
        if base_param is not None:
            for module in (backbone, projection, head):
                if module is not None:
                    module.to(device=base_param.device, dtype=base_param.dtype)

    def _build_shadow_backbone(self, config: ShadowConfig) -> nn.Module:
        """Build a fresh, randomly-initialized backbone mirroring the base architecture (Section 3.1)."""
        base_backbone = _get_backbone(self.model)
        cfg = deepcopy(base_backbone.config)
        _set_config_attr(cfg, ("num_hidden_layers", "n_layer", "num_layers"), config.shadow_num_hidden_layers or 1)
        if config.shadow_hidden_size is not None:
            _set_config_attr(cfg, ("hidden_size", "n_embd", "d_model"), config.shadow_hidden_size)
        if config.shadow_intermediate_size is not None:
            _set_config_attr(cfg, ("intermediate_size", "ffn_dim", "n_inner"), config.shadow_intermediate_size)
        if config.shadow_num_attention_heads is not None:
            _set_config_attr(cfg, ("num_attention_heads", "n_head", "num_heads"), config.shadow_num_attention_heads)
        _normalize_layer_config(cfg)
        return base_backbone.__class__(cfg)

    def _load_shadow_backbone(self, config: ShadowConfig) -> tuple[nn.Module, Optional[nn.Module]]:
        """Load a pretrained shadow backbone from the id/path in `config.shadow_model`.

        Returns `(backbone, projection)`. `projection` is `None` for a plain `AutoModel`, or a trained
        shadow_hidden -> base_hidden `nn.Linear` when loading a "projected" shadow checkpoint (`model_type ==
        'causal_lm_with_hidden_projection'`), which bundles a small backbone with a projection aligned to a larger base.
        """
        import json

        from transformers import AutoModel
        from transformers.utils import cached_file

        config_file = cached_file(config.shadow_model, "config.json")
        with open(config_file) as f:
            raw_config = json.load(f)
        if raw_config.get("model_type") == "causal_lm_with_hidden_projection":
            return self._load_projected_shadow_backbone(config, raw_config)
        return AutoModel.from_pretrained(config.shadow_model), None

    @staticmethod
    def _load_projected_shadow_backbone(config: ShadowConfig, raw_config: dict) -> tuple[nn.Module, nn.Module]:
        """Load the backbone + trained projection out of a `causal_lm_with_hidden_projection` checkpoint."""
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel
        from transformers.utils import cached_file

        inner = dict(raw_config["shadow_model_config"])
        model_type = inner.pop("model_type")
        backbone = AutoModel.from_config(AutoConfig.for_model(model_type, **inner))

        weights = load_file(cached_file(config.shadow_model, "model.safetensors"))
        backbone_prefix = "shadow_model."
        backbone_state = {
            key[len(backbone_prefix) :]: value for key, value in weights.items() if key.startswith(backbone_prefix)
        }
        backbone.load_state_dict(backbone_state, strict=False)

        projection_weight = weights.get("shadow_hidden_projection.weight")
        if projection_weight is None:
            raise ValueError(
                f"'{config.shadow_model}' is a projected shadow checkpoint but has no 'shadow_hidden_projection.weight'."
            )
        out_features, in_features = projection_weight.shape
        projection = nn.Linear(in_features, out_features, bias=False)
        projection.weight.data = projection_weight.to(projection.weight.dtype)
        return backbone, projection

    def _build_shadow_head(self, config: ShadowConfig) -> tuple[Optional[nn.Module], bool]:
        """The task head applied to the final shadow state for the auxiliary loss (Eq. 8-9).

        Returns `(head, head_is_lm)`. For the (large) causal-LM head we avoid storing a copy unless the user opts into
        training it via `modules_to_save=["shadow_lm_head"]`; otherwise the frozen base head is reused at loss time
        (`shadow_auxiliary_loss`). The (small) classifier head is copied and trained by default.
        """
        if config.task_type == TaskType.CAUSAL_LM:
            if "shadow_lm_head" not in set(config.modules_to_save or []):
                return None, True  # reuse the frozen base LM head at loss time
            base_head = self.model.get_output_embeddings()
            return (deepcopy(base_head), True) if base_head is not None else (None, True)
        if config.task_type == TaskType.SEQ_CLS:
            for attr in ("score", "classifier"):
                candidate = getattr(self.model, attr, None)
                if isinstance(candidate, nn.Module):
                    return deepcopy(candidate), False
            return None, False
        return None, False

    # -------------------------------------------------------------------------------------------- boundary hooks

    def _post_injection_hook(self, model: nn.Module, config: ShadowConfig, adapter_name: str) -> None:
        # The shadow state rides the base decoder loop; boundary hooks seed `s^(0)`, wrap/unwrap the carrier, and pack
        # the dual KV cache (`ShadowCache`) when `use_cache=True`.
        self._ensure_shadow_containers()
        self._register_boundary_hooks()

    def _register_boundary_hooks(self) -> None:
        for handle in self._boundary_hook_handles:
            handle.remove()
        self._boundary_hook_handles = []

        wrapped = [module for _, module in self.model.named_modules() if isinstance(module, ShadowLayer)]
        if not wrapped:
            self._boundary_layers = []
            return

        entry, exit_ = wrapped[0], wrapped[-1]
        self._boundary_layers = [entry, exit_]
        # Seed `s^(0)` from the *raw* model inputs (input_ids / 2D attention mask), which are only available at the top
        # of the base model's forward -- inside a decoder block the mask is already a 4D causal mask. Also unpack a
        # `ShadowCache` so the base model only sees its own past.
        self._boundary_hook_handles.append(
            self.model.register_forward_pre_hook(self._seed_shadow_pre_hook, with_kwargs=True)
        )
        # Re-pack base + shadow pasts into a `ShadowCache` on the way out (generation threads this object as
        # `past_key_values`).
        self._boundary_hook_handles.append(
            self.model.register_forward_hook(self._pack_shadow_cache_hook, with_kwargs=True)
        )
        # Wrap the first wrapped block's input into a carrier, and unwrap the last block's output back to a tensor.
        self._boundary_hook_handles.append(
            entry.register_forward_pre_hook(self._wrap_entry_pre_hook, with_kwargs=True)
        )
        self._boundary_hook_handles.append(exit_.register_forward_hook(self._unwrap_exit_hook, with_kwargs=True))

    def _shadow_path_active(self) -> bool:
        if not self._boundary_layers:
            return False
        entry = self._boundary_layers[0]
        if entry.disable_adapters:
            return False
        active = self.active_adapters
        return bool(active) and active[0] in entry.shadow_down

    @staticmethod
    def _unpack_past_key_values(past: Any) -> tuple[Any, Any]:
        """Split a [`ShadowCache`] into `(base_past, shadow_past)`; plain pasts are treated as base-only."""
        if isinstance(past, ShadowCache):
            return past.base, past.shadow
        return past, None

    def _seed_shadow_pre_hook(self, module: nn.Module, args: tuple, kwargs: dict):
        self._seed_shadow_state = None
        self._shadow_past_out = None
        self._should_pack_shadow_cache = False

        past = kwargs.get("past_key_values")
        base_past, shadow_past = self._unpack_past_key_values(past)
        # Always rewrite kwargs when a ShadowCache was supplied -- the base model cannot consume it.
        if isinstance(past, ShadowCache):
            kwargs = {**kwargs, "past_key_values": base_past}

        if not self._shadow_path_active():
            return args, kwargs

        use_cache = kwargs.get("use_cache")
        if use_cache is None:
            use_cache = bool(getattr(getattr(self.model, "config", None), "use_cache", False))

        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        self._seed_shadow_state, self._shadow_past_out = self._compute_initial_shadow_state(
            self.active_adapters[0],
            input_ids=input_ids,
            attention_mask=kwargs.get("attention_mask"),
            position_ids=kwargs.get("position_ids"),
            inputs_embeds=kwargs.get("inputs_embeds"),
            past_key_values=shadow_past,
            use_cache=use_cache,
        )
        self._should_pack_shadow_cache = bool(use_cache)
        return args, kwargs

    def _pack_shadow_cache_hook(self, module: nn.Module, args: tuple, kwargs: dict, output: Any):
        """Attach a [`ShadowCache`] so the next decode step can advance both paths incrementally."""
        if not self._should_pack_shadow_cache:
            return output
        self._should_pack_shadow_cache = False
        shadow_past = self._shadow_past_out
        self._shadow_past_out = None

        if hasattr(output, "past_key_values"):
            output.past_key_values = ShadowCache(base=output.past_key_values, shadow=shadow_past)
            return output
        return output

    def _wrap_entry_pre_hook(self, module: ShadowLayer, args: tuple, kwargs: dict):
        """Wrap the first block's input (the embeddings) into a [`ShadowCarrier`] seeded with `s^(0)` (Eq. 1)."""
        if self._seed_shadow_state is None:
            return
        hidden = args[0] if args else kwargs["hidden_states"]
        if hidden.shape[:-1] != self._seed_shadow_state.shape[:-1]:
            raise ValueError(
                f"Shadow state sequence shape {tuple(self._seed_shadow_state.shape[:-1])} does not match base hidden "
                f"states {tuple(hidden.shape[:-1])}. When using a KV cache, both the base model and the shadow "
                "backbone must see the same new-token length (pass a `ShadowCache` as `past_key_values`)."
            )
        carrier = ShadowCarrier(hidden, self._seed_shadow_state)
        if args:
            return (carrier, *args[1:]), kwargs
        return args, {**kwargs, "hidden_states": carrier}

    def _unwrap_exit_hook(self, module: ShadowLayer, args: tuple, kwargs: dict, output: Any):
        """Unwrap the last block's carrier back to a plain hidden-states tensor for the base model's final norm."""
        if not isinstance(output, ShadowCarrier):
            return output
        return output.hidden

    def _compute_initial_shadow_state(
        self,
        adapter_name: str,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        past_key_values: Any = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Any]:
        backbone = self.shadow_backbone[adapter_name]
        share = self._shadow_share_embeddings.get(adapter_name, False)

        kwargs = {
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "return_dict": True,
            "past_key_values": past_key_values,
        }
        if position_ids is not None:
            kwargs["position_ids"] = position_ids

        if share and inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either `input_ids` or `inputs_embeds` must be provided.")
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if share or inputs_embeds is not None:
            out = backbone(inputs_embeds=inputs_embeds, **kwargs)
        else:
            out = backbone(input_ids=input_ids, **kwargs)

        if not hasattr(out, "last_hidden_state"):
            raise TypeError("The shadow backbone did not return a `last_hidden_state`; architecture is unsupported.")
        shadow_state = self.shadow_projection[adapter_name](out.last_hidden_state)
        return shadow_state, getattr(out, "past_key_values", None)

    # ------------------------------------------------------------------------------------------ trainability

    @staticmethod
    def _freeze_backbone_embeddings(backbone: nn.Module) -> None:
        """Keep a pretrained shadow backbone's (large) input/output embeddings frozen -- fine-tuning the embedding
        table is not parameter-efficient (it can dominate the trainable parameter count)."""
        for getter in ("get_input_embeddings", "get_output_embeddings"):
            embed = None
            with contextlib.suppress(Exception):
                embed = getattr(backbone, getter)()
            if isinstance(embed, nn.Module):
                embed.requires_grad_(False)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        super()._mark_only_adapters_as_trainable(model)
        # The model-level backbones/heads live on the tuner, not on `model`; only the active adapter's pieces train.
        for adapter_name, backbone in self.shadow_backbone.items():
            backbone.requires_grad_(adapter_name in self.active_adapters)
            # For a pretrained shadow backbone, keep its embeddings frozen. (A "mirror" backbone either shares the
            # frozen base embeddings -- absent here -- or has randomly-initialized ones that must stay trainable.)
            if self.peft_config[adapter_name].shadow_model != "mirror":
                self._freeze_backbone_embeddings(backbone)
        for adapter_name, projection in self.shadow_projection.items():
            projection.requires_grad_(adapter_name in self.active_adapters)
        # A head is only stored when it is meant to be trained (the SEQ_CLS classifier, or an opt-in copy of the LM head
        # via `modules_to_save=["shadow_lm_head"]`); the frozen base LM head is reused directly and never stored.
        for adapter_name, head in self.shadow_head.items():
            head.requires_grad_(adapter_name in self.active_adapters)

    # ------------------------------------------------------------------------------------------- public forward

    def forward(self, *args: Any, **kwargs: Any):
        labels = kwargs.get("labels")
        attention_mask = kwargs.get("attention_mask")
        output = self.model(*args, **kwargs)

        # Compute the shadow path's own task loss (Eq. 8-9) on the standalone prediction head(s^(0)) -- `s^(0)` is
        # `_seed_shadow_state`, set by the seed pre-hook during the forward above. Expose it (unweighted, for
        # logging/inspection) as `output.shadow_loss` *and* on the tuner as `last_shadow_loss` -- the latter survives
        # DDP/FSDP, which reconstruct the model output from its registered fields and drop extra attributes. Then add
        # its `auxiliary_loss_weight`-scaled contribution to the task loss.
        self.last_shadow_loss = None
        if labels is not None and getattr(output, "loss", None) is not None and self._shadow_path_active():
            shadow_loss = self.shadow_auxiliary_loss(labels, attention_mask=attention_mask)
            if shadow_loss is not None:
                self.last_shadow_loss = shadow_loss.detach()
                output.shadow_loss = self.last_shadow_loss
                weight = self.peft_config[self.active_adapters[0]].auxiliary_loss_weight
                if weight > 0:
                    output.loss = output.loss + weight * shadow_loss
        return output

    def _resolve_shadow_head(self, adapter_name: str) -> Optional[nn.Module]:
        """The stored (trainable) shadow head, or the frozen base LM head for the default causal-LM case."""
        if adapter_name in self.shadow_head:
            return self.shadow_head[adapter_name]
        if self._shadow_head_is_lm.get(adapter_name, False):
            return self.model.get_output_embeddings()
        return None

    def shadow_auxiliary_loss(
        self, labels: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """The shadow path's own task loss, `CE(shadow_head(s^(0)), labels)` (unweighted; `forward` applies the weight).

        The loss is computed on the *initial* shadow state `s^(0)` (the shadow backbone output, projected) -- exactly
        what the standalone `unload_shadow()` model computes as `head(projection(backbone(x)))`. This is what makes the
        detached shadow network usable on its own; training it on `s^(L)` (the final state, which depends on the base
        model's per-layer outputs and does not exist standalone) would leave the detached model untrained.
        """
        adapter_name = self.active_adapters[0]
        shadow_state = self._seed_shadow_state
        head = self._resolve_shadow_head(adapter_name)
        if shadow_state is None or head is None:
            return None

        if self._shadow_head_is_lm.get(adapter_name, False):
            shadow_logits = head(shadow_state)
            return _shifted_ce_loss(shadow_logits, labels)
        pooled = _pool_last_token(shadow_state, attention_mask)
        shadow_logits = head(pooled)
        return F.cross_entropy(shadow_logits, labels)

    # ------------------------------------------------------------------------------------- PEFT tuner interface

    @property
    def active_adapters(self) -> list[str]:
        adapter_names = super().active_adapters
        if len(adapter_names) > 1:
            raise ValueError(
                f"ShadowPEFT supports only one active adapter at a time, but {len(adapter_names)} are active."
            )
        return adapter_names

    def _check_merge_allowed(self):
        raise NotImplementedError(
            "ShadowPEFT cannot be merged into the base model: the adaptation is an input-dependent, layer-space "
            "trajectory, not a static weight-space delta. Use `unload()` to recover the plain base model, or "
            "`unload_shadow()` for a standalone shadow model."
        )

    def delete_adapter(self, adapter_name: str) -> None:
        super().delete_adapter(adapter_name)
        for container in (self.shadow_backbone, self.shadow_projection, self.shadow_head):
            if adapter_name in container:
                del container[adapter_name]
        for bookkeeping in (
            self._shadow_share_embeddings,
            self._shadow_head_is_lm,
        ):
            bookkeeping.pop(adapter_name, None)
        # Coverage may have changed (the deleted adapter's blocks could be unwrapped now); rebind the boundary hooks.
        self._register_boundary_hooks()

    def unload_shadow(self, adapter_name: Optional[str] = None) -> nn.Module:
        """Return the shadow backbone (+ head) as a standalone model, *without* the base model.

        The ShadowPEFT analogue of `merge_and_unload`: where that would hand back the base model with the adaptation
        baked in, this hands back only the lightweight shadow network for high-efficiency / edge inference. It runs
        `head(projection(backbone(x)))` -- the per-block updates require the base outputs and so do not exist
        standalone. The result behaves like a normal causal LM (supports `generate()` and KV caching), which is how you
        evaluate the shadow path's own performance, independent of the base model.

        Assign the result to a variable; this is not an in-place operation.
        """
        self._ensure_shadow_containers()
        if adapter_name is None:
            adapter_name = self.active_adapters[0]
        if adapter_name not in self.shadow_backbone:
            raise ValueError(f"No shadow backbone found for adapter '{adapter_name}'.")
        backbone = deepcopy(self.shadow_backbone[adapter_name])
        # If the backbone shared the frozen base embeddings (its own `embed_tokens` was removed), re-attach a copy so
        # the detached model is self-contained and can run from `input_ids`.
        if self._shadow_share_embeddings.get(adapter_name, False) and hasattr(backbone, "embed_tokens"):
            if getattr(backbone, "embed_tokens", None) is None:
                backbone.embed_tokens = deepcopy(self.model.get_input_embeddings())
        projection = deepcopy(self.shadow_projection[adapter_name])
        head = self._resolve_shadow_head(adapter_name)
        is_classification = self.peft_config[adapter_name].task_type == TaskType.SEQ_CLS
        return DetachedShadowModel(
            backbone,
            projection,
            deepcopy(head) if head is not None else None,
            is_classification=is_classification,
        )
