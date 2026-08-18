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
import json
from copy import deepcopy
from typing import Any, Optional

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig
from transformers.utils import cached_file

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TaskType

from .config import ShadowConfig
from .diffusers import (
    DetachedFluxShadowModel,
    build_detached_flux_shadow,
    freeze_flux_stem_embeddings,
    validate_pretrained_flux_shadow,
)
from .layers import DetachedShadowModel, ShadowCache, ShadowCarrier, ShadowLayer


# --------------------------------------------------------------------------------------------------- backbone helpers


def _is_flux_like(model: nn.Module) -> bool:
    """Diffusers Flux / Flux2 transformers expose `transformer_blocks` instead of HF `layers`/`h`."""
    return isinstance(getattr(model, "single_transformer_blocks", None), nn.ModuleList) or (
        isinstance(getattr(model, "transformer_blocks", None), nn.ModuleList)
        and not isinstance(getattr(model, "layers", None), nn.ModuleList)
    )


def _get_backbone(model: nn.Module) -> nn.Module:
    """Return the module that holds the transformer decoder stack (e.g. `LlamaModel` inside `LlamaForCausalLM`)."""

    def _has_layer_stack(module: nn.Module) -> bool:
        return (
            isinstance(getattr(module, "layers", None), nn.ModuleList)
            or isinstance(getattr(module, "h", None), nn.ModuleList)
            or isinstance(getattr(module, "single_transformer_blocks", None), nn.ModuleList)
            or isinstance(getattr(module, "transformer_blocks", None), nn.ModuleList)
        )

    # Diffusers DiT / Flux models *are* the backbone (no nested `.model`).
    if _is_flux_like(model):
        return model

    for attr in ("model", "transformer", "base_model", "decoder"):
        backbone = getattr(model, attr, None)
        if not isinstance(backbone, nn.Module):
            continue
        # Some architectures nest the layer stack one level deeper (e.g. OPT: `model.decoder.layers`).
        nested = getattr(backbone, "decoder", None)
        if isinstance(nested, nn.Module) and _has_layer_stack(nested):
            return nested
        if _has_layer_stack(backbone):
            return backbone
        # Fall back to the first matching container even if layers are not yet recognizable; callers that need the
        # layer stack will raise a more specific error from `_get_decoder_layers`.
        return backbone
    if _has_layer_stack(model):
        return model
    raise AttributeError(
        "Unable to automatically locate the transformer backbone inside the supplied model "
        "(required when `target_modules` is not set). Please set `target_modules` explicitly "
        "to the decoder blocks you want to wrap, e.g. `r'.*\\.layers\\.\\d+$'`."
    )


def _get_decoder_layers(backbone: nn.Module) -> tuple[nn.ModuleList, str]:
    """Return `(layers, attr_name)` for the decoder-layer `nn.ModuleList` of a backbone."""
    # Prefer the longer contiguous Flux single-stream stack when both are present.
    for attr in ("layers", "h", "single_transformer_blocks", "transformer_blocks"):
        candidate = getattr(backbone, attr, None)
        if isinstance(candidate, nn.ModuleList):
            return candidate, attr
    raise AttributeError(
        "Unable to automatically find a `nn.ModuleList` of decoder layers (expected `.layers`/`.h`) "
        "inside the backbone (required when `target_modules` is not set). Please set `target_modules` "
        "explicitly to the decoder blocks you want to wrap, e.g. `r'.*\\.layers\\.\\d+$'`."
    )


def _get_hidden_size(config: Any) -> int:
    for attr in ("hidden_size", "n_embd", "d_model"):
        if hasattr(config, attr):
            return int(getattr(config, attr))
    # Diffusers Flux / DiT configs store width as heads * head_dim.
    if hasattr(config, "num_attention_heads") and hasattr(config, "attention_head_dim"):
        return int(config.num_attention_heads) * int(config.attention_head_dim)
    raise AttributeError("Unable to infer the hidden size from the model config.")


class _TokenShadowConfig(PretrainedConfig):
    """Configuration for the token-wise shadow backbone used by Flux-like models."""

    model_type = "shadow_token_backbone"

    def __init__(self, hidden_size: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.hidden_size = int(hidden_size)


class _TokenShadowBackbone(nn.Module):
    """A tiny token-wise MLP used as the ShadowPEFT mirror backbone on non-causal (e.g. Flux) models.

    Causal-LM ShadowPEFT mirrors the HF decoder class; Diffusers transformers do not share that constructor/config
    contract, so for those architectures we seed `s^(0)` with a small residual MLP over the entry hidden states.
    """

    def __init__(self, in_features: int, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__()
        self.config = _TokenShadowConfig(hidden_size=hidden_size)
        self.proj_in = nn.Linear(in_features, hidden_size)
        n_layers = max(1, int(num_layers))
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            raise ValueError("`_TokenShadowBackbone` requires `inputs_embeds`.")
        hidden = self.proj_in(inputs_embeds)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return type("TokenShadowOutput", (), {"last_hidden_state": hidden, "past_key_values": None})()


def _set_config_attr(config: Any, names: tuple[str, ...], value: int) -> None:
    for attr in names:
        if hasattr(config, attr):
            setattr(config, attr, int(value))
            return
    raise AttributeError(
        f"Unable to update the model config: none of the expected attributes {names} are defined. "
        'This is required when building a mirrored shadow backbone (`shadow_model="mirror"`). '
        "Please use a supported decoder architecture, or set `shadow_model` to a pretrained model "
        "id/path instead."
    )


def _normalize_layer_config(config: Any) -> Any:
    """Keep per-layer config fields consistent after reducing `num_hidden_layers`.

    When building a mirrored shadow backbone we deepcopy the base config and shrink `num_hidden_layers` (often to 1).
    Some architectures also store per-layer lists or layer-count-dependent fields (e.g. Qwen3 `layer_types`,
    `max_window_layers`) sized for the full base model. Leaving those unchanged would leave them longer than the new
    layer count and break construction, so truncate/pad them here.
    """
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
        del module.embed_tokens


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
    backbone produces an initial shadow state that rides the base decoder loop; at every targeted block the discrepancy
    between the base hidden states and the shadow state is injected into the block, and the shadow state is advanced by
    a gated residual update. Only the shadow components are trained. See [`ShadowConfig`] for the configuration.

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
            backbone = _get_backbone(self.model)
            layers, attr = _get_decoder_layers(backbone)
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
        """Idempotently create the model-level shadow containers and runtime bookkeeping.

        Primary creation happens in [`ShadowModel._pre_injection_hook`]. This helper is also called from
        `_create_and_replace` / `unload_shadow` so those paths remain safe if the containers are not yet present.
        """
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
            self._deferred_shadow_seed: bool = False

    def _pre_injection_hook(self, model: nn.Module, config: ShadowConfig, adapter_name: str) -> None:
        self._ensure_shadow_containers()

    def _check_new_adapter_config(self, config: ShadowConfig) -> None:
        super()._check_new_adapter_config(config)
        if len(self.peft_config) > 1 and any(
            peft_config.task_type == TaskType.SEQ_CLS for peft_config in self.peft_config.values()
        ):
            raise ValueError(
                "ShadowPEFT does not support multiple sequence classification adapters because each adapter requires "
                "both a classifier in `modules_to_save` and a separate `shadow_head`."
            )

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
            if isinstance(backbone, DetachedFluxShadowModel):
                # The compact Flux student is initialized from the leading channels of the base model. Start its
                # return projection as the matching zero-padded identity instead of destroying that structure with a
                # random projection.
                with torch.no_grad():
                    projection.weight.zero_()
                    projection.weight[:shadow_hidden, :shadow_hidden].copy_(
                        torch.eye(shadow_hidden, dtype=projection.weight.dtype, device=projection.weight.device)
                    )
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
        """Build a mirrored shadow backbone, pretrained-initialized for Flux2 and fresh for decoder models."""
        # Flux2 exposes a complete Diffusers config/constructor contract. Build a reduced model initialized from
        # evenly-spaced pretrained base blocks; unlike generic DiT fallbacks it can also reduce width while preserving
        # the latent/text I/O contract required by the pipeline.
        if _is_flux_like(self.model):
            base_hidden = self._base_hidden_size()
            base_double = getattr(self.model, "transformer_blocks", None)
            base_single = getattr(self.model, "single_transformer_blocks", None)
            if (
                config.shadow_model == "mirror"
                and isinstance(base_double, nn.ModuleList)
                and isinstance(base_single, nn.ModuleList)
                and hasattr(self.model.__class__, "from_config")
                and hasattr(self.model.config, "num_layers")
                and hasattr(self.model.config, "num_single_layers")
            ):
                num_layers = config.shadow_num_hidden_layers or 1
                num_single_layers = 2 * num_layers
                return DetachedFluxShadowModel.from_base(
                    self.model,
                    num_layers=num_layers,
                    num_single_layers=num_single_layers,
                    share_embeddings=config.share_embeddings,
                    hidden_size=config.shadow_hidden_size,
                    num_attention_heads=config.shadow_num_attention_heads,
                    intermediate_size=config.shadow_intermediate_size,
                )

            # Generic Diffusers/DiT fallback for architectures without a reconstructable Flux2 config.
            shadow_hidden = config.shadow_hidden_size or base_hidden
            return _TokenShadowBackbone(
                in_features=base_hidden,
                hidden_size=shadow_hidden,
                num_layers=config.shadow_num_hidden_layers or 1,
            )

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

        Returns `(backbone, projection)`. `projection` is `None` for a plain `AutoModel`, or a trained shadow_hidden ->
        base_hidden `nn.Linear` when loading a "projected" shadow checkpoint (`model_type ==
        'causal_lm_with_hidden_projection'`), which bundles a small backbone with a projection aligned to a larger
        base.
        """
        config_file = cached_file(config.shadow_model, "config.json")
        with open(config_file) as f:
            raw_config = json.load(f)
        if _is_flux_like(self.model) and raw_config.get("_class_name") == "Flux2Transformer2DModel":
            from diffusers import Flux2Transformer2DModel

            loaded = Flux2Transformer2DModel.from_pretrained(config.shadow_model)
            return validate_pretrained_flux_shadow(self.model, loaded), None
        if raw_config.get("model_type") == "causal_lm_with_hidden_projection":
            return self._load_projected_shadow_backbone(config, raw_config)
        return AutoModel.from_pretrained(config.shadow_model), None

    @staticmethod
    def _load_projected_shadow_backbone(config: ShadowConfig, raw_config: dict) -> tuple[nn.Module, nn.Module]:
        """Load the backbone + trained projection out of a `causal_lm_with_hidden_projection` checkpoint."""
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

        Returns `(head, head_is_lm)`. For causal LM, the base model's output head is reused at loss time. Users can
        train and save that head through PEFT's normal `modules_to_save=["lm_head"]` mechanism. The (small) classifier
        head is copied and trained by default.
        """
        if config.task_type == TaskType.CAUSAL_LM:
            return None, True
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
        self._register_boundary_hooks()

    def _register_boundary_hooks(self) -> None:
        # Drop any previously registered hooks so a re-bind (e.g. after adding/deleting an adapter) does not stack
        # duplicate callbacks on the same modules.
        for handle in self._boundary_hook_handles:
            handle.remove()
        self._boundary_hook_handles = []

        # Find the first/last ShadowLayer wrappers: they are the entry and exit of the shadow trajectory through the
        # base decoder. Then attach the four boundary hooks below.
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
        # Top-of-model pre-hook: reset per-forward bookkeeping, unpack a `ShadowCache` so the base model only sees its
        # own past, and compute the initial shadow state `s^(0)` from the raw inputs for the entry carrier.
        self._seed_shadow_state = None
        self._shadow_past_out = None
        self._should_pack_shadow_cache = False
        self._deferred_shadow_seed = False

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
        inputs_embeds = kwargs.get("inputs_embeds")
        if input_ids is None and inputs_embeds is None:
            # Custom decoder models may expose a differently named main input that already represents hidden states.
            main_input_name = getattr(self.model, "main_input_name", None)
            main_input = kwargs.get(main_input_name) if main_input_name else None
            if isinstance(main_input, torch.Tensor):
                if main_input_name == "input_ids":
                    input_ids = main_input
                else:
                    inputs_embeds = main_input
        # Diffusers DiT / Flux forwards pass pre-patch latents as `hidden_states`, not token ids / embeds. Seed
        # `s^(0)` later from the first wrapped block's hidden states (post-embed, matching width).
        if input_ids is None and inputs_embeds is None:
            self._deferred_shadow_seed = True
            self._should_pack_shadow_cache = False
            return args, kwargs

        self._seed_shadow_state, self._shadow_past_out = self._compute_initial_shadow_state(
            self.active_adapters[0],
            input_ids=input_ids,
            attention_mask=kwargs.get("attention_mask"),
            position_ids=kwargs.get("position_ids"),
            inputs_embeds=inputs_embeds,
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
        hidden = args[0] if args else kwargs.get("hidden_states")
        if hidden is None:
            return

        # Flux / DiT: the top-of-model hook could not seed from tokens, so build `s^(0)` from this block's input.
        # Recompute on every call instead of caching for the rest of the forward: under gradient checkpointing this
        # hook runs a second time during recomputation, and reusing the cached state would leave the shadow backbone
        # out of the recomputed graph (torch then rejects the mismatch in saved-tensor counts).
        if self._deferred_shadow_seed:
            backbone_kwargs = dict(kwargs)
            # Flux gradient checkpointing invokes single-stream blocks positionally as
            # (hidden_states, encoder_hidden_states, temb_mod, image_rotary_emb, joint_attention_kwargs).
            for name, value in zip(
                ("encoder_hidden_states", "temb_mod", "image_rotary_emb", "joint_attention_kwargs"),
                args[1:],
            ):
                backbone_kwargs.setdefault(name, value)
            self._seed_shadow_state, self._shadow_past_out = self._compute_initial_shadow_state(
                self.active_adapters[0],
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                inputs_embeds=hidden,
                past_key_values=None,
                use_cache=False,
                backbone_kwargs=backbone_kwargs,
            )

        if self._seed_shadow_state is None:
            return
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
        if isinstance(output, tuple) and output and isinstance(output[0], ShadowCarrier):
            return (output[0].hidden, *output[1:])
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
        backbone_kwargs: Optional[dict[str, Any]] = None,
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

        if isinstance(backbone, DetachedFluxShadowModel):
            out = backbone(inputs_embeds=inputs_embeds, block_kwargs=backbone_kwargs)
        elif share or inputs_embeds is not None:
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
        if isinstance(backbone, DetachedFluxShadowModel):
            freeze_flux_stem_embeddings(backbone)
            return
        for getter in ("get_input_embeddings", "get_output_embeddings"):
            embed = None
            with contextlib.suppress(Exception):
                embed = getattr(backbone, getter)()
            if isinstance(embed, nn.Module):
                embed.requires_grad_(False)

    def _sync_shadow_module_trainability(self, inference_mode: bool = False) -> None:
        """Make the model-level shadow modules' `requires_grad` match the active adapter.

        The backbones/projections/heads live on the tuner, not on `self.model`, so the generic `BaseTuner` machinery --
        which walks the base model looking for `ShadowLayer`s -- never reaches them. Both injection
        (`_mark_only_adapters_as_trainable`) and adapter switching (`set_adapter`) must call this, otherwise the
        previously active backbone would stay trainable while the newly activated one stays frozen.
        """
        self._ensure_shadow_containers()
        # Only the SEQ_CLS classifier is stored in `shadow_head`. The causal-LM head is managed by the base model and
        # PEFT's normal modules_to_save mechanism.
        for container in (self.shadow_backbone, self.shadow_projection, self.shadow_head):
            for adapter_name, module in container.items():
                module.requires_grad_(adapter_name in self.active_adapters and not inference_mode)
        for adapter_name, backbone in self.shadow_backbone.items():
            # For a pretrained shadow backbone, keep its embeddings frozen. (A "mirror" backbone either shares the
            # frozen base embeddings -- absent here -- or has randomly-initialized ones that must stay trainable.)
            if self.peft_config[adapter_name].shadow_model != "mirror":
                self._freeze_backbone_embeddings(backbone)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        super()._mark_only_adapters_as_trainable(model)
        self._sync_shadow_module_trainability()

    # ------------------------------------------------------------------------------------------- public forward

    def forward(self, *args: Any, **kwargs: Any):
        labels = kwargs.get("labels")
        attention_mask = kwargs.get("attention_mask")
        output = self.model(*args, **kwargs)

        # Compute the shadow path's own task loss (Eq. 8-9) on the standalone prediction head(s^(0)) -- `s^(0)` is
        # `_seed_shadow_state`, set by the seed pre-hook during the forward above.
        #
        # Training uses the *live* `shadow_loss` tensor: it is scaled by `auxiliary_loss_weight` and added into
        # `output.loss`, so autograd (and DDP/FSDP gradient sync on the shadow params) still see it.
        #
        # Separately, an unweighted *detached* copy is exposed for logging/inspection as `output.shadow_loss` and
        # `self.last_shadow_loss`. Detach keeps logging from retaining the graph or accidentally driving a second
        # backward. Storing it on the tuner matters because DDP/FSDP (and some Trainer paths) rebuild/replace the
        # model output from registered `ModelOutput` fields and drop ad-hoc attributes like `shadow_loss`; the module
        # attribute remains readable after the forward. This has been designed for that logging case -- there is no
        # dedicated DDP/FSDP integration test for the aux loss beyond ordinary `output.loss.backward()`.
        self.last_shadow_loss = None
        if labels is not None and getattr(output, "loss", None) is not None and self._shadow_path_active():
            shadow_loss = self.shadow_auxiliary_loss(labels, attention_mask=attention_mask)
            if shadow_loss is not None:
                # Logging copy only (no grad). The live `shadow_loss` below is what trains.
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

    def set_adapter(self, adapter_name: str | list[str], inference_mode: bool = False) -> None:
        adapter_names = [adapter_name] if isinstance(adapter_name, str) else adapter_name
        if len(adapter_names) != 1:
            raise ValueError(f"ShadowPEFT requires exactly one active adapter, but got {len(adapter_names)} adapters.")
        super().set_adapter(adapter_name, inference_mode=inference_mode)
        # `super()` only retargets the wrapped blocks; the tuner-level shadow modules need the same switch.
        self._sync_shadow_module_trainability(inference_mode=inference_mode)

    def _check_merge_allowed(self):
        raise NotImplementedError(
            "ShadowPEFT cannot be merged into the base model: the adaptation is an input-dependent, layer-space "
            "trajectory, not a static weight-space delta. Use `unload()` to recover the plain base model, or "
            "`unload_shadow()` for a standalone shadow model."
        )

    def _replace_module(self, parent: nn.Module, child_name: str, new_module: nn.Module, child: nn.Module) -> None:
        # Shadow wraps whole decoder blocks (not Linear), so there is no single `.weight` to rebind as in LoRA.
        setattr(parent, child_name, new_module)
        meta = torch.device("meta")
        for param in child.parameters():
            if param.device != meta:
                new_module.to(param.device)
                break

    def _unload_and_optionally_merge(
        self,
        merge: bool = True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        # Drop boundary hooks before unwrapping so they are not left dangling on removed modules.
        for handle in getattr(self, "_boundary_hook_handles", []):
            handle.remove()
        if hasattr(self, "_boundary_hook_handles"):
            self._boundary_hook_handles = []
            self._boundary_layers = []
        return super()._unload_and_optionally_merge(
            merge=merge, progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
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

    def unload_shadow(self, adapter_name: Optional[str] = None, copy: bool = False) -> nn.Module:
        """Return the shadow backbone (+ head) as a standalone model, *without* the base model.

        The ShadowPEFT analogue of `merge_and_unload`: where that would hand back the base model with the adaptation
        baked in, this hands back only the lightweight shadow network for high-efficiency / edge inference. It runs
        `head(projection(backbone(x)))` -- the per-block updates require the base outputs and so do not exist
        standalone. For language models, the result behaves like a normal causal LM (supports `generate()` and KV
        caching). For Flux-style Diffusers transformers, the result preserves the original transformer interface and
        uses a reduced-depth copy initialized from the pretrained stem, blocks, and output modules, so it can be
        assigned directly to the original pipeline.

        Args:
            adapter_name (`str`, *optional*):
                The adapter whose shadow network to unload. Defaults to the active adapter.
            copy (`bool`, *optional*, defaults to `False`):
                If `True`, deep-copy the returned model so it is independent of this one (uses more memory). If `False`
                (default), share modules -- similar to `merge_and_unload`, which reuses modules rather than cloning
                them. Mutating one model then affects the other.

        Assign the result to a variable and use it; with `copy=False` the modules remain shared with this model.

        Pass `copy=True` when you intend to `save_pretrained` the standalone model. A shadow backbone that shares the
        frozen base input embeddings (the default `"mirror"` setup) reaches them through a reference that is not a
        submodule of the returned model, so a `copy=False` checkpoint is missing the embedding table; `copy=True`
        re-attaches a private copy and saves a complete checkpoint.
        """
        self._ensure_shadow_containers()
        if adapter_name is None:
            adapter_name = self.active_adapters[0]
        if adapter_name not in self.shadow_backbone:
            raise ValueError(f"No shadow backbone found for adapter '{adapter_name}'.")

        # Only Flux-style models with a single-stream stack can be reconstructed as a pipeline-compatible detached
        # model. Other Diffusers backbones retain the existing generic detached-shadow behavior.
        if isinstance(getattr(self.model, "single_transformer_blocks", None), nn.ModuleList):
            return build_detached_flux_shadow(
                self.model,
                self.shadow_backbone[adapter_name],
                self.shadow_projection[adapter_name],
                copy=copy,
            )

        maybe_copy = deepcopy if copy else (lambda module: module)
        backbone = maybe_copy(self.shadow_backbone[adapter_name])
        projection = maybe_copy(self.shadow_projection[adapter_name])
        head = self._resolve_shadow_head(adapter_name)
        if head is not None:
            head = maybe_copy(head)

        # If the backbone shared the frozen base embeddings (its own `embed_tokens` was removed), restore access so the
        # detached model can run from `input_ids`. With `copy=True`, re-attach a private copy on the backbone. With
        # `copy=False`, pass a shared reference into `DetachedShadowModel` without re-parenting the module.
        shared_input_embeddings = None
        if self._shadow_share_embeddings.get(adapter_name, False):
            embeds = self.model.get_input_embeddings()
            if copy:
                backbone.embed_tokens = deepcopy(embeds)
            else:
                shared_input_embeddings = embeds

        is_classification = self.peft_config[adapter_name].task_type == TaskType.SEQ_CLS
        return DetachedShadowModel(
            backbone,
            projection,
            head,
            is_classification=is_classification,
            input_embeddings=shared_input_embeddings,
        )
