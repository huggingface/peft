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
from transformers import GenerationConfig, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, SequenceClassifierOutput

from peft.tuners.tuners_utils import BaseTunerLayer

from .config import ShadowConfig


class ShadowCarrier:
    """Couples the base `hidden_states` with the parallel `shadow_states` so the pair rides the base decoder loop."""

    def __init__(self, hidden: torch.Tensor, shadow: torch.Tensor) -> None:
        self.hidden = hidden
        self.shadow = shadow


class ShadowCache:
    """Paired KV caches for incremental ShadowPEFT decoding.

    Autoregressive generation needs a base-model cache (keys/values computed under shadow injection) *and* a separate
    shadow-backbone cache (to advance `s^(0)` token-by-token). Generation only inspects Cache-like methods on the object
    stored as `past_key_values` (seq length, reorder, crop); those delegate to the base half. The shadow half is unpacked
    before the base forward and re-packed on the way out -- see [`ShadowModel`] hooks.
    """

    __slots__ = ("base", "shadow")

    def __init__(self, base: Any = None, shadow: Any = None) -> None:
        self.base = base
        self.shadow = shadow

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if self.base is None:
            return 0
        return self.base.get_seq_length(layer_idx)

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        if self.base is None:
            return -1
        return self.base.get_max_cache_shape(layer_idx)

    @property
    def is_compileable(self) -> bool:
        # Dual-cache decoding is not a single compileable Cache layout.
        return False

    def has_previous_state(self, layer_idx: Optional[int] = None) -> bool:
        if self.base is None:
            return False
        return self.base.has_previous_state(layer_idx)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.base is not None and hasattr(self.base, "reorder_cache"):
            self.base.reorder_cache(beam_idx)
        if self.shadow is not None and hasattr(self.shadow, "reorder_cache"):
            self.shadow.reorder_cache(beam_idx)

    def crop(self, max_length: int) -> None:
        if self.base is not None and hasattr(self.base, "crop"):
            self.base.crop(max_length)
        if self.shadow is not None and hasattr(self.shadow, "crop"):
            self.shadow.crop(max_length)

    def __repr__(self) -> str:
        return f"ShadowCache(base={self.base!r}, shadow={self.shadow!r})"


class ShadowLayer(nn.Module, BaseTunerLayer):
    # Trainable, per-adapter sub-modules (`shadow_dropout` is parameter-free but listed here so its per-adapter entry
    # is created/deleted alongside the others).
    adapter_layer_names = (
        "shadow_down",
        "shadow_up",
        "shadow_update_transform",
        "shadow_update_gate",
        "shadow_dropout",
    )
    # Non-tensor / non-trainable per-adapter bookkeeping.
    other_param_names = ("shadow_r", "shadow_alpha")

    def __init__(
        self, base_layer: nn.Module, adapter_name: str, config: ShadowConfig, hidden_size: int, **kwargs
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        # `d` is the base block's hidden size. The shadow state always lives at this width so that the discrepancy
        # `h - s` in Eq. 2 is well defined; the shadow *backbone* may use a different width internally and project.
        self.hidden_size = int(hidden_size)

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
        return self.hidden_size

    def forward(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
        # `hidden_states` is a ShadowCarrier exactly when the shadow path is active for this pass (the entry pre-hook
        # wrapped it). Otherwise -- adapters disabled, or this block not adapted for the active adapter -- it is a plain
        # tensor and we behave exactly like the frozen block.
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
        return ShadowCarrier(new_hidden, new_shadow)

    @staticmethod
    def _block_output_hidden_states(out: Any) -> torch.Tensor:
        """Pull the hidden-states tensor out of the block's return value (bare tensor / tuple / structured output)."""
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, tuple):
            return out[0]
        if getattr(out, "hidden_states", None) is not None:
            return out.hidden_states
        if getattr(out, "last_hidden_state", None) is not None:
            return out.last_hidden_state
        raise TypeError(f"Unsupported decoder block output type: {type(out)}. Expected a Tensor or a tuple.")

    def shadow_inject(
        self, hidden_states: torch.Tensor, shadow_states: torch.Tensor, adapter_name: str
    ) -> torch.Tensor:
        # Inject the shadow state into the block input (Section 3.2, Eq. 2-4).
        # delta       = h_out^(l-1) - s^(l-1)              (Eq. 2)
        # delta_tilde = Dropout(delta . W_down) . W_up     (Eq. 3)
        # h^(l)       = h_out^(l-1) + alpha * delta_tilde  (Eq. 4)
        # The adapter weights may live in a different (e.g. fp32) dtype than the base hidden states (e.g. bf16) when
        # `autocast_adapter_dtype` is enabled; cast around the low-rank projection and add the correction back in the
        # base dtype.
        down = self.shadow_down[adapter_name]
        adapter_dtype = down.weight.dtype
        delta = (hidden_states - shadow_states).to(adapter_dtype)  # Eq. 2
        delta = self.shadow_dropout[adapter_name](delta)
        delta = self.shadow_up[adapter_name](down(delta))  # Eq. 3
        return hidden_states + (self.shadow_alpha[adapter_name] * delta).to(hidden_states.dtype)  # Eq. 4

    def shadow_update(
        self, hidden_states: torch.Tensor, shadow_states: torch.Tensor, adapter_name: str
    ) -> torch.Tensor:
        # Advance the shadow state from the block output (Section 3.3, Eq. 5-7).
        # t^(l) = T^(l)(h_out^(l))                       (Eq. 5)
        # g^(l) = sigmoid(G^(l)(h_out^(l)))              (Eq. 6)
        # s^(l) = (1 - g^(l)) * s^(l-1) + g^(l) * t^(l)  (Eq. 7)
        transform = self.shadow_update_transform[adapter_name]
        adapter_dtype = transform[0].weight.dtype
        shadow_dtype = shadow_states.dtype
        hidden = hidden_states.to(adapter_dtype)
        candidate = transform(hidden)  # Eq. 5
        gate = torch.sigmoid(self.shadow_update_gate[adapter_name](hidden))  # Eq. 6
        new_shadow = (1.0 - gate) * shadow_states.to(adapter_dtype) + gate * candidate  # Eq. 7
        return new_shadow.to(shadow_dtype)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError(
            "ShadowPEFT cannot be merged into the base weights: the adaptation is an input-dependent, layer-space "
            "trajectory (the shadow state evolves with the data), not a static weight-space delta. Use "
            "`unload_shadow()` to obtain a standalone shadow model instead."
        )

    def unmerge(self) -> None:
        raise NotImplementedError("ShadowPEFT does not support merging, so there is nothing to unmerge.")

    def __getattr__(self, name: str):
        # The base model's forward reads attributes off the decoder block it is iterating over (e.g. Qwen3 reads
        # `decoder_layer.attention_type` to pick the right causal mask). Since this layer *wraps* the block, delegate
        # any attribute not found on the wrapper itself to the wrapped `base_layer`.
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "base_layer":
                raise
            return getattr(self.base_layer, name)

    def __repr__(self) -> str:
        return "shadow." + super().__repr__()


class DetachedShadowModel(PreTrainedModel, GenerationMixin):
    """A standalone shadow network (backbone + projection + head), detached from the base model.

    Returned by [`ShadowModel.unload_shadow`]. It runs `head(projection(backbone(x)))` -- the per-block shadow updates
    depend on the base model's outputs and so do not exist standalone. This is the lightweight component that can be
    deployed on its own for high-efficiency / edge inference (Section 3.5 of the paper).

    Because it is just the shadow backbone with a task head, it behaves like a normal task model. For a causal-LM head
    it returns a [`~transformers.modeling_outputs.CausalLMOutputWithPast`] and supports `generate()` and KV caching.
    For a sequence-classification head it pools the last token and returns a
    [`~transformers.modeling_outputs.SequenceClassifierOutput`]. This is how you evaluate the shadow path's own
    performance, independent of the base model.
    """

    main_input_name = "input_ids"
    base_model_prefix = "backbone"
    _supports_cache_class = True
    # The wrapped backbone implements attention; advertise support so `PreTrainedModel` init doesn't reject the
    # inherited attention implementation of the wrapper itself.
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    # CausalLM-specific kwargs that `generate` may pass but the backbone (a base model) does not accept.
    _causal_lm_only_kwargs = ("logits_to_keep", "num_logits_to_keep")

    def __init__(
        self,
        backbone: PreTrainedModel,
        projection: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        is_classification: bool = False,
    ) -> None:
        super().__init__(backbone.config)
        self.backbone = backbone
        self.shadow_hidden_projection = projection if projection is not None else nn.Identity()
        self.head = head
        self.is_classification = is_classification
        base_generation_config = getattr(backbone, "generation_config", None)
        if base_generation_config is not None:
            self.generation_config = base_generation_config
        else:
            self.generation_config = GenerationConfig.from_model_config(self.config)

    def can_generate(self) -> bool:
        # `PreTrainedModel.__init__` calls this before the attributes are assigned, so read them defensively. A
        # classification head is not autoregressive, so it cannot generate.
        return getattr(self, "head", None) is not None and not getattr(self, "is_classification", False)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone.set_input_embeddings(value)

    def get_output_embeddings(self):
        return None if getattr(self, "is_classification", False) else self.head

    @staticmethod
    def _pool_last_token(hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Pool the last non-padding token representation (used for sequence classification)."""
        if attention_mask is None:
            return hidden[:, -1, :]
        token_counts = (attention_mask.long().sum(dim=1) - 1).clamp(min=0)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, token_counts]

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        for key in self._causal_lm_only_kwargs:
            kwargs.pop(key, None)
        kwargs["return_dict"] = True
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden = self.shadow_hidden_projection(out.last_hidden_state)
        if self.head is None:
            return hidden

        if self.is_classification:
            pooled = self._pool_last_token(hidden, attention_mask)
            logits = self.head(pooled)
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(logits, labels.view(-1))
            return SequenceClassifierOutput(loss=loss, logits=logits)

        logits = self.head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=getattr(out, "past_key_values", None))
