# Copyright 2024-present the HuggingFace Inc. team.
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
import weakref
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationConfig, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, SequenceClassifierOutput

from peft.utils import TaskType

from .config import ShadowConfig
from .model_utils import (
    build_implicit_shadow_model,
    count_parameters,
    extract_backbone_model,
    get_backbone,
    get_decoder_layers,
    get_hidden_size,
    normalize_layer_config,
    remove_embed_tokens,
)
from .modules import ShadowInjectionModel, ShadowUpdateModel
from .projected_causal_lm import AutoModelForCausalLMWithHiddenProjection


@dataclass
class ShadowCausalLMOutputWithPast(CausalLMOutputWithPast):
    shadow_logits: Optional[torch.Tensor] = None
    shadow_loss: Optional[torch.Tensor] = None


@dataclass
class ShadowSequenceClassifierOutput(SequenceClassifierOutput):
    shadow_logits: Optional[torch.Tensor] = None
    shadow_loss: Optional[torch.Tensor] = None


def _shifted_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _pool_last_token(hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Pool the last non-padding token representation (used for sequence classification)."""
    if attention_mask is None:
        return hidden[:, -1, :]
    token_counts = (attention_mask.long().sum(dim=1) - 1).clamp(min=0)
    batch_idx = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch_idx, token_counts]


class _ShadowLayerWrapper(nn.Module):
    """
    Wrap a single decoder layer to apply Shadow injection (before the layer) and update (after the layer).

    Architecture-agnostic: the layer's first positional arg (or ``hidden_states`` kwarg) must be the hidden state
    tensor, and the layer must return either a Tensor or a tuple whose first element is the hidden states.
    """

    def __init__(self, layer: nn.Module, layer_idx: int, adapter: "ShadowModel") -> None:
        super().__init__()
        self.layer = layer
        self.layer_idx = int(layer_idx)
        # Don't register `adapter` as a submodule (would create a reference cycle in repr/state_dict).
        object.__setattr__(self, "_adapter_ref", weakref.ref(adapter))

    def _get_adapter(self) -> "ShadowModel":
        adapter = self._adapter_ref()
        if adapter is None:
            raise RuntimeError("Shadow adapter reference is gone.")
        return adapter

    def __getattr__(self, name: str):
        try:
            return nn.Module.__getattr__(self, name)
        except AttributeError:
            layer = nn.Module.__getattr__(self, "layer")
            return getattr(layer, name)

    def forward(self, *args, **kwargs):
        adapter = self._get_adapter()

        if "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
            use_kw = True
        elif len(args) > 0:
            hidden_states = args[0]
            use_kw = False
        else:
            raise TypeError("Decoder layer wrapper could not find hidden_states in args/kwargs.")

        # Apply injection before the layer (no-op for layer 0, mirroring the reference implementation).
        if adapter._adapters_enabled and self.layer_idx > 0:
            shadow = adapter._shadow_hidden_states
            if shadow is None:
                raise RuntimeError(
                    "Shadow state was not initialized. Call the Shadow model's forward(), not the base model directly."
                )
            sidx = self.layer_idx - 1
            hidden_states = adapter.shadow_injection_model(hidden_states, shadow, sidx)
            if use_kw:
                kwargs["hidden_states"] = hidden_states
            else:
                args = (hidden_states,) + args[1:]

        out = self.layer(*args, **kwargs)

        if isinstance(out, torch.Tensor):
            hs_out = out
            rest = None
        elif isinstance(out, tuple):
            hs_out = out[0]
            rest = out[1:]
        else:
            if hasattr(out, "hidden_states") and out.hidden_states is not None:
                hs_out = out.hidden_states
            elif hasattr(out, "last_hidden_state"):
                hs_out = out.last_hidden_state
            else:
                raise TypeError(f"Unsupported decoder layer output type: {type(out)}. Expected Tensor/tuple.")
            rest = None

        # Update shadow state after the layer.
        if adapter._adapters_enabled and self.layer_idx > 0:
            shadow = adapter._shadow_hidden_states
            sidx = self.layer_idx - 1
            adapter._shadow_hidden_states = adapter.shadow_update_model(hs_out, shadow, sidx)

        if rest is None:
            return out
        return (hs_out,) + rest


class ShadowModel(nn.Module, GenerationMixin):
    """
    PEFT tuner that augments a frozen base decoder-only model with a lightweight, trainable parallel *shadow* network
    (ShadowPEFT). The base model's decoder layers are wrapped to inject learned corrections, and a small shadow backbone
    evolves a parallel hidden state that conditions those corrections.

    This class is instantiated by [`PeftModel`]; users typically create it via [`get_peft_model`]. An optional
    pre-trained explicit shadow model can be supplied via ``get_peft_model(model, config, shadow_model=...)``.
    """

    prefix = "shadow_"

    main_input_name = "input_ids"
    _is_stateful: bool = False

    def __init__(self, model: PreTrainedModel, config: dict, adapter_name: str) -> None:
        super().__init__()
        self.model = model
        self.peft_config = config
        self.active_adapter = adapter_name
        self._adapters_enabled = True

        shadow_config: ShadowConfig = config[adapter_name]
        explicit_shadow_model = getattr(shadow_config, "shadow_model", None)

        # Generation/HF-compatibility plumbing.
        self.config = getattr(self.model, "config", None)
        if self.config is not None and hasattr(self.config, "use_cache"):
            self.config.use_cache = False
        base_gen_cfg = getattr(self.model, "generation_config", None)
        if base_gen_cfg is not None:
            self.generation_config = base_gen_cfg
        elif self.config is not None:
            self.generation_config = GenerationConfig.from_model_config(self.config)
        else:
            self.generation_config = GenerationConfig()
        if hasattr(self.generation_config, "use_cache"):
            self.generation_config.use_cache = False

        self.shadow_inference_mode = shadow_config.shadow_inference_mode
        self.shadow_loss_weight = float(shadow_config.shadow_loss_weight)
        self.task_type = shadow_config.task_type
        self._explicit_shadow_requires_grad_state = self._get_explicit_shadow_requires_grad_state(
            explicit_shadow_model
        )

        # Freeze the base model.
        for p in self.model.parameters():
            p.requires_grad = False

        base_backbone, base_layers, base_layers_attr = get_decoder_layers(self.model)
        num_base_layers = len(base_layers)
        if num_base_layers < 2:
            raise ValueError("Shadow requires at least 2 decoder layers to apply injection.")
        num_adapt_layers = num_base_layers - 1
        hidden_size = get_hidden_size(self.model)
        self.base_hidden_size = int(hidden_size)

        self._build_shadow_backbone(shadow_config, explicit_shadow_model, hidden_size)

        # Adapter modules (always trainable).
        self.shadow_injection_model = ShadowInjectionModel(
            num_layers=num_adapt_layers,
            hidden_size=hidden_size,
            injection_hidden_size=shadow_config.injection_hidden_size,
            dropout=shadow_config.dropout,
            alpha=shadow_config.alpha,
        )
        self.shadow_update_model = ShadowUpdateModel(
            num_layers=num_adapt_layers,
            hidden_size=hidden_size,
            gate_hidden_size=shadow_config.gate_hidden_size,
            dropout=shadow_config.dropout,
        )

        # Task-specific shadow head (causal LM / sequence classification).
        self._build_shadow_head(shadow_config)

        # Mutable per-forward state read by the wrapped layers.
        self._shadow_hidden_states: Optional[torch.Tensor] = None

        # Wrap base decoder layers in-place.
        wrapped = nn.ModuleList([])
        for i, layer in enumerate(base_layers):
            if isinstance(layer, _ShadowLayerWrapper) and layer._adapter_ref() is self:
                wrapped.append(layer)
            else:
                wrapped.append(_ShadowLayerWrapper(layer, layer_idx=i, adapter=self))
        setattr(base_backbone, base_layers_attr, wrapped)

        self._mark_trainable(shadow_config)
        self._restore_explicit_shadow_requires_grad_state()

        # Match the shadow modules to the base model's device and dtype. The shadow backbone, injection/update adapters
        # and projection are created in the default float dtype, while the base model is often loaded in fp16/bf16
        # (e.g. `from_pretrained` keeps the checkpoint dtype). Without this, injected hidden states would be upcast and
        # mismatch the base model's parameter dtype.
        base_param = next(self.model.parameters(), None)
        if base_param is not None:
            for module in (
                self.shadow_model,
                self.shadow_injection_model,
                self.shadow_update_model,
                self.shadow_hidden_projection,
                self.shadow_lm_head,
                self.shadow_classifier_head,
            ):
                if module is not None:
                    module.to(device=base_param.device, dtype=base_param.dtype)

    # ------------------------------------------------------------------ build helpers

    @staticmethod
    def _get_module_requires_grad_state(module: Optional[nn.Module]) -> dict[str, bool]:
        if module is None:
            return {}
        return {name: param.requires_grad for name, param in module.named_parameters()}

    @classmethod
    def _get_explicit_shadow_requires_grad_state(cls, explicit_shadow_model) -> dict[str, dict[str, bool]]:
        """Record explicit-shadow parameter trainability before ShadowPEFT enables adapter training."""
        if explicit_shadow_model is None:
            return {}

        get_output_embeddings = getattr(explicit_shadow_model, "get_output_embeddings", None)
        if callable(get_output_embeddings):
            shadow_head = get_output_embeddings()
        else:
            shadow_head = getattr(explicit_shadow_model, "lm_head", None)
        shadow_projection = getattr(explicit_shadow_model, "shadow_hidden_projection", None)

        if isinstance(explicit_shadow_model, AutoModelForCausalLMWithHiddenProjection):
            shadow_backbone = explicit_shadow_model.shadow_model
            shadow_head = getattr(explicit_shadow_model, "lm_head", shadow_head)
            shadow_projection = getattr(explicit_shadow_model, "shadow_hidden_projection", shadow_projection)
        else:
            shadow_backbone = extract_backbone_model(explicit_shadow_model)

        return {
            "shadow_model": cls._get_module_requires_grad_state(shadow_backbone),
            "shadow_hidden_projection": cls._get_module_requires_grad_state(shadow_projection),
            "shadow_lm_head": cls._get_module_requires_grad_state(shadow_head),
        }

    @staticmethod
    def _restore_module_requires_grad_state(module: Optional[nn.Module], state: dict[str, bool]) -> None:
        if module is None or not state:
            return
        for name, param in module.named_parameters():
            if name not in state:
                continue
            requires_grad = state[name]
            param.requires_grad = requires_grad

    def _restore_explicit_shadow_requires_grad_state(self) -> None:
        """Preserve explicit shadow-model trainability after enabling ShadowPEFT adapter modules."""
        if not self._explicit_shadow_requires_grad_state:
            return

        self._restore_module_requires_grad_state(
            self.shadow_model,
            self._explicit_shadow_requires_grad_state.get("shadow_model", {}),
        )
        self._restore_module_requires_grad_state(
            self.shadow_hidden_projection,
            self._explicit_shadow_requires_grad_state.get("shadow_hidden_projection", {}),
        )
        self._restore_module_requires_grad_state(
            self.shadow_lm_head,
            self._explicit_shadow_requires_grad_state.get("shadow_lm_head", {}),
        )

    def _build_shadow_backbone(self, shadow_config: ShadowConfig, explicit_shadow_model, hidden_size: int) -> None:
        self._explicit_shadow_model = explicit_shadow_model is not None
        extracted_shadow_projection = None

        if explicit_shadow_model is None:
            self.shadow_model = build_implicit_shadow_model(
                self.model,
                num_shadow_layers=shadow_config.num_shadow_layers,
                shadow_intermediate_size=shadow_config.shadow_intermediate_size,
                shadow_num_attention_heads=shadow_config.shadow_num_attention_heads,
                shadow_num_key_value_heads=shadow_config.shadow_num_key_value_heads,
                shadow_head_dim=shadow_config.shadow_head_dim,
            )
        else:
            if isinstance(explicit_shadow_model, AutoModelForCausalLMWithHiddenProjection):
                if isinstance(getattr(explicit_shadow_model, "shadow_hidden_projection", None), nn.Linear):
                    extracted_shadow_projection = deepcopy(explicit_shadow_model.shadow_hidden_projection)
                explicit_shadow_model = explicit_shadow_model.shadow_model
            else:
                if isinstance(getattr(explicit_shadow_model, "shadow_hidden_projection", None), nn.Linear):
                    extracted_shadow_projection = deepcopy(explicit_shadow_model.shadow_hidden_projection)
                    try:
                        delattr(explicit_shadow_model, "shadow_hidden_projection")
                    except Exception:
                        explicit_shadow_model._modules.pop("shadow_hidden_projection", None)
            self.shadow_model = extract_backbone_model(explicit_shadow_model)

        # Auto-enable embedding sharing for explicit shadow models that have no input embeddings.
        self._explicit_share_base_embeddings = False
        if self._explicit_shadow_model:
            with contextlib.suppress(Exception):
                get_inp = getattr(self.shadow_model, "get_input_embeddings", None)
                if callable(get_inp) and get_inp() is None:
                    self._explicit_share_base_embeddings = True

        shadow_hidden_size = get_hidden_size(self.shadow_model)
        self.shadow_hidden_size = int(shadow_hidden_size)

        if int(shadow_hidden_size) != int(hidden_size):
            cand = extracted_shadow_projection
            if cand is None:
                c = getattr(self.shadow_model, "shadow_hidden_projection", None)
                if isinstance(c, nn.Linear):
                    cand = c
            if (
                cand is not None
                and cand.in_features == int(shadow_hidden_size)
                and cand.out_features == int(hidden_size)
            ):
                self.shadow_hidden_projection = deepcopy(cand)
            else:
                self.shadow_hidden_projection = nn.Linear(int(shadow_hidden_size), int(hidden_size), bias=False)
        else:
            self.shadow_hidden_projection = nn.Identity()

        # Configure embedding sharing (feed base inputs_embeds into the shadow backbone).
        self._shadow_supports_inputs_embeds = self._configure_shadow_embedding_sharing()

    def _configure_shadow_embedding_sharing(self) -> bool:
        shadow_backbone = get_backbone(self.shadow_model)
        try:
            sig = inspect.signature(shadow_backbone.forward)
            supports_inputs_embeds = "inputs_embeds" in sig.parameters
        except (TypeError, ValueError):
            supports_inputs_embeds = False

        share = (not self._explicit_shadow_model) or self._explicit_share_base_embeddings
        if supports_inputs_embeds and share:
            remove_embed_tokens(self.shadow_model)
            if shadow_backbone is not self.shadow_model:
                remove_embed_tokens(shadow_backbone)
        return supports_inputs_embeds

    def _build_shadow_head(self, shadow_config: ShadowConfig) -> None:
        self.shadow_lm_head = None
        self.shadow_classifier_head = None
        if self.task_type == TaskType.CAUSAL_LM:
            base_head = self.model.get_output_embeddings()
            if base_head is None:
                raise AttributeError("Base model does not expose output embeddings/lm_head for the shadow head.")
            self.shadow_lm_head = deepcopy(base_head)
        elif self.task_type == TaskType.SEQ_CLS:
            head = None
            for attr in ("score", "classifier"):
                cand = getattr(self.model, attr, None)
                if isinstance(cand, nn.Module):
                    head = cand
                    break
            if head is None:
                raise AttributeError(
                    "Base model does not expose a classifier head (`score` or `classifier`) for the shadow head."
                )
            self.shadow_classifier_head = deepcopy(head)

    def _mark_trainable(self, shadow_config: ShadowConfig) -> None:
        # Base model already frozen; enable shadow adapter modules.
        for module in (self.shadow_model, self.shadow_injection_model, self.shadow_update_model):
            for p in module.parameters():
                p.requires_grad = True
        for p in self.shadow_hidden_projection.parameters():
            p.requires_grad = True

        requested = set(shadow_config.modules_to_save or [])

        # Shadow task head trainability is controlled independently of `modules_to_save` (which carries base-model
        # module names, e.g. the classifier auto-added by PeftModelForSequenceClassification):
        # - shadow classifier head (SEQ_CLS): trainable by default (small, task-specific head).
        # - shadow lm head (CAUSAL_LM): frozen by default (a copy of the large base head); opt in via
        #   `modules_to_save=["shadow_lm_head"]`.
        if self.shadow_classifier_head is not None:
            for p in self.shadow_classifier_head.parameters():
                p.requires_grad = True
        if self.shadow_lm_head is not None:
            train_head = "shadow_lm_head" in requested
            for p in self.shadow_lm_head.parameters():
                p.requires_grad = train_head

        # Base-model modules requested in modules_to_save (e.g. lm_head). Sequence/token-classification heads
        # (score / classifier) are handled by PeftModel's ModulesToSaveWrapper, so we skip them here.
        base_requested = requested - {"shadow_lm_head", "shadow_classifier_head", "classifier", "score"}
        if base_requested:
            for name, module in self.model.named_modules():
                leaf = name.split(".")[-1]
                if leaf in base_requested:
                    for p in module.parameters():
                        p.requires_grad = True

    # ------------------------------------------------------------------ shadow forward internals

    def _compute_initial_shadow_hidden(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        shadow_backbone = get_backbone(self.shadow_model)

        share_base = (not self._explicit_shadow_model) or self._explicit_share_base_embeddings
        if self._shadow_supports_inputs_embeds and inputs_embeds is None and share_base:
            if input_ids is None:
                raise ValueError("Either `input_ids` or `inputs_embeds` must be provided.")
            base_embed = self.model.get_input_embeddings()
            if base_embed is None:
                raise AttributeError("Base model does not expose input embeddings.")
            inputs_embeds = base_embed(input_ids)

        kwargs = {"use_cache": False, "past_key_values": None, "output_hidden_states": False, "return_dict": True}
        if self._shadow_supports_inputs_embeds and inputs_embeds is not None:
            out = shadow_backbone(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, **kwargs
            )
        else:
            out = shadow_backbone(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, **kwargs
            )

        if not hasattr(out, "last_hidden_state"):
            raise TypeError(
                "Shadow backbone did not return an object with `last_hidden_state`; architecture is unsupported."
            )
        shadow_hidden = out.last_hidden_state
        return self.shadow_hidden_projection(shadow_hidden)

    # ------------------------------------------------------------------ public forward

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        # When adapters are disabled, run the plain base model.
        if not self._adapters_enabled:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                **kwargs,
            )

        if self.shadow_inference_mode == "shadow_only":
            return self._forward_shadow_only(input_ids, attention_mask, position_ids, inputs_embeds, labels)

        # base_shadow: run base model (with injection) and also compute the shadow path.
        kwargs["use_cache"] = False
        kwargs["past_key_values"] = None
        kwargs.setdefault("return_dict", True)

        self._shadow_hidden_states = self._compute_initial_shadow_hidden(
            input_ids, attention_mask, position_ids, inputs_embeds
        )
        initial_shadow_hidden = self._shadow_hidden_states
        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                **kwargs,
            )
        finally:
            self._shadow_hidden_states = None

        base_logits = getattr(outputs, "logits", None)
        if base_logits is None:
            raise TypeError("Base model output is missing `logits`.")
        loss = getattr(outputs, "loss", None)

        if self.task_type == TaskType.SEQ_CLS and self.shadow_classifier_head is not None:
            pooled = _pool_last_token(initial_shadow_hidden, attention_mask)
            shadow_logits = self.shadow_classifier_head(pooled)
            shadow_loss = None
            if labels is not None:
                if loss is None:
                    loss = F.cross_entropy(base_logits, labels)
                shadow_loss = F.cross_entropy(shadow_logits, labels)
                if self.shadow_loss_weight > 0:
                    loss = loss + self.shadow_loss_weight * shadow_loss
            return ShadowSequenceClassifierOutput(
                loss=loss,
                logits=base_logits,
                shadow_logits=shadow_logits,
                shadow_loss=shadow_loss,
                hidden_states=getattr(outputs, "hidden_states", None),
                attentions=getattr(outputs, "attentions", None),
            )

        if self.shadow_lm_head is not None:
            shadow_logits = self.shadow_lm_head(initial_shadow_hidden)
            shadow_loss = None
            if labels is not None:
                if loss is None:
                    loss = _shifted_ce_loss(base_logits, labels)
                shadow_loss = _shifted_ce_loss(shadow_logits, labels)
                if self.shadow_loss_weight > 0:
                    loss = loss + self.shadow_loss_weight * shadow_loss
            return ShadowCausalLMOutputWithPast(
                loss=loss,
                logits=base_logits,
                shadow_logits=shadow_logits,
                shadow_loss=shadow_loss,
                past_key_values=None,
                hidden_states=getattr(outputs, "hidden_states", None),
                attentions=getattr(outputs, "attentions", None),
            )

        # No shadow head (e.g. feature extraction / token classification): base path only.
        return outputs

    def _forward_shadow_only(self, input_ids, attention_mask, position_ids, inputs_embeds, labels):
        shadow_hidden = self._compute_initial_shadow_hidden(input_ids, attention_mask, position_ids, inputs_embeds)
        if self.task_type == TaskType.SEQ_CLS and self.shadow_classifier_head is not None:
            pooled = _pool_last_token(shadow_hidden, attention_mask)
            shadow_logits = self.shadow_classifier_head(pooled)
            loss = F.cross_entropy(shadow_logits, labels) if labels is not None else None
            return ShadowSequenceClassifierOutput(
                loss=loss, logits=shadow_logits, shadow_logits=shadow_logits, shadow_loss=loss
            )
        if self.shadow_lm_head is None:
            raise ValueError("shadow_only inference requires a shadow task head (CAUSAL_LM or SEQ_CLS task type).")
        shadow_logits = self.shadow_lm_head(shadow_hidden)
        loss = _shifted_ce_loss(shadow_logits, labels) if labels is not None else None
        return ShadowCausalLMOutputWithPast(
            loss=loss, logits=shadow_logits, shadow_logits=shadow_logits, shadow_loss=loss, past_key_values=None
        )

    # ------------------------------------------------------------------ generation hooks

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        position_ids = kwargs.get("position_ids")
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": False,
            "past_key_values": None,
        }

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, **kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder, **kwargs
        )
        model_kwargs["past_key_values"] = None
        model_kwargs["use_cache"] = False
        return model_kwargs

    # ------------------------------------------------------------------ PEFT tuner interface

    @property
    def active_adapters(self) -> list[str]:
        return [self.active_adapter]

    def set_inference_mode(self, mode: str) -> None:
        if mode not in ("base_shadow", "shadow_only"):
            raise ValueError(f"mode must be 'base_shadow' or 'shadow_only', got {mode!r}.")
        self.shadow_inference_mode = mode

    def enable_adapter_layers(self) -> None:
        self._adapters_enabled = True

    def disable_adapter_layers(self) -> None:
        self._adapters_enabled = False

    def add_adapter(self, adapter_name: str, config: ShadowConfig) -> None:
        if adapter_name in self.peft_config and adapter_name != self.active_adapter:
            raise ValueError(f"Adapter with name '{adapter_name}' already exists.")
        if adapter_name != self.active_adapter:
            raise NotImplementedError(
                "ShadowModel currently supports a single adapter. Create a new PEFT model for a different adapter."
            )
        self.peft_config[adapter_name] = config

    def set_adapter(self, adapter_name: str) -> None:
        if adapter_name != self.active_adapter:
            raise ValueError(f"Adapter '{adapter_name}' does not exist on this ShadowModel.")

    def print_trainable_parameters(self) -> None:
        trainable, total = count_parameters(self)
        pct = (100.0 * trainable / total) if total else 0.0
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}")

    # ------------------------------------------------------------------ export

    @torch.no_grad()
    def export_shadow(self) -> PreTrainedModel:
        """
        Export a standalone, HF-compatible shadow model suitable for shadow-only inference / further pretraining.

        When shadow and base hidden sizes match, the base ``embed_tokens``/``lm_head`` are copied into a fresh model of
        the base class. When they differ and a trained projection is present, an
        [`AutoModelForCausalLMWithHiddenProjection`] bundling backbone + projection + base ``lm_head`` is returned.
        """
        shadow_h = int(self.shadow_hidden_size)
        base_h = int(self.base_hidden_size)
        hidden_match = shadow_h == base_h

        base_embed = self.model.get_input_embeddings()
        base_head = self.model.get_output_embeddings()

        target_device = target_dtype = None
        ref = base_head if (base_head is not None and getattr(base_head, "weight", None) is not None) else base_embed
        if ref is not None and getattr(ref, "weight", None) is not None:
            target_device = ref.weight.device
            target_dtype = ref.weight.dtype

        shadow_cfg = normalize_layer_config(deepcopy(self.shadow_model.config))

        if hidden_match:
            exported = self.model.__class__(shadow_cfg)
            exported_backbone = get_backbone(exported)
            exported_backbone.load_state_dict(get_backbone(self.shadow_model).state_dict(), strict=False)
            if base_embed is None or base_head is None:
                raise AttributeError("Base model must expose input and output embeddings to export the shadow model.")
            exported.set_input_embeddings(deepcopy(base_embed))
            exported.set_output_embeddings(deepcopy(base_head))
            tie_fn = getattr(exported, "tie_weights", None)
            if callable(tie_fn):
                tie_fn()
        elif isinstance(self.shadow_hidden_projection, nn.Linear) and base_head is not None:
            shadow_task = self.model.__class__(shadow_cfg)
            get_backbone(shadow_task).load_state_dict(get_backbone(self.shadow_model).state_dict(), strict=False)
            shadow_embed = None
            get_inp = getattr(self.shadow_model, "get_input_embeddings", None)
            if callable(get_inp):
                try:
                    shadow_embed = get_inp()
                except Exception:
                    shadow_embed = None
            if shadow_embed is not None:
                shadow_task.set_input_embeddings(deepcopy(shadow_embed))
            exported = AutoModelForCausalLMWithHiddenProjection.wrap(
                shadow_model=shadow_task,
                shadow_hidden_projection=deepcopy(self.shadow_hidden_projection),
                lm_head=deepcopy(base_head),
                init_optimal_projection=False,
            )
        else:
            exported = self.model.__class__(shadow_cfg)
            get_backbone(exported).load_state_dict(get_backbone(self.shadow_model).state_dict(), strict=False)

        if target_device is not None and target_dtype is not None:
            exported = exported.to(device=target_device, dtype=target_dtype)
        return exported

    # ------------------------------------------------------------------ attribute delegation

    def __getattr__(self, name: str):
        try:
            return nn.Module.__getattr__(self, name)
        except AttributeError:
            if name == "model":
                raise
            return getattr(nn.Module.__getattr__(self, "model"), name)
