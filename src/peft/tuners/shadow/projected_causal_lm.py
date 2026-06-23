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

"""Standalone HF model that wraps a small shadow backbone with a hidden-size projection and a (larger) base ``lm_head``.

This is the canonical distribution format for pre-trained shadow models that target a larger base model's hidden /
vocabulary space, and the type returned by [`ShadowModel.export_shadow`] when shadow and base hidden sizes differ.
"""

import contextlib
import importlib
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .model_utils import extract_backbone_model, get_backbone


def _import_from_path(path: str):
    """Import ``"some.module:ClassName"`` or ``"some.module.ClassName"`` and return the symbol."""
    if ":" in path:
        mod, name = path.split(":", 1)
    else:
        mod, name = path.rsplit(".", 1)
    m = importlib.import_module(mod)
    return getattr(m, name)


def _shifted_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


class AutoModelForCausalLMWithHiddenProjectionConfig(PretrainedConfig):
    """
    Config for a projected CausalLM::

        shadow_causal_lm (hidden=shadow_h)
          -> shadow_hidden_projection (shadow_h -> base_h)
          -> lm_head (base_h -> vocab)
    """

    model_type = "causal_lm_with_hidden_projection"

    def __init__(
        self,
        shadow_model_class: str = "",
        shadow_model_config_class: str = "",
        shadow_model_config: Optional[dict[str, Any]] = None,
        base_hidden_size: int = 0,
        vocab_size: int = 0,
        **kwargs,
    ) -> None:
        # Set hidden_size to base_hidden_size so transformers utilities use the correct dimension. The shadow model's
        # internal hidden size is stored separately in shadow_model_config.
        if "hidden_size" not in kwargs and base_hidden_size > 0:
            kwargs["hidden_size"] = int(base_hidden_size)

        super().__init__(vocab_size=int(vocab_size or kwargs.get("vocab_size", 0)), **kwargs)
        self.shadow_model_class = str(shadow_model_class)
        self.shadow_model_config_class = str(shadow_model_config_class)
        self.shadow_model_config = dict(shadow_model_config or {})
        self.base_hidden_size = int(base_hidden_size)

        if self.base_hidden_size > 0:
            self.hidden_size = int(self.base_hidden_size)

        # Expose shadow model attributes needed for generation and other HF utilities.
        _shadow_cfg = self.shadow_model_config
        if _shadow_cfg:
            for attr in (
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "intermediate_size",
                "max_position_embeddings",
                "rope_theta",
                "rms_norm_eps",
                "layer_types",
                "sliding_window",
                "attention_dropout",
                "head_dim",
                "use_sliding_window",
                "max_window_layers",
            ):
                if attr in _shadow_cfg and not hasattr(self, attr):
                    setattr(self, attr, _shadow_cfg[attr])


@dataclass
class AutoModelForCausalLMWithHiddenProjectionOutput(CausalLMOutputWithPast):
    """Same as ``CausalLMOutputWithPast`` but explicit for this model type."""


class AutoModelForCausalLMWithHiddenProjection(PreTrainedModel, GenerationMixin):
    """
    A CausalLM wrapper that is saved/loaded like a normal HF model, but projects the shadow hidden size to the base
    hidden size before applying ``lm_head``.
    """

    config_class = AutoModelForCausalLMWithHiddenProjectionConfig

    _is_stateful: bool = False
    _supports_sdpa: bool = True

    def __init__(self, config: AutoModelForCausalLMWithHiddenProjectionConfig) -> None:
        super().__init__(config)

        if not config.shadow_model_class:
            raise ValueError("Missing `shadow_model_class` in AutoModelForCausalLMWithHiddenProjectionConfig.")
        if not config.shadow_model_config_class:
            raise ValueError("Missing `shadow_model_config_class` in AutoModelForCausalLMWithHiddenProjectionConfig.")
        if not config.shadow_model_config:
            raise ValueError("Missing `shadow_model_config` in AutoModelForCausalLMWithHiddenProjectionConfig.")
        if int(getattr(config, "base_hidden_size", 0)) <= 0:
            raise ValueError("Missing/invalid `base_hidden_size` in AutoModelForCausalLMWithHiddenProjectionConfig.")
        if int(getattr(config, "vocab_size", 0)) <= 0:
            raise ValueError("Missing/invalid `vocab_size` in AutoModelForCausalLMWithHiddenProjectionConfig.")

        cfg_cls = _import_from_path(config.shadow_model_config_class)
        if not issubclass(cfg_cls, PretrainedConfig):
            raise TypeError(
                f"shadow_model_config_class must be a PretrainedConfig, got {cfg_cls} from "
                f"{config.shadow_model_config_class}"
            )
        shadow_cfg = cfg_cls.from_dict(config.shadow_model_config)

        # Propagate attn_implementation from the outer config to the inner backbone.
        _attn_impl = getattr(config, "_attn_implementation_internal", None) or getattr(
            config, "_attn_implementation", None
        )
        if _attn_impl:
            shadow_cfg._attn_implementation = _attn_impl

        shadow_hidden_size = int(getattr(shadow_cfg, "hidden_size", getattr(shadow_cfg, "n_embd", 0)))
        if shadow_hidden_size <= 0:
            raise ValueError("Could not infer shadow hidden size from shadow model config.")

        # Instantiate ONLY the backbone class (never a task model with lm_head).
        model_cls = _import_from_path(config.shadow_model_class)
        model_cls_name = model_cls.__name__

        if model_cls_name.endswith("ForCausalLM"):
            backbone_cls_name = model_cls_name.replace("ForCausalLM", "Model")
            try:
                backbone_cls = getattr(
                    __import__(model_cls.__module__, fromlist=[backbone_cls_name]), backbone_cls_name
                )
                self.shadow_model = backbone_cls(shadow_cfg)
            except (AttributeError, ImportError) as e:
                raise ValueError(
                    f"Could not instantiate backbone class '{backbone_cls_name}' from module '{model_cls.__module__}'. "
                    f"Please update the saved config's 'shadow_model_class' to point directly to the backbone "
                    f"(e.g., 'transformers.models.qwen3.modeling_qwen3:Qwen3Model' instead of ':Qwen3ForCausalLM'). "
                    f"Error: {e}"
                ) from e
        else:
            self.shadow_model = model_cls(shadow_cfg)
            if hasattr(self.shadow_model, "model") and isinstance(self.shadow_model.model, nn.Module):
                backbone = self.shadow_model.model
                if hasattr(self.shadow_model, "config") and not hasattr(backbone, "config"):
                    backbone.config = self.shadow_model.config
                self.shadow_model = backbone

        if hasattr(self.shadow_model, "lm_head"):
            warnings.warn(
                f"shadow_model ({type(self.shadow_model).__name__}) has an 'lm_head' attribute, which may cause "
                "weight loading conflicts. Consider using the backbone class instead."
            )

        self.shadow_hidden_projection = nn.Linear(shadow_hidden_size, int(config.base_hidden_size), bias=False)
        self.lm_head = nn.Linear(int(config.base_hidden_size), int(config.vocab_size), bias=False)

        # Keep module order in repr: projection before lm_head.
        with contextlib.suppress(Exception):
            lm = self._modules.pop("lm_head", None)
            if lm is not None:
                self._modules["lm_head"] = lm

        self.post_init()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        freeze_backbone: bool = False,
        freeze_embed_tokens: bool = True,
        freeze_lm_head: bool = True,
        **kwargs,
    ) -> "AutoModelForCausalLMWithHiddenProjection":
        kwargs_for_super = {
            k: v for k, v in kwargs.items() if k not in ("freeze_backbone", "freeze_embed_tokens", "freeze_lm_head")
        }
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs_for_super)

        if freeze_backbone:
            for param in model.shadow_model.parameters():
                param.requires_grad = False
        if freeze_embed_tokens:
            embed = model.get_input_embeddings()
            if embed is not None:
                for param in embed.parameters():
                    param.requires_grad = False
        if freeze_lm_head:
            for param in model.lm_head.parameters():
                param.requires_grad = False
        return model

    @classmethod
    def wrap(
        cls,
        shadow_model: PreTrainedModel,
        shadow_hidden_projection: nn.Linear,
        lm_head: nn.Module,
        init_optimal_projection: bool = True,
        reference_lm_head: Optional[nn.Module] = None,
    ) -> "AutoModelForCausalLMWithHiddenProjection":
        """
        Wrap an already-instantiated shadow backbone + projection + ``lm_head`` into a single loadable model.

        When ``init_optimal_projection=True`` the projection is initialized to minimize
        ``||W_lm_large @ W_proj - W_lm_small||`` via a pseudo-inverse, providing a better fine-tuning start point
        (requires ``reference_lm_head``).
        """
        shadow_backbone = extract_backbone_model(shadow_model)
        shadow_cfg_dict = shadow_backbone.config.to_dict()

        cfg = AutoModelForCausalLMWithHiddenProjectionConfig(
            shadow_model_class=f"{shadow_backbone.__class__.__module__}:{shadow_backbone.__class__.__name__}",
            shadow_model_config_class=(
                f"{shadow_backbone.config.__class__.__module__}:{shadow_backbone.config.__class__.__name__}"
            ),
            shadow_model_config=shadow_cfg_dict,
            base_hidden_size=int(getattr(lm_head, "in_features", getattr(cls, "base_hidden_size", 0))),
            vocab_size=int(getattr(shadow_backbone.config, "vocab_size", 0) or getattr(lm_head, "out_features", 0)),
        )
        out = cls(cfg)
        ref = next(shadow_backbone.parameters(), None)
        if ref is not None:
            out = out.to(device=ref.device, dtype=ref.dtype)

        out.shadow_model.load_state_dict(shadow_backbone.state_dict(), strict=True)

        if init_optimal_projection:
            if reference_lm_head is None:
                raise ValueError(
                    "When init_optimal_projection=True, you must provide reference_lm_head (the original model's "
                    "lm_head to approximate)."
                )
            W_old = reference_lm_head.weight.data  # [vocab, shadow_hidden]
            W_lm_frozen = lm_head.weight.data  # [vocab, base_hidden]
            # Solve W_lm_frozen @ W_proj.T = W_old  =>  W_proj.T = pinv(W_lm_frozen) @ W_old
            W_lm_pinv = torch.linalg.pinv(W_lm_frozen.float())  # [base_hidden, vocab]
            W_proj_optimal_T = W_lm_pinv @ W_old.float()  # [base_hidden, shadow_hidden]
            out.shadow_hidden_projection.weight.data = W_proj_optimal_T.to(out.shadow_hidden_projection.weight.dtype)
        else:
            out.shadow_hidden_projection.load_state_dict(shadow_hidden_projection.state_dict(), strict=True)

        out.lm_head.load_state_dict(lm_head.state_dict(), strict=True)
        return out

    def get_input_embeddings(self):
        get_inp = getattr(self.shadow_model, "get_input_embeddings", None)
        return get_inp() if callable(get_inp) else None

    def set_input_embeddings(self, value):
        set_inp = getattr(self.shadow_model, "set_input_embeddings", None)
        if callable(set_inp):
            return set_inp(value)
        raise AttributeError("Underlying shadow model does not support set_input_embeddings().")

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self, **kwargs):
        # lm_head and embeddings are intentionally separate (different hidden sizes).
        pass

    def _init_weights(self, module):
        # Weights are loaded from checkpoint instead of randomly initialized during loading.
        pass

    def prepare_inputs_for_generation(self, *args, **kwargs):
        fn = getattr(self.shadow_model, "prepare_inputs_for_generation", None)
        if callable(fn):
            return fn(*args, **kwargs)
        return super().prepare_inputs_for_generation(*args, **kwargs)

    def _reorder_cache(self, past_key_values, beam_idx):
        fn = getattr(self.shadow_model, "_reorder_cache", None)
        if callable(fn):
            return fn(past_key_values, beam_idx)
        return past_key_values

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values=None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if return_dict is None:
            return_dict = True

        backbone = get_backbone(self.shadow_model)
        out = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=True,
            **kwargs,
        )

        hidden = out.last_hidden_state
        hidden_base = self.shadow_hidden_projection(hidden)
        logits = self.lm_head(hidden_base)

        loss = None
        if labels is not None:
            loss = _shifted_ce_loss(logits, labels)

        if not return_dict:
            if loss is not None:
                return (loss, logits, getattr(out, "past_key_values", None), None, None)
            return (logits, getattr(out, "past_key_values", None), None, None)

        return AutoModelForCausalLMWithHiddenProjectionOutput(
            loss=loss,
            logits=logits,
            past_key_values=getattr(out, "past_key_values", None),
            hidden_states=getattr(out, "hidden_states", None),
            attentions=getattr(out, "attentions", None),
        )
