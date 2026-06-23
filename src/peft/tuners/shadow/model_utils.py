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

"""Architecture-agnostic helpers for locating the transformer backbone/decoder stack of HF models.

These mirror common Hugging Face naming conventions so that the Shadow adapter can support most decoder-only
architectures (LLaMA/Mistral/Qwen via ``.model.layers``, GPT-2 via ``.transformer.h``, nested decoders via
``.model.decoder.layers``).
"""

import contextlib
from copy import deepcopy
from typing import Any, Optional

from torch import nn
from transformers import PreTrainedModel


def get_backbone(model: PreTrainedModel) -> nn.Module:
    """Return the module that contains the transformer decoder stack."""
    for attr in ("model", "transformer", "base_model", "decoder"):
        backbone = getattr(model, attr, None)
        if backbone is not None:
            return backbone
    # Some HF classes are already the "backbone" (e.g. LlamaModel, Qwen3Model).
    if hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList):
        return model
    if hasattr(model, "h") and isinstance(model.h, nn.ModuleList):
        return model
    if (
        hasattr(model, "decoder")
        and hasattr(model.decoder, "layers")
        and isinstance(model.decoder.layers, nn.ModuleList)
    ):
        return model.decoder
    raise AttributeError("Unable to locate transformer backbone inside the supplied model.")


def get_decoder_layers(model: PreTrainedModel) -> tuple[nn.Module, nn.ModuleList, str]:
    """Return ``(backbone, layers, layers_attr_name)`` for the decoder layer stack."""
    backbone = get_backbone(model)
    if hasattr(backbone, "layers") and isinstance(backbone.layers, nn.ModuleList):
        return backbone, backbone.layers, "layers"
    if hasattr(backbone, "h") and isinstance(backbone.h, nn.ModuleList):
        return backbone, backbone.h, "h"
    if (
        hasattr(backbone, "decoder")
        and hasattr(backbone.decoder, "layers")
        and isinstance(backbone.decoder.layers, nn.ModuleList)
    ):
        return backbone.decoder, backbone.decoder.layers, "layers"
    raise AttributeError(
        "Unsupported model: cannot find a ModuleList of decoder layers on the backbone (expected `.layers` or `.h`)."
    )


def get_hidden_size(model: PreTrainedModel) -> int:
    cfg = model.config
    for attr in ("hidden_size", "n_embd", "d_model"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    raise AttributeError("Unable to infer hidden size from model.config.")


def _set_intermediate_size(cfg: Any, value: int) -> bool:
    for attr in ("intermediate_size", "ffn_dim", "n_inner"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, int(value))
            return True
    return False


def _set_num_attention_heads(cfg: Any, value: int) -> bool:
    for attr in ("num_attention_heads", "n_head", "num_heads", "attention_heads"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, int(value))
            return True
    return False


def _set_num_key_value_heads(cfg: Any, value: int) -> bool:
    for attr in ("num_key_value_heads", "num_kv_heads", "n_kv_head"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, int(value))
            return True
    return False


def _set_head_dim(cfg: Any, value: int) -> bool:
    for attr in ("head_dim", "attention_head_dim"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, int(value))
            return True
    return False


def _set_num_hidden_layers(cfg: Any, n: int) -> None:
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, int(n))
            return
    raise AttributeError("Unable to set number of layers on this config.")


def extract_backbone_model(model: PreTrainedModel) -> PreTrainedModel:
    """Normalize an HF task model (e.g. ``Qwen3ForCausalLM``) to its backbone-only model (``Qwen3Model``)."""
    for attr in ("model", "transformer", "base_model", "decoder"):
        cand = getattr(model, attr, None)
        if isinstance(cand, PreTrainedModel):
            return cand
    return model


def remove_embed_tokens(module: nn.Module) -> None:
    """Drop ``embed_tokens`` from a module so it can be driven via shared base ``inputs_embeds``."""
    if hasattr(module, "embed_tokens") and isinstance(module.embed_tokens, nn.Module):
        try:
            module.embed_tokens = None
        except Exception:
            # Some models implement embed_tokens as a read-only property.
            module._modules.pop("embed_tokens", None)


def build_implicit_shadow_model(
    base_model: PreTrainedModel,
    num_shadow_layers: int,
    shadow_intermediate_size: Optional[int] = None,
    shadow_num_attention_heads: Optional[int] = None,
    shadow_num_key_value_heads: Optional[int] = None,
    shadow_head_dim: Optional[int] = None,
) -> PreTrainedModel:
    """
    Create an implicit shadow model by instantiating the same backbone class as ``base_model`` with a copied config but
    fewer layers and (optionally) smaller MLP/attention sizes. The shadow model is randomly initialized.
    """
    backbone = None
    for attr in ("model", "transformer", "base_model", "decoder"):
        cand = getattr(base_model, attr, None)
        if isinstance(cand, PreTrainedModel):
            backbone = cand
            break
    if backbone is None:
        backbone = base_model

    cfg = deepcopy(backbone.config)
    if num_shadow_layers < 1:
        raise ValueError(f"num_shadow_layers must be >= 1, got {num_shadow_layers}")
    _set_num_hidden_layers(cfg, num_shadow_layers)

    if shadow_intermediate_size is not None:
        _set_intermediate_size(cfg, shadow_intermediate_size)

    if shadow_num_attention_heads is not None and not _set_num_attention_heads(cfg, shadow_num_attention_heads):
        raise ValueError(
            "shadow_num_attention_heads was set, but this model config does not expose a recognized attention-heads "
            "field (e.g. num_attention_heads / n_head)."
        )
    if shadow_num_key_value_heads is not None and not _set_num_key_value_heads(cfg, shadow_num_key_value_heads):
        raise ValueError(
            "shadow_num_key_value_heads was set, but this model config does not expose a recognized kv-heads field "
            "(e.g. num_key_value_heads / num_kv_heads)."
        )
    if shadow_head_dim is not None and not _set_head_dim(cfg, shadow_head_dim):
        raise ValueError(
            "shadow_head_dim was set, but this model config does not expose a recognized head-dim field (e.g. head_dim)."
        )

    if hasattr(cfg, "use_cache"):
        cfg.use_cache = False

    # `layer_types` (Qwen3 etc.) must match the reduced number of layers.
    normalize_layer_config(cfg)

    return backbone.__class__(cfg)


def normalize_layer_config(cfg: Any) -> Any:
    """Best-effort fixups so configs with per-layer lists stay consistent with a reduced ``num_hidden_layers``."""
    try:
        num_layers = int(cfg.num_hidden_layers)
    except Exception:
        num_layers = None

    if num_layers is not None and getattr(cfg, "layer_types", None) is not None:
        lt = cfg.layer_types
        lt_list = list(lt)
        if len(lt_list) == 0:
            lt_list = ["full_attention"] * num_layers
        elif len(lt_list) > num_layers:
            lt_list = lt_list[:num_layers]
        elif len(lt_list) < num_layers:
            lt_list = lt_list + [lt_list[-1]] * (num_layers - len(lt_list))
        cfg.layer_types = lt_list

    if num_layers is not None and hasattr(cfg, "max_window_layers"):
        with contextlib.suppress(Exception):
            mw = cfg.max_window_layers
            if mw is not None and int(mw) > num_layers:
                cfg.max_window_layers = int(num_layers)

    return cfg


def count_parameters(module: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def resolve_explicit_shadow_model_name(shadow_model: PreTrainedModel) -> Optional[str]:
    """Best-effort hub/local path for an explicit shadow model."""
    for obj in (shadow_model, getattr(shadow_model, "config", None)):
        if obj is None:
            continue
        for attr in ("name_or_path", "_name_or_path"):
            path = getattr(obj, attr, None)
            if path:
                return str(path)
    return None


def load_explicit_shadow_model(model_name_or_path: str, torch_dtype=None, **kwargs) -> PreTrainedModel:
    """Load an explicit shadow model used to initialize ShadowPEFT architecture at reload time."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from .projected_causal_lm import AutoModelForCausalLMWithHiddenProjection

    hf_kwargs = dict(kwargs)
    if torch_dtype is not None:
        hf_kwargs["torch_dtype"] = torch_dtype
    try:
        model_type = getattr(AutoConfig.from_pretrained(model_name_or_path, **hf_kwargs), "model_type", None)
    except Exception:
        model_type = None
    if model_type == "causal_lm_with_hidden_projection":
        return AutoModelForCausalLMWithHiddenProjection.from_pretrained(
            model_name_or_path, freeze_backbone=False, **hf_kwargs
        )
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **hf_kwargs)
