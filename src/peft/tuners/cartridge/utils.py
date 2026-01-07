# Copyright 2025-present the HuggingFace Inc. team.
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

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import torch
from safetensors.torch import save_file

from peft.config import PeftConfig
from peft.utils import PeftType
from peft.utils.constants import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME
from peft.utils.save_and_load import load_peft_weights


def _to_legacy_past_key_values(past_key_values: Any):
    # Support both legacy tuples and transformers.Cache-like objects.
    if isinstance(past_key_values, (tuple, list)):
        return past_key_values
    to_legacy = getattr(past_key_values, "to_legacy_cache", None)
    if callable(to_legacy):
        return to_legacy()
    if hasattr(past_key_values, "__iter__"):
        legacy = list(past_key_values)
        if legacy and isinstance(legacy[0], (tuple, list)) and len(legacy[0]) >= 2:
            return [(layer[0], layer[1]) for layer in legacy]
    raise TypeError(
        "Unsupported `past_key_values` type. Expected a legacy tuple/list, an object with `to_legacy_cache()`, or an "
        "iterable of (key, value) tuples."
    )


def prompt_embeddings_from_past_key_values(
    past_key_values: Any,
    *,
    num_virtual_tokens: int,
) -> torch.Tensor:
    """
    Convert a (legacy) `past_key_values` cache into the flattened prompt embeddings tensor saved by PEFT.

    The output matches the layout expected by `PeftModel.get_prompt()` for prefix-style prompt learning: shape
    `[num_virtual_tokens, num_layers * 2 * token_dim]`.
    """
    legacy = _to_legacy_past_key_values(past_key_values)
    if len(legacy) == 0:
        raise ValueError("Empty `past_key_values`.")

    # Each layer: (key, value), where key/value are [batch, num_heads, seq_len, head_dim]
    num_layers = len(legacy)
    key0, value0 = legacy[0]
    if key0.ndim != 4:
        raise ValueError(f"Expected key/value tensors with rank 4, got key.ndim={key0.ndim}.")
    if key0.shape[0] != 1:
        raise ValueError(
            "This helper expects `past_key_values` from a single-sequence prefill (batch=1). "
            f"Got batch={key0.shape[0]}."
        )
    num_heads = key0.shape[1]
    seq_len = key0.shape[2]
    head_dim = key0.shape[3]
    if seq_len < num_virtual_tokens:
        raise ValueError(f"Need at least {num_virtual_tokens} cached tokens, got {seq_len}.")

    packed = torch.empty(
        num_virtual_tokens,
        num_layers * 2,
        num_heads,
        head_dim,
        device=key0.device,
        dtype=key0.dtype,
    )
    for layer_idx, (k, v) in enumerate(legacy):
        if k.shape[:2] != (1, num_heads) or v.shape[:2] != (1, num_heads):
            raise ValueError("Inconsistent head shapes across layers in `past_key_values`.")
        if k.shape[2] < num_virtual_tokens or v.shape[2] < num_virtual_tokens:
            raise ValueError("Not enough cached tokens in `past_key_values` for the requested cartridge length.")
        if k.shape[3] != head_dim or v.shape[3] != head_dim:
            raise ValueError("Inconsistent head_dim across layers in `past_key_values`.")
        packed[:, 2 * layer_idx] = k[0, :, :num_virtual_tokens, :].transpose(0, 1).contiguous()
        packed[:, 2 * layer_idx + 1] = v[0, :, :num_virtual_tokens, :].transpose(0, 1).contiguous()

    return packed.reshape(num_virtual_tokens, -1)


@torch.no_grad()
def initialize_kv_prefix_from_past_key_values(
    model,
    *,
    adapter_name: Optional[str] = None,
    past_key_values: Any,
    num_virtual_tokens: Optional[int] = None,
) -> torch.Tensor:
    """
    Initialize a KV-prefix prompt-learning adapter from an existing cached prefix (`past_key_values`).

    Returns the prompt embeddings tensor that was loaded into the adapter.
    """
    if adapter_name is None:
        adapter_name = model.active_adapter
    config = model.peft_config[adapter_name]
    if config.peft_type not in (PeftType.CARTRIDGE, PeftType.PREFIX_TUNING):
        raise ValueError(
            f"Adapter '{adapter_name}' must be a CARTRIDGE or PREFIX_TUNING adapter (got {config.peft_type})."
        )
    if getattr(config, "prefix_projection", False):
        raise ValueError(
            "Initialization from KV cache is not supported for prefix tuning with `prefix_projection=True`."
        )
    if num_virtual_tokens is None:
        num_virtual_tokens = config.num_virtual_tokens

    prompt_embeddings = prompt_embeddings_from_past_key_values(past_key_values, num_virtual_tokens=num_virtual_tokens)
    model.prompt_encoder[adapter_name].load_prompt_embeddings(prompt_embeddings)
    return prompt_embeddings


@torch.no_grad()
def initialize_kv_prefix_from_text(
    model,
    tokenizer,
    *,
    text: str,
    adapter_name: Optional[str] = None,
    num_virtual_tokens: Optional[int] = None,
    use_chat_template: bool = True,
    max_length: Optional[int] = None,
) -> torch.Tensor:
    """
    Convenience initializer: prefill the base model on `text` and load the resulting cache prefix into the adapter.
    """
    if adapter_name is None:
        adapter_name = model.active_adapter
    config = model.peft_config[adapter_name]
    if config.peft_type not in (PeftType.CARTRIDGE, PeftType.PREFIX_TUNING):
        raise ValueError(
            f"Adapter '{adapter_name}' must be a CARTRIDGE or PREFIX_TUNING adapter (got {config.peft_type})."
        )
    if getattr(config, "prefix_projection", False):
        raise ValueError(
            "Initialization from KV cache is not supported for prefix tuning with `prefix_projection=True`."
        )
    if num_virtual_tokens is None:
        num_virtual_tokens = config.num_virtual_tokens

    def _tokenize_plain():
        toks = tokenizer(text, return_tensors="pt", truncation=max_length is not None, max_length=max_length)
        return toks["input_ids"]

    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            input_ids = tokenizer.apply_chat_template(
                [{"role": "system", "content": text}],
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,
                return_tensors="pt",
            )
        except (TypeError, ValueError):
            # Some tokenizers don't support the full signature or do not define a chat template.
            input_ids = _tokenize_plain()
        else:
            if max_length is not None and input_ids.shape[1] > max_length:
                input_ids = input_ids[:, :max_length]
    else:
        input_ids = _tokenize_plain()

    input_ids = input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with model.disable_adapter():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    return initialize_kv_prefix_from_past_key_values(
        model,
        adapter_name=adapter_name,
        past_key_values=outputs.past_key_values,
        num_virtual_tokens=num_virtual_tokens,
    )


def compose_cartridge_adapters(
    adapter_paths: Sequence[str | Path],
    *,
    output_path: str | Path,
    safe_serialization: bool = True,
) -> None:
    """
    Compose multiple CARTRIDGE adapters by concatenating their prompt embeddings.

    This implements the paper's "composition via concatenation" behavior at the adapter level (no runtime
    multi-adapter).
    """
    adapter_paths = [Path(p) for p in adapter_paths]
    if len(adapter_paths) < 2:
        raise ValueError("Need at least 2 adapters to compose.")

    configs = [PeftConfig.from_pretrained(str(p)) for p in adapter_paths]
    for p, cfg in zip(adapter_paths, configs):
        if cfg.peft_type != PeftType.CARTRIDGE:
            raise ValueError(f"Adapter at '{p}' is not a CARTRIDGE adapter (got {cfg.peft_type}).")

    base = configs[0]
    for cfg in configs[1:]:
        for attr in ("task_type", "token_dim", "num_layers", "num_attention_heads", "num_transformer_submodules"):
            if getattr(cfg, attr, None) != getattr(base, attr, None):
                raise ValueError(f"Incompatible CARTRIDGE configs for attribute '{attr}'.")

    weights = [load_peft_weights(str(p), device="cpu") for p in adapter_paths]
    prompt_embeddings = [w["prompt_embeddings"] for w in weights]
    composed = torch.cat(prompt_embeddings, dim=0)

    num_virtual_tokens = composed.shape[0]
    # Preserve the "frozen prefix tokens" count of the first adapter only (matches a single attention-sink prefix).
    num_frozen_tokens = base.num_frozen_tokens
    out_cfg = replace(base, num_virtual_tokens=num_virtual_tokens, num_frozen_tokens=num_frozen_tokens)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    out_cfg.save_pretrained(str(output_path))

    if safe_serialization:
        save_file({"prompt_embeddings": composed}, str(output_path / SAFETENSORS_WEIGHTS_NAME))
    else:
        torch.save({"prompt_embeddings": composed}, str(output_path / WEIGHTS_NAME))
