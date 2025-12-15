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

from typing import Any, Optional

import torch

from peft.tuners.cartridge.utils import prompt_embeddings_from_past_key_values
from peft.utils import PeftType


@torch.no_grad()
def initialize_prefix_tuning_from_past_key_values(
    model,
    *,
    adapter_name: Optional[str] = None,
    past_key_values: Any,
    num_virtual_tokens: Optional[int] = None,
) -> torch.Tensor:
    """
    Initialize a PREFIX_TUNING adapter's parameters from an existing cached prefix (`past_key_values`).

    This is supported only for `prefix_projection=False` (the default), because then the adapter parameters are the KV
    prefix itself (`embedding.weight`).
    """
    if adapter_name is None:
        adapter_name = model.active_adapter
    config = model.peft_config[adapter_name]
    if config.peft_type != PeftType.PREFIX_TUNING:
        raise ValueError(f"Adapter '{adapter_name}' is not a PREFIX_TUNING adapter (got {config.peft_type}).")
    if getattr(config, "prefix_projection", False):
        raise ValueError(
            "Initialization from KV cache is not supported for prefix tuning with `prefix_projection=True`."
        )
    if num_virtual_tokens is None:
        num_virtual_tokens = config.num_virtual_tokens

    prompt_embeddings = prompt_embeddings_from_past_key_values(past_key_values, num_virtual_tokens=num_virtual_tokens)
    loader = getattr(model.prompt_encoder[adapter_name], "load_prompt_embeddings", None)
    if callable(loader):
        loader(prompt_embeddings)
    else:
        model.prompt_encoder[adapter_name].embedding.load_state_dict({"weight": prompt_embeddings}, strict=True)
    return prompt_embeddings


@torch.no_grad()
def initialize_prefix_tuning_from_text(
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
    Convenience initializer for PREFIX_TUNING: prefill the base model on `text` and load the cache prefix into the
    adapter.
    """
    if adapter_name is None:
        adapter_name = model.active_adapter
    config = model.peft_config[adapter_name]
    if config.peft_type != PeftType.PREFIX_TUNING:
        raise ValueError(f"Adapter '{adapter_name}' is not a PREFIX_TUNING adapter (got {config.peft_type}).")
    if getattr(config, "prefix_projection", False):
        raise ValueError(
            "Initialization from KV cache is not supported for prefix tuning with `prefix_projection=True`."
        )
    if num_virtual_tokens is None:
        num_virtual_tokens = config.num_virtual_tokens

    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            input_ids = tokenizer.apply_chat_template(
                [{"role": "system", "content": text}],
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            )
        except TypeError:
            input_ids = torch.tensor(tokenizer.apply_chat_template([{"role": "system", "content": text}]))[None, :]
    else:
        toks = tokenizer(text, return_tensors="pt", truncation=max_length is not None, max_length=max_length)
        input_ids = toks["input_ids"]

    input_ids = input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with model.disable_adapter():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    return initialize_prefix_tuning_from_past_key_values(
        model,
        adapter_name=adapter_name,
        past_key_values=outputs.past_key_values,
        num_virtual_tokens=num_virtual_tokens,
    )
