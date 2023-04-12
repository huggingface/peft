# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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

import importlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.utils.config import PeftConfig, PeftType
from peft.utils.other import _freeze_adapter, _get_submodules


def is_llama_available() -> bool:
    """Check if Llama is available in the transformers library (it's not in earlier versions)."""
    try:
        return importlib.util.find_spec("transformers.models.llama.modeling_llama") is not None
    except ModuleNotFoundError:
        return False


if is_llama_available():
    # We guard the import statement so that this won't get in the way of using peft with earlier versions of
    # transformers that don't have Llama.
    from transformers.models.llama.modeling_llama import (
        LlamaForCausalLM,
        LlamaModel,
        apply_rotary_pos_emb,
    )


def is_adaption_prompt_trainable(params: str) -> bool:
    """Return True if module is trainable under adaption prompt fine-tuning."""
    return params.split(".")[-1].startswith("adaption_")


@dataclass
class AdaptionPromptConfig(PeftConfig):
    """Stores the configuration of an [`AdaptionPromptModel`]."""

    target_modules: str = field(
        default=None, metadata={"help": "Name of the attention submodules to insert adaption prompts into."}
    )
    adapter_len: int = field(default=None, metadata={"help": "Number of adapter tokens to insert"})
    adapter_layers: int = field(default=None, metadata={"help": "Number of adapter layers (from the top)"})

    def __post_init__(self):
        self.peft_type = PeftType.ADAPTION_PROMPT


def prepare_config(
    peft_config: AdaptionPromptConfig,
    model,
) -> AdaptionPromptConfig:
    """Prepare the config based on the llama model type."""
    if peft_config.target_modules is not None:
        # Assume user knows what they are doing.
        return peft_config

    if isinstance(model, (LlamaModel, LlamaForCausalLM)):
        peft_config.target_modules = "self_attn"
    else:
        raise ValueError(f"Unsupported model type for adaption prompt: '{type(model)}'")

    return peft_config


class AdaptionPromptModel(nn.Module):
    """
    Implements adaption prompts as described in https://arxiv.org/pdf/2303.16199.pdf.

    This implementation only supports Llama models at the moment, but it can be extended to other models. It is
    implemented by replacing some of the LlamaAttention modules with AdaptedAttention modules that wrap the original
    ones, but bypass the original forward() so that we can inject trainable tokens and a gate.

    Notes on the multi-adapter pattern:
    - We store the states for different adapters by keeping a dictionary of AdaptedAttention modules indexed by adapter
      name.
    - Every time we switch adapters, we remove the modules of the currently active adapter from the LlamaModel layers,
      store them in the dictionary, and replace them with the modules of the new adapter.
    - To avoid duplicated and potentially inconsistent state, the currently active adapter is always removed from the
      dictionary.
    - Disabling the adapter would also result in the modules being removed from the LlamaModel layers.
    """

    def __init__(self, model, configs: Dict, adapter_name: str):
        super().__init__()
        self.model = model
        # Store adapter configs by name.
        self._configs: Dict[str, AdaptionPromptConfig] = {}
        # Store lists of the parents of the attention modules by adapter name.
        self._parents: Dict[str, List[nn.Module]] = {}
        # Store lists of cached AdaptedAttention modules by name.
        self._cached_adapters: Dict[str, List] = {}
        # The name of the currently active adapter.
        self._active_adapter = None
        # Whether the adapter is enabled.
        self._enabled = True
        self.forward = self.model.forward
        self.add_adapter(adapter_name, configs[adapter_name])
        self._mark_only_adaption_prompts_as_trainable()

    def add_adapter(self, adapter_name: str, config: AdaptionPromptConfig) -> None:
        """Add an adapter with the given name and config."""
        config = prepare_config(config, self.model)
        if adapter_name in self._configs:
            raise ValueError(f"Adapter with name '{adapter_name}' already exists.")

        parents = []
        for name, _ in self.model.named_modules():
            if name.endswith(config.target_modules):
                parent, _, _ = _get_submodules(self.model, name)
                parents.append(parent)
        if len(parents) < config.adapter_layers:
            raise ValueError(
                f"Config specifies more adapter layers '{config.adapter_layers}'"
                f" than the model has '{len(parents)}'."
            )
        # Note that if the target modules are not in Sequential, ModuleList, or
        # some other PyTorch ordered container, the behavior is undefined as we
        # assume here that the order of the modules is the same as the order of
        # the transformer decoder layers.
        parents = parents[-config.adapter_layers :]
        self._parents[adapter_name] = parents

        # It is only None during initialization.
        # If it is disabled, we don't have to remove the modules.
        if self._active_adapter is not None and self._enabled:
            self._remove_adapted_attentions(self._active_adapter)
        self._active_adapter = adapter_name
        self._configs[adapter_name] = config
        self._create_adapted_attentions(config, parents)
        if not self._enabled:
            self._remove_adapted_attentions(self._active_adapter)

        if config.inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def set_adapter(self, adapter_name: str) -> None:
        """Set the model to use the adapter with the given name."""
        if self._active_adapter == adapter_name:
            return
        if adapter_name not in self._configs:
            raise ValueError(f"Adapter with name '{adapter_name}' does not exist.")

        if self._enabled:
            self._remove_adapted_attentions(self._active_adapter)
            self._set_adapted_attentions(adapter_name)

        self._active_adapter = adapter_name

    def enable_adapter_layers(self):
        """Enable adapter layers by swapping in cached AdaptedAttention modules."""
        self._enabled = True
        self._set_adapted_attentions(self._active_adapter)

    def disable_adapter_layers(self):
        """Disable adapter layers by swapping out AdaptedAttention modules."""
        self._enabled = False
        self._remove_adapted_attentions(self._active_adapter)

    def _create_adapted_attentions(self, config: AdaptionPromptConfig, parents: List[nn.Module]) -> None:
        """Wrap LlamaAttention modules with newly created AdaptedAttention modules."""
        for par in parents:
            setattr(
                par, config.target_modules, AdaptedAttention(config.adapter_len, getattr(par, config.target_modules))
            )

    def _set_adapted_attentions(self, adapter_name: str) -> None:
        """Replace LlamaAttention modules with cached AdaptedAttention modules."""
        cached = self._cached_adapters[adapter_name]
        del self._cached_adapters[adapter_name]
        config = self._configs[adapter_name]
        for i, par in enumerate(self._parents[adapter_name]):
            setattr(par, config.target_modules, cached[i])

    def _remove_adapted_attentions(self, adapter_name: str) -> None:
        """Remove AdaptedAttention modules from the model and store them in the cache."""
        config = self._configs[adapter_name]
        adapted_attentions = []
        for par in self._parents[adapter_name]:
            attn = getattr(par, config.target_modules)
            adapted_attentions.append(attn)
            setattr(par, config.target_modules, attn.model)
        self._cached_adapters[adapter_name] = adapted_attentions

    def _mark_only_adaption_prompts_as_trainable(self) -> None:
        """Freeze all parameters of the model except the adaption prompts."""
        for n, p in self.model.named_parameters():
            if not is_adaption_prompt_trainable(n):
                p.requires_grad = False

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            # This is necessary as e.g. causal models have various methods that we
            # don't want to re-implement here.
            return getattr(self.model, name)


class AdaptedAttention(nn.Module):
    """This module wraps a LLamaAttention module and injects adaption prompts."""

    def __init__(self, adapter_len: int, model):
        assert not isinstance(model, AdaptedAttention)
        super().__init__()
        self.model = model
        self.adapter_len = adapter_len
        # Don't think this was specified in the paper, but we follow the official repo which used an Embedding
        # which initializes the tokens with standard normal values.
        # https://github.com/ZrrSkywalker/LLaMA-Adapter/blob/41c3546fe1997ab8a65809dc8d8f9252b19d9faf/llama/model.py#L234
        # (bsz, adapter_len, hidden_size)
        self.adaption_prompt = nn.Parameter(torch.empty(1, adapter_len, self.model.hidden_size).normal_())
        # Initialize the gate to 0 as this is "zero-init".
        self.adaption_gate = nn.Parameter(torch.zeros(1))

    def _modified_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        """
        Modified from:
        https://github.com/huggingface/transformers/blob/d143087d189a183b153090c8b37daa6c7c031039/src/transformers/models/llama/modeling_llama.py#L185

        The original forward function is bypassed in favor of this one, which was modified to:
        1. Return query_states.
        2. Not apply o_proj yet.

        The alternatives are:
        1. to refactor the LlamaAttention module in the transformers package,
        2. or to use the original forward() and recompute some of the intermediate local variables.
        """
        bsz, q_len, _ = hidden_states.size()

        # (bsz, num_heads, q_len, head_dim)
        query_states = (
            self.model.q_proj(hidden_states)
            .view(bsz, q_len, self.model.num_heads, self.model.head_dim)
            .transpose(1, 2)
        )
        # (bsz, num_heads, q_len, head_dim)
        key_states = (
            self.model.k_proj(hidden_states)
            .view(bsz, q_len, self.model.num_heads, self.model.head_dim)
            .transpose(1, 2)
        )
        # (bsz, num_heads, q_len, head_dim)
        value_states = (
            self.model.v_proj(hidden_states)
            .view(bsz, q_len, self.model.num_heads, self.model.head_dim)
            .transpose(1, 2)
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.model.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.model.head_dim)

        if attn_weights.size() != (bsz, self.model.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.model.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.model.num_heads, q_len, self.model.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.model.num_heads, q_len, self.model.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.model.hidden_size)

        if not output_attentions:
            attn_weights = None

        return query_states, attn_output, attn_weights, past_key_value

    def forward(self, **kwargs):
        """
        Forward pass for the adapter which wraps the original LlamaAttention module.

        "Official" paper implementation:
        https://github.com/ZrrSkywalker/LLaMA-Adapter/blob/41c3546fe1997ab8a65809dc8d8f9252b19d9faf/llama/model.py#L141

        Args:
            kwargs: See the original LlamaAttention module.
        """
        if kwargs.get("use_cache", False):
            raise NotImplementedError("use_cache is not currently supported.")
        if kwargs.get("output_attention", False):
            raise NotImplementedError("output_attention is not currently supported.")

        # The original forward() was modified to return query_states and not to apply
        # `self.model.o_proj` to output yet.
        # query_states: (bsz, num_heads, q_len, head_dim)
        query_states, output, _, _ = self._modified_forward(**kwargs)
        bsz = output.shape[0]
        q_len = output.shape[1]
        # (bsz, num_heads, adapter_len, head_dim)
        adapter_k = (
            self.model.k_proj(self.adaption_prompt)
            .view(1, self.adapter_len, self.model.num_heads, self.model.head_dim)
            .repeat(bsz, 1, 1, 1)
            .transpose(1, 2)
        )
        # (bsz, num_heads, adapter_len, head_dim)
        adapter_v = (
            self.model.v_proj(self.adaption_prompt)
            .view(1, self.adapter_len, self.model.num_heads, self.model.head_dim)
            .repeat(bsz, 1, 1, 1)
            .transpose(1, 2)
        )
        # (bsz, num_heads, q_len, adapter_len)
        scores = torch.matmul(query_states, adapter_k.transpose(2, 3)) / math.sqrt(self.model.head_dim)
        # Upcast attention to fp32
        # (bsz, num_heads, q_len, adapter_len)
        scores = self.adaption_gate * F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # (bsz, q_len, num_heads * head_dim)
        adapter_output = torch.matmul(scores, adapter_v).transpose(1, 2).reshape(bsz, q_len, -1)
        output = output + adapter_output
        # (bsz, q_len, hidden_size)
        output = self.model.o_proj(output)

        return output, None, None
