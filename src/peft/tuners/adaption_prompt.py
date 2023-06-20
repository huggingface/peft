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

import math
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.utils.config import PeftConfig, PeftType
from peft.utils.other import _freeze_adapter, _get_submodules


def llama_rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.

    This function was duplicated verbatim from:
    https://github.com/huggingface/transformers/blob/1de8ce9ee1191ba761a593ac15d9ccbf5851bfc5/src/transformers/models/llama/modeling_llama.py#L126

    This was done to eliminate the Llama transformers implementation as a dependency of this file. Note that some other
    functions were also adapted from the transformers implementation but were modified.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def llama_apply_rotary_pos_emb(q, cos, sin, position_ids):
    """
    Apply rotary position embedding to query states in the Llama model.

    This function was adapted from:
    https://github.com/huggingface/transformers/blob/1de8ce9ee1191ba761a593ac15d9ccbf5851bfc5/src/transformers/models/llama/modeling_llama.py#L133

    It was modified to remove unnecessary processing of key states.
    """
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    q_embed = (q * cos) + (llama_rotate_half(q) * sin)
    return q_embed


def llama_compute_query_states(model: nn.Module, **kwargs) -> torch.Tensor:
    """
    Compute query states for Llama models specifically.

    They need to be recomputed as the forward() method of the original LlamaModel in the transformers library does not
    return them. See the related discussion in the PR: https://github.com/huggingface/peft/pull/268
    """
    hidden_states = kwargs.get("hidden_states")
    position_ids = kwargs.get("position_ids")
    past_key_value = kwargs.get("past_key_value")
    bsz, q_len, _ = hidden_states.size()
    query_states = model.q_proj(hidden_states).view(bsz, q_len, model.num_heads, model.head_dim).transpose(1, 2)
    value_states = model.v_proj(hidden_states).view(bsz, q_len, model.num_heads, model.head_dim).transpose(1, 2)

    seq_len = q_len
    if past_key_value is not None:
        seq_len += past_key_value[0].shape[-2]
    cos, sin = model.rotary_emb(value_states, seq_len=seq_len)

    return llama_apply_rotary_pos_emb(query_states, cos, sin, position_ids)


# Contains the config that is specific to a transformers model type.
ModelTypeConfig = namedtuple(
    "ModelTypeConfig", ["compute_query_states", "target_modules", "k_proj_layer", "v_proj_layer", "o_proj_layer"]
)
# Mapping of transformers model types to their specific configuration.
TRANSFORMERS_MODEL_CONFIG = {
    "llama": ModelTypeConfig(
        compute_query_states=llama_compute_query_states,
        target_modules="self_attn",
        k_proj_layer="k_proj",
        v_proj_layer="v_proj",
        o_proj_layer="o_proj",
    ),
}


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
    if model.config.model_type not in TRANSFORMERS_MODEL_CONFIG:
        raise ValueError("Unsupported model type for adaption prompt: '{model.config.model_type}'.")

    model_config = TRANSFORMERS_MODEL_CONFIG[model.config.model_type]

    if peft_config.target_modules is None:
        peft_config.target_modules = model_config.target_modules

    return peft_config


class AdaptionPromptModel(nn.Module):
    """
    Implements adaption prompts as described in https://arxiv.org/pdf/2303.16199.pdf.

    The top L attention modules are replaced with AdaptedAttention modules that wrap the original ones, but insert
    trainable prompts with gates (for zero init).

    Notes on the multi-adapter pattern:
    - We store the states of different adapters by keeping a dictionary of AdaptedAttention modules indexed by adapter
      name.
    - Every time we switch adapters, we remove the modules of the currently active adapter from the model, store them
      in the dictionary, and replace them with the modules of the new adapter.
    - To avoid duplicated and potentially inconsistent state, the currently active adapter is always removed from the
      dictionary.
    - Disabling the adapter would also result in the modules being removed from the model.
    """

    def __init__(self, model, configs: Dict, adapter_name: str):
        super().__init__()
        self.model = model
        # Store adapter configs by name.
        self._configs: Dict[str, AdaptionPromptConfig] = {}
        # Store lists of the parents of the affected attention modules by adapter name.
        # We keep references to the parents so we can swap the adapters in-and-out of the model.
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
                par, _, _ = _get_submodules(self.model, name)
                parents.append(par)
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
            attn = AdaptedAttention(
                model_type=self.model.config.model_type,
                adapter_len=config.adapter_len,
                model=getattr(par, config.target_modules),
            )
            setattr(par, config.target_modules, attn)

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

    def __init__(self, model_type: str, adapter_len: int, model):
        """
        Initialize object.

        Args:
            model_type: The transformer model type. This is used to retrieve the right method to
                compute query states.
            adapter_len: The length of the adaption prompt to insert.
            model: The original transformer attention module that is being wrapped.
        """
        assert not isinstance(model, AdaptedAttention)
        super().__init__()
        self.model_type = model_type
        self.model = model
        self.adapter_len = adapter_len
        # Assume all parameters of the attention model we are wrapping are on the same device.
        device = next(model.parameters()).device
        # Don't think this was specified in the paper, but we follow the official repo which used an Embedding
        # which initializes the tokens with standard normal values.
        # https://github.com/ZrrSkywalker/LLaMA-Adapter/blob/41c3546fe1997ab8a65809dc8d8f9252b19d9faf/llama/model.py#L234
        # (bsz, adapter_len, hidden_size)
        target_dtype = (
            model.q_proj.weight.dtype if model.q_proj.weight.dtype not in [torch.int8, torch.uint8] else torch.float32
        )
        self.adaption_prompt = nn.Parameter(
            torch.empty(1, adapter_len, self.model.hidden_size, device=device, dtype=target_dtype).normal_()
        )
        # Initialize the gate to 0 as this is "zero-init".
        self.adaption_gate = nn.Parameter(torch.zeros(1, device=device, dtype=target_dtype))

    def forward(self, **kwargs):
        """
        Forward pass for the adapter which wraps the original LlamaAttention module.

        "Official" paper implementation:
        https://github.com/ZrrSkywalker/LLaMA-Adapter/blob/41c3546fe1997ab8a65809dc8d8f9252b19d9faf/llama/model.py#L141

        Args:
            kwargs: See the original LlamaAttention module.
        """
        if kwargs.get("output_attention", False):
            raise NotImplementedError("output_attention is not currently supported.")

        output, _, past_key_value = self.model(**kwargs)
        bsz = output.shape[0]
        q_len = output.shape[1]
        embed_dim = output.shape[2]
        k_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].k_proj_layer
        v_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].v_proj_layer
        o_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].o_proj_layer

        if k_proj_layer == v_proj_layer:
            _, key, value = getattr(self.model, k_proj_layer)(self.adaption_prompt).split(embed_dim, dim=2)
        else:
            key = getattr(self.model, k_proj_layer)(self.adaption_prompt)
            value = getattr(self.model, v_proj_layer)(self.adaption_prompt)
        # (bsz, num_heads, adapter_len, head_dim)
        adapter_k = (
            key.view(1, self.adapter_len, self.model.num_heads, self.model.head_dim)
            .repeat(bsz, 1, 1, 1)
            .transpose(1, 2)
        )
        # (bsz, num_heads, adapter_len, head_dim)
        adapter_v = (
            value.view(1, self.adapter_len, self.model.num_heads, self.model.head_dim)
            .repeat(bsz, 1, 1, 1)
            .transpose(1, 2)
        )

        # Recompute query states.
        compute_query_states = TRANSFORMERS_MODEL_CONFIG[self.model_type].compute_query_states
        # (bsz, num_heads, q_len, head_dim)
        query_states = compute_query_states(model=self.model, **kwargs)

        previous_dtype = query_states.dtype
        # (bsz, num_heads, q_len, adapter_len)
        scores = torch.matmul(query_states, adapter_k.transpose(2, 3).to(previous_dtype)) / math.sqrt(
            self.model.head_dim
        )
        # Upcast attention to fp32
        # (bsz, num_heads, q_len, adapter_len)
        scores = self.adaption_gate * F.softmax(scores, dim=-1, dtype=torch.float32).to(previous_dtype)
        # (bsz, q_len, num_heads * head_dim)
        adapter_output = torch.matmul(scores, adapter_v).transpose(1, 2).reshape(bsz, q_len, -1)
        # (bsz, q_len, hidden_size)
        if o_proj_layer is not None:
            adapter_output = getattr(self.model, o_proj_layer)(adapter_output)

        # Add adaption prompt output to original output.
        output = output + adapter_output

        # Restore original dtype.
        output = output.to(previous_dtype)
        return output, None, past_key_value
