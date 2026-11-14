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
"""
Runtime check for training on tokens whose input embedding row is all zero and frozen.

Some checkpoints ship special tokens (e.g. chat template tags) that were never trained, so their embedding rows are
zero. With the embedding frozen, as is the default for most PEFT methods, the model cannot learn to use these tokens
and the training loss gives no hint. Zero rows are common and usually harmless, so a static check on the weights would
mostly produce false alarms. Instead, a forward hook on the input embedding looks at the first few training batches and
only reports tokens that actually occur in the data.

The hook only reads the input ids and the embedding output, never the weight, so it also works when the weight is
sharded, e.g. with FSDP. It does no collective communication: each rank checks its own batches.
"""

import warnings
from typing import Literal

import torch
from torch import nn

from .other import TrainableTokensWrapper
from .warning import PeftWarning


# Number of training batches to inspect before the hook removes itself.
MAX_CHECKED_BATCHES = 20
# Rows whose absolute values are all below this count as zero. Some checkpoints store untrained rows as tiny values
# (around 1e-37) instead of exact zeros.
ZERO_TOLERANCE = 1e-6
# Config attributes holding token ids that are expected to have a zero row: padding, and multimodal placeholders whose
# embedding is replaced by encoder features after the lookup.
IGNORED_TOKEN_ID_ATTRIBUTES = (
    "pad_token_id",
    "image_token_id",
    "image_token_index",
    "video_token_id",
    "video_token_index",
    "audio_token_id",
    "audio_token_index",
)


class ZeroEmbeddingCheck:
    """
    Forward hook on the input embedding that reports tokens with an all-zero, frozen embedding row in the training
    data.

    A class is used instead of a closure so that the model stays picklable.
    """

    def __init__(
        self, mode: Literal["warn", "raise"], ignored_token_ids: set[int], embedding_name: str, output_name: str | None
    ):
        self.mode = mode
        self.ignored_token_ids = ignored_token_ids
        self.embedding_name = embedding_name
        self.output_name = output_name
        self.num_checked_batches = 0
        self.handle = None

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def __call__(self, module: nn.Module, args: tuple, output) -> None:
        if not (module.training and torch.is_grad_enabled()) or not args:
            return
        input_ids = args[0]
        if (
            not isinstance(input_ids, torch.Tensor)
            or not isinstance(output, torch.Tensor)
            or input_ids.is_floating_point()
            or output.shape[:-1] != input_ids.shape
        ):
            return

        trainable_ids = set()
        if isinstance(module, TrainableTokensWrapper):
            # only the rows listed in trainable_token_indices are updated
            if not module.disable_adapters:
                for adapter_name in module.active_adapters:
                    trainable_ids.update(module.token_adapter.token_indices.get(adapter_name, []))
        elif output.grad_fn is not None:
            # The output depends on trainable parameters, e.g. with modules_to_save, LoRA on the embedding, or full
            # fine-tuning, so every row can be updated. Checking grad_fn instead of requires_grad matters: activation
            # checkpointing calls output.requires_grad_() on the frozen embedding output, which sets requires_grad but
            # leaves grad_fn empty.
            self.remove()
            return

        self.num_checked_batches += 1
        with torch.no_grad():
            is_zero = output.abs().amax(dim=-1) < ZERO_TOLERANCE
            zero_ids = set(torch.unique(input_ids[is_zero]).tolist())
        frozen_zero_ids = sorted(zero_ids - self.ignored_token_ids - trainable_ids)

        if frozen_zero_ids:
            self.remove()
            self._report(frozen_zero_ids)
        elif self.num_checked_batches >= MAX_CHECKED_BATCHES:
            self.remove()

    def _report(self, token_ids: list[int]) -> None:
        indices = f"'{self.embedding_name}': {token_ids}"
        if self.output_name is not None:
            indices += f", '{self.output_name}': {token_ids}"
        msg = (
            f"The training data contains the token ids {token_ids}, whose input embedding rows are all zero and "
            "frozen. This usually means that these tokens were never trained, e.g. special tokens of a chat template. "
            "As long as these rows stay frozen, the model is unlikely to learn to use or generate these tokens, even "
            "if the training loss goes down. To train them, pass "
            f"`trainable_token_indices={{{indices}}}` to the PEFT config (the output embedding is only needed if it is "
            "not tied to the input embedding), ideally after initializing these rows, e.g. with the mean embedding of "
            "related tokens. Alternatively, add the embedding layer to `target_modules` or `modules_to_save`. If this "
            "is expected, pass `attach_checks=False` to `get_peft_model` to disable this check."
        )
        if self.mode == "raise":
            raise ValueError(msg)
        warnings.warn(msg, PeftWarning)


def _get_ignored_token_ids(model: nn.Module, embedding: nn.Module) -> set[int]:
    ignored_token_ids = set()
    padding_idx = next((m.padding_idx for m in embedding.modules() if isinstance(m, nn.Embedding)), None)
    if padding_idx is not None:
        ignored_token_ids.add(padding_idx)

    config = getattr(model, "config", None)
    for cfg in (config, getattr(config, "text_config", None)):
        for attribute in IGNORED_TOKEN_ID_ATTRIBUTES:
            value = getattr(cfg, attribute, None)
            if isinstance(value, int):
                ignored_token_ids.add(value)
            elif isinstance(value, (list, tuple)):
                ignored_token_ids.update(v for v in value if isinstance(v, int))
    return ignored_token_ids


def _get_module_name(model: nn.Module, module: nn.Module | None) -> str | None:
    if module is None:
        return None
    return next((name.rpartition(".")[-1] for name, m in model.named_modules() if m is module), None)


def attach_zero_embedding_check(model: nn.Module, mode: Literal["warn", "raise"]) -> None:
    """
    Attach a [`ZeroEmbeddingCheck`] to the input embedding of `model`, if the model has one.

    The hook stays inactive during evaluation and removes itself after the first report or after `MAX_CHECKED_BATCHES`
    training batches. It is attached to the embedding as it is when the PEFT model is created; adapters that are added
    later and wrap the embedding are not taken into account.
    """
    try:
        embedding = model.get_input_embeddings()
    except (AttributeError, NotImplementedError):
        return
    if not isinstance(embedding, nn.Module):
        return

    try:
        output_embedding = model.get_output_embeddings()
    except (AttributeError, NotImplementedError):
        output_embedding = None

    remove_zero_embedding_checks(embedding)
    hook = ZeroEmbeddingCheck(
        mode=mode,
        ignored_token_ids=_get_ignored_token_ids(model, embedding),
        embedding_name=_get_module_name(model, embedding) or "embed_tokens",
        output_name=_get_module_name(model, output_embedding),
    )
    hook.handle = embedding.register_forward_hook(hook)


def remove_zero_embedding_checks(model: nn.Module) -> None:
    """Remove all [`ZeroEmbeddingCheck`] hooks from `model` and its submodules."""
    for module in model.modules():
        for hook in list(module._forward_hooks.values()):
            if isinstance(hook, ZeroEmbeddingCheck):
                hook.remove()
