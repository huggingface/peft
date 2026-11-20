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
Runtime checks that `get_peft_model` can attach to a model to catch problems that do not show up in the training loss.

Each check is a [`RuntimeCheck`]: a forward hook on one module of the model, which inspects the first training batches
and removes itself afterwards. Currently there is one check:

- [`ZeroEmbeddingCheck`]: training on tokens whose input embedding row is all zero and frozen. Some checkpoints ship
  special tokens (e.g. chat template tags) that were never trained, so their embedding rows are zero. With the
  embedding frozen, as is the default for most PEFT methods, the model cannot learn to use these tokens and the
  training loss gives no hint. Zero rows are common and usually harmless, so a static check on the weights would mostly
  produce false alarms. Instead, the hook looks at the first few training batches and only reports tokens that actually
  occur in the data. It only reads the input ids and the embedding output, never the weight, so it also works when the
  weight is sharded, e.g. with FSDP. It does no collective communication: each rank checks its own batches.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import nn

from .other import TrainableTokensWrapper
from .warning import PeftWarning


# Number of training batches to inspect before a check removes itself.
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


class RuntimeCheck(ABC):
    """
    Base class of the runtime checks attached by [`attach_runtime_checks`].

    A check is a forward hook on the module returned by [`~RuntimeCheck.get_target_module`]. It is only active while
    training, and it removes itself once it has reported a problem or has seen enough batches. Checks are classes
    instead of closures so that the model stays picklable.

    Subclasses implement `get_target_module`, `create` and `__call__`.
    """

    def __init__(self, mode: Literal["warn", "raise"]):
        self.mode = mode
        self.handle = None

    @staticmethod
    @abstractmethod
    def get_target_module(model: nn.Module) -> nn.Module | None:
        """The module to hook in `model`, or `None` if the check does not apply to it."""

    @classmethod
    @abstractmethod
    def create(cls, model: nn.Module, mode: Literal["warn", "raise"]) -> "RuntimeCheck":
        """A new, not yet registered check for `model`."""

    @abstractmethod
    def __call__(self, module: nn.Module, args: tuple, kwargs: dict, output) -> None:
        """The forward hook, registered with `with_kwargs=True`."""

    def register(self, model: nn.Module) -> None:
        """Hook the check into the current target module of `model`, replacing an earlier registration."""
        self.remove()
        target = self.get_target_module(model)
        if target is not None:
            self.handle = target.register_forward_hook(self, with_kwargs=True)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def report(self, msg: str) -> None:
        if self.mode == "raise":
            raise RuntimeError(msg)
        warnings.warn(msg, PeftWarning)


def _get_input_embeddings(model: nn.Module) -> nn.Module | None:
    try:
        embedding = model.get_input_embeddings()
    except (AttributeError, NotImplementedError):
        return None
    return embedding if isinstance(embedding, nn.Module) else None


def _get_ignored_token_ids(model: nn.Module, embedding: nn.Module) -> set[int]:
    ignored_token_ids = set()
    padding_idx = next((m.padding_idx for m in embedding.modules() if isinstance(m, nn.Embedding)), None)
    if padding_idx is not None:
        ignored_token_ids.add(padding_idx)

    # The tokenizer is not available here, so the model config is the best source. Transformers models usually store
    # these ids in their config (pad_token_id, and image_token_id etc. for multimodal models), but it is not
    # guaranteed. A missing id can only lead to a false positive, which is why the check is opt-in.
    config = getattr(model, "config", None)
    configs = [c for c in (config, getattr(config, "text_config", None)) if c is not None]
    for config in configs:
        for attribute in IGNORED_TOKEN_ID_ATTRIBUTES:
            value = getattr(config, attribute, None)
            if isinstance(value, int):
                ignored_token_ids.add(value)
            elif isinstance(value, (list, tuple)):
                ignored_token_ids.update(v for v in value if isinstance(v, int))
    return ignored_token_ids


class ZeroEmbeddingCheck(RuntimeCheck):
    """
    Forward hook on the input embedding that reports tokens with an all-zero, frozen embedding row in the training
    data.
    """

    def __init__(self, mode: Literal["warn", "raise"], ignored_token_ids: set[int]):
        super().__init__(mode)
        self.ignored_token_ids = ignored_token_ids
        self.num_checked_batches = 0

    @staticmethod
    def get_target_module(model: nn.Module) -> nn.Module | None:
        return _get_input_embeddings(model)

    @classmethod
    def create(cls, model: nn.Module, mode: Literal["warn", "raise"]) -> "ZeroEmbeddingCheck":
        embedding = _get_input_embeddings(model)
        ignored = _get_ignored_token_ids(model, embedding) if embedding is not None else set()
        return cls(mode=mode, ignored_token_ids=ignored)

    def __call__(self, module: nn.Module, args: tuple, kwargs: dict, output) -> None:
        if not (module.training and torch.is_grad_enabled()):
            return
        # Every training batch counts, including skipped ones, so that the check always expires.
        self.num_checked_batches += 1
        if self.num_checked_batches >= MAX_CHECKED_BATCHES:
            self.remove()

        # The input embedding is normally called as `embedding(input_ids)`, but it may also be called with keyword
        # arguments only (e.g. `input=` for nn.Embedding), or be wrapped in a module whose output is not a plain
        # tensor. The check only makes sense for integer ids with one embedding vector per id, so anything else is
        # skipped. The conditions are ordered so that `output.shape` is only read if `output` is a tensor; nothing
        # here can raise.
        input_ids = args[0] if args else next(iter(kwargs.values()), None)
        if not isinstance(output, torch.Tensor):
            return
        if not isinstance(input_ids, torch.Tensor) or input_ids.is_floating_point():
            return
        if output.shape[:-1] != input_ids.shape:
            return

        if getattr(module, "disable_adapters", False):
            # e.g. a reference forward pass in the disable_adapter() context, the adapters may still train these rows
            return

        # local import to avoid circular import
        from peft.tuners.trainable_tokens.layer import TrainableTokensLayer

        layer = module.token_adapter if isinstance(module, TrainableTokensWrapper) else module
        trainable_ids = set()
        if isinstance(layer, TrainableTokensLayer):
            if any(param.requires_grad for param in layer.base_layer.parameters()):
                # the whole embedding is trained, e.g. full fine-tuning on top of trainable tokens
                self.remove()
                return
            # otherwise only the rows listed in the token indices are updated
            for adapter_name in layer.active_adapters:
                trainable_ids.update(layer.token_indices.get(adapter_name, []))
        elif output.grad_fn is not None:
            # The output depends on trainable parameters, e.g. with modules_to_save, LoRA on the embedding, or full
            # fine-tuning, so every row can be updated. Checking grad_fn instead of requires_grad matters: activation
            # checkpointing calls output.requires_grad_() on the frozen embedding output, which sets requires_grad but
            # leaves grad_fn empty.
            self.remove()
            return

        with torch.no_grad():
            is_zero = output.abs().amax(dim=-1) < ZERO_TOLERANCE
            zero_ids = set(torch.unique(input_ids[is_zero]).tolist())
        frozen_zero_ids = sorted(zero_ids - self.ignored_token_ids - trainable_ids)

        if frozen_zero_ids:
            self.remove()
            self.report(
                f"The training data contains the token ids {frozen_zero_ids}, whose input embedding rows are all zero "
                "and frozen. This usually means that these tokens were never trained, e.g. special tokens of a chat "
                "template. As long as these rows stay frozen, the model is unlikely to learn to use or generate these "
                "tokens, even if the training loss goes down. To train them, pass these token ids to "
                "`trainable_token_indices` in the `LoraConfig`, for the input embedding and, if it is not tied to the "
                "input embedding, the output embedding too. Ideally, initialize these rows first, e.g. with the mean "
                "embedding of related tokens. Alternatively, or if you're not using LoRA, add the embedding layer to "
                "`target_modules` or `modules_to_save` if the PEFT method supports it. If these tokens are not "
                "supposed to be learned, e.g. because they are masked out of the loss, pass `runtime_checks=False` to "
                "`get_peft_model` to disable this check."
            )


# All checks attached by `attach_runtime_checks`. A new check only needs to subclass `RuntimeCheck` and be added here.
RUNTIME_CHECKS: tuple[type[RuntimeCheck], ...] = (ZeroEmbeddingCheck,)


def _find_runtime_checks(model: nn.Module) -> list[RuntimeCheck]:
    return [
        hook for module in model.modules() for hook in module._forward_hooks.values() if isinstance(hook, RuntimeCheck)
    ]


def attach_runtime_checks(model: nn.Module, mode: Literal["warn", "raise"]) -> None:
    """
    Attach every check in `RUNTIME_CHECKS` that applies to `model`, replacing checks attached earlier.

    The checks stay inactive during evaluation and remove themselves after the first report or after a few training
    batches.
    """
    remove_runtime_checks(model)
    for check_cls in RUNTIME_CHECKS:
        if check_cls.get_target_module(model) is not None:
            check_cls.create(model, mode).register(model)


def move_runtime_checks(model: nn.Module) -> None:
    """
    Move the attached checks to the current target modules of `model`.

    This is needed when an adapter added after the checks were attached wraps a target module, e.g. the input
    embedding, since a check has to see the output of the wrapper, not of the frozen base layer inside it.
    """
    for check in _find_runtime_checks(model):
        check.register(model)


def remove_runtime_checks(model: nn.Module) -> None:
    """Remove all [`RuntimeCheck`] hooks from `model` and its submodules."""
    for check in _find_runtime_checks(model):
        check.remove()
