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

from __future__ import annotations

from contextlib import contextmanager
from functools import partial

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_DUMMY_TARGET_MODULES_MAPPING,
    get_quantization_kwargs,
    resolve_quantization_backend,
)

from .layer import Conv1d, Conv2d, Conv3d, DummyLayer, Embedding, Linear


def _adapter_names_pre_forward_hook(target, args, kwargs, adapter_names):
    # pre-forward hook to inject the adapter_names argument when using mixed adapter batches inference
    kwargs["adapter_names"] = adapter_names
    return args, kwargs


def _get_encoder(model: nn.Module) -> nn.Module | None:
    """Check if the model has an encoder and if it has, returns it; otherwise returns None"""
    if not hasattr(model, "get_encoder"):
        return None

    encoder = model.get_encoder()
    # https://github.com/huggingface/transformers/pull/42156
    # new logic in transformers v5: all PretrainedModels return a model here, but it is self if there is no encoder
    if encoder is model:
        return None
    return encoder


def _get_tuner_layer_class(target_base_layer: torch.nn.Module) -> type[DummyLayer] | None:
    layer_cls: type[DummyLayer] | None = None
    if isinstance(target_base_layer, (torch.nn.Linear, Conv1D)):
        layer_cls = Linear
    elif isinstance(target_base_layer, torch.nn.Conv1d):
        layer_cls = Conv1d
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        layer_cls = Conv2d
    elif isinstance(target_base_layer, torch.nn.Conv3d):
        layer_cls = Conv3d
    elif isinstance(target_base_layer, torch.nn.Embedding):
        layer_cls = Embedding
    elif (quant_backend := resolve_quantization_backend(target_base_layer)) is not None:
        layer_cls = {"linear": Linear}.get(quant_backend.layer_type)

    return layer_cls


class DummyModel(BaseTuner):
    """
    Creates a Dummy adapter model from a pretrained model.

    The Dummy method is not a research method. It is a minimal, deliberately trivial tuner that serves as a template
    for contributors who want to add a new PEFT method and as a test vehicle for PEFT features that any tuner should
    support (saving, loading, merging, quantization, ...).

    Its implementation is essentially a plain LoRA without any of the optimizations or extras that LoRA offers, see
    [`DummyConfig`] for details, which also contains a usage example.

    Args:
        model (`torch.nn.Module`):
            The model to which the adapter tuner layers will be attached.
        peft_config ([`DummyConfig`]): The configuration of the Dummy model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The Dummy model.

    **Attributes**:
        - **model** ([`~torch.nn.Module`]) -- The model to be adapted.
        - **peft_config** ([`DummyConfig`]): The configuration of the Dummy model.
    """

    prefix: str = "dummy_"
    tuner_layer_cls = DummyLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_DUMMY_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        dummy_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        kwargs = {}
        kwargs.update(get_quantization_kwargs(self))

        # If it is not a DummyLayer, create a new module, else update it with new adapters
        if isinstance(target, DummyLayer):
            target.update_layer(adapter_name, config=dummy_config, **kwargs)
        else:
            new_module = self._create_new_module(dummy_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(dummy_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        layer_cls = _get_tuner_layer_class(target_base_layer)
        if layer_cls is None:
            raise TypeError(
                f"Target module {target} is not supported. Currently, `torch.nn.Linear`, "
                "`transformers.pytorch_utils.Conv1D`, `torch.nn.Conv1d/2d/3d` (optionally quantized), and"
                " `torch.nn.Embedding` are supported."
            )

        # Conv1D layers store the weight like (fan_in, fan_out) and therefore need transposed delta weights. This is
        # determined automatically from the base layer type, there is no user-facing config option for it.
        if isinstance(target_base_layer, Conv1D):
            kwargs["fan_in_fan_out"] = True

        new_module = layer_cls(target, adapter_name, config=dummy_config, **kwargs)
        return new_module

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        # If adapter_names is passed as an argument, we inject it into the forward arguments of the Dummy layers.
        adapter_names = kwargs.pop("adapter_names", None)
        if adapter_names is None:
            # nothing to do
            yield
            return

        if self.training:
            raise ValueError("Cannot pass `adapter_names` when the model is in training mode.")

        # Check that users only passed actually existing adapters.
        # Note: We cannot do this on the layer level, as each individual layer may not have each adapter. Still, we want
        # to check that there is at least one layer with the given name, or else something like typos can easily slip.
        expected_adapters = set()
        for layer in self.modules():
            if isinstance(layer, DummyLayer):
                expected_adapters |= layer.dummy_lora_A.keys()
        unique_adapters = {name for name in adapter_names if name != "__base__"}
        unexpected_adapters = unique_adapters - expected_adapters
        if unexpected_adapters:
            raise ValueError(f"Trying to infer with non-existing adapter(s): {', '.join(sorted(unexpected_adapters))}")

        # deal with beam search
        original_adapter_names = adapter_names[:]
        num_beams = kwargs.get("num_beams", None)
        uses_beam_search = isinstance(num_beams, int) and (num_beams > 1)
        if uses_beam_search:
            if not isinstance(adapter_names, (list, tuple)):
                raise TypeError(f"Got adapter names of type {type(adapter_names)}, expected a list of str.")
            # When there is beam search, the inputs are repeated n times, thus we repeat each adapter name n times and
            # then flatten the nested list. For encoder-decoder models, this extended list should not be applied to the
            # encoder part. Further below, the original argument is thus restored for the encoder.
            adapter_names = [n for n in adapter_names for _ in range(num_beams)]

        hook_handles = []
        for module in self.modules():
            if isinstance(module, DummyLayer):
                pre_forward = partial(_adapter_names_pre_forward_hook, adapter_names=adapter_names)
                handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                hook_handles.append(handle)

        encoder = _get_encoder(self.model)
        if uses_beam_search and (encoder is not None):
            # For encoder-decoder models, even when applying beam search, the encoder part of the model should not use
            # the extended adapter_names. This is because the encoder still uses the original, non-extended samples.
            for module in encoder.modules():
                if isinstance(module, DummyLayer):
                    # Add another hook to overwrite the kwargs with the original adapter names -- this is easier than
                    # trying to exclude the encoder.
                    pre_forward = partial(_adapter_names_pre_forward_hook, adapter_names=original_adapter_names)
                    handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                    hook_handles.append(handle)

        yield

        for handle in hook_handles:
            handle.remove()
