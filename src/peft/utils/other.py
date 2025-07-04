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
from __future__ import annotations

import copy
import functools
import inspect
import os
import re
import warnings
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, Optional, Union

import accelerate
import torch
from accelerate import FullyShardedDataParallelPlugin
from accelerate.hooks import add_hook_to_module, remove_hook_from_module
from accelerate.utils import is_npu_available, is_xpu_available
from huggingface_hub import file_exists
from huggingface_hub.errors import EntryNotFoundError, HFValidationError
from packaging import version
from safetensors.torch import storage_ptr, storage_size
from transformers import PreTrainedModel

from ..import_utils import is_auto_gptq_available, is_gptqmodel_available, is_torch_tpu_available
from .constants import (
    CONFIG_NAME,
    EMBEDDING_LAYER_NAMES,
    INCLUDE_LINEAR_LAYERS_SHORTHAND,
    SAFETENSORS_WEIGHTS_NAME,
    TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_C3A_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_FOURIERFT_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    TRANSFORMERS_MODELS_TO_RANDLORA_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_VBLORA_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING,
    WEIGHTS_NAME,
    bloom_model_postprocess_past_key_value,
    starcoder_model_postprocess_past_key_value,
)


mlu_available = False
if version.parse(accelerate.__version__) >= version.parse("0.29.0"):
    from accelerate.utils import is_mlu_available

    mlu_available = is_mlu_available()


__all__ = [
    "CONFIG_NAME",
    "EMBEDDING_LAYER_NAMES",
    "INCLUDE_LINEAR_LAYERS_SHORTHAND",
    "SAFETENSORS_WEIGHTS_NAME",
    "TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_C3A_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_FOURIERFT_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING",
    "TRANSFORMERS_MODELS_TO_RANDLORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_VBLORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING",
    "WEIGHTS_NAME",
    "bloom_model_postprocess_past_key_value",
    "starcoder_model_postprocess_past_key_value",
]


# Get current device name based on available devices
def infer_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    elif mlu_available:
        return "mlu"
    elif is_xpu_available():
        return "xpu"
    elif is_npu_available():
        return "npu"
    return "cpu"


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None):
    r"""
    Note this method only works for `transformers` models.

    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32 4- Freezing the base model layers to ensure they are not updated during training


    Args:
        model (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
        use_gradient_checkpointing (`bool`, *optional*, defaults to `True`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        gradient_checkpointing_kwargs (`dict`, *optional*, defaults to `None`):
            Keyword arguments to pass to the gradient checkpointing function, please refer to the documentation of
            `torch.utils.checkpoint.checkpoint` for more details about the arguments that you can pass to that method.
            Note this is only available in the latest transformers versions (> 4.34.1).
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    is_gptq_quantized = getattr(model, "quantization_method", None) == "gptq"
    is_aqlm_quantized = getattr(model, "quantization_method", None) == "aqlm"
    is_eetq_quantized = getattr(model, "quantization_method", None) == "eetq"
    is_torchao_quantized = getattr(model, "quantization_method", None) == "torchao"
    is_hqq_quantized = getattr(model, "quantization_method", None) == "hqq" or getattr(model, "hqq_quantized", False)

    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {}

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    if (
        not is_gptq_quantized
        and not is_aqlm_quantized
        and not is_eetq_quantized
        and not is_hqq_quantized
        and not is_torchao_quantized
    ):
        # cast all non INT8 parameters to fp32
        for param in model.parameters():
            if (
                (param.dtype == torch.float16) or (param.dtype == torch.bfloat16)
            ) and param.__class__.__name__ != "Params4bit":
                param.data = param.data.to(torch.float32)

    if (
        loaded_in_kbit
        or is_gptq_quantized
        or is_aqlm_quantized
        or is_eetq_quantized
        or is_hqq_quantized
        or is_torchao_quantized
    ) and use_gradient_checkpointing:
        # When having `use_reentrant=False` + gradient_checkpointing, there is no need for this hack
        if "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]:
            # For backward compatibility
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # To support older transformers versions, check if the model supports gradient_checkpointing_kwargs
        _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
            inspect.signature(model.gradient_checkpointing_enable).parameters
        )

        if not _supports_gc_kwargs and len(gradient_checkpointing_kwargs) > 0:
            warnings.warn(
                "gradient_checkpointing_kwargs is not supported in this version of transformers. The passed kwargs will be ignored."
                " if you want to use that feature, please upgrade to the latest version of transformers.",
                FutureWarning,
            )

        gc_enable_kwargs = (
            {} if not _supports_gc_kwargs else {"gradient_checkpointing_kwargs": gradient_checkpointing_kwargs}
        )

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable(**gc_enable_kwargs)
    return model


# copied from transformers.models.bart.modeling_bart
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`): input ids
        pad_token_id (`int`): The id of the `padding` token.
        decoder_start_token_id (`int`): The id of the `start` token.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class AuxiliaryTrainingWrapper(torch.nn.Module):
    """Wrap a specific module so that it can be trained and saved in a way that is tangential to how
    PEFT normally works, e.g. fully training a classification layer instead of using an adapter.

    """

    def __init__(self, module_to_save, adapter_name, **kwargs):
        """Extra kwargs will be passed to `self.init_modules` and `self.update`."""
        super().__init__()
        self.original_module = module_to_save
        self._active_adapter = [adapter_name]
        self._disable_adapters = False
        self._adapters = set()

        self.init_modules(adapter_name, **kwargs)

        self.update(adapter_name, **kwargs)
        self.check_module()

    def init_modules(self, adapter_name, **kwargs):
        """A place to initialize PyTorch modules in `__init__` before the call to `self.update()`."""
        raise NotImplementedError

    def _error_message_name(self):
        """Returns a user friendly identifier for error messages, e.g. for type compatibility error messages from
        `check_module()` so that the user can backtrack where the error comes from. A generic "training wrapper" is
        less helpful than "modules_to_save", for example.
        """
        return "training wrapper"

    def check_module(self):
        """Perform some sanity checks on the module to ensure that it works"""
        # Try to anticipate some modules that users could try to target that would not work.
        # Note: It's not possible to check hasattr(module, "forward"), since that returns True for ModuleDict and
        # ModuleList, even though their forward methods cannot be called
        forbidden_classes = (torch.nn.ModuleDict, torch.nn.ModuleList, torch.nn.ParameterDict, torch.nn.ParameterList)
        if isinstance(self.original_module, forbidden_classes):
            cls_name = self.original_module.__class__
            raise TypeError(f"{self._error_message_name()} cannot be applied to modules of type {cls_name}")

        # local import to avoid circular import
        from peft.tuners.tuners_utils import BaseTunerLayer

        if isinstance(self.original_module, BaseTunerLayer):
            # e.g. applying a training wrapper to a lora layer makes no sense
            cls_name = self.original_module.__class__
            raise TypeError(f"{self._error_message_name()} cannot be applied to modules of type {cls_name}")

    @property
    def disable_adapters(self) -> bool:
        # use a property to ensure that disable_adapters is not set directly, instead use the enable_adapters method
        return self._disable_adapters

    @property
    def active_adapter(self) -> Union[list[str], str]:
        # use a property to ensure that active_adapter is not set directly, instead use the set_adapter method
        return self._active_adapter

    @property
    def active_adapters(self) -> list[str]:
        if isinstance(self._active_adapter, str):
            return [self._active_adapter]
        return self._active_adapter

    def _hasattr_wrapped(self, name, modules):
        """Infrastructure to enable the implementing class to delegate attributes to other modules.
        Returns True if the implementing class knows how to handle attribute `name`.

        Gets passed `modules` which is PyTorch's internal list of assigned modules from `nn.Module`.
        """
        return False

    def _getattr_wrapped(self, name, modules):
        """If `_hasattr_wrapped` returns True for `name`, then this function should return the corresponding
        value associated with `name`.
        """
        return None

    def __getattr__(self, name: str):
        # Note: This whole method may seem overly complex at first but PyTorch messes with __getattr__ in a way that
        # requires very careful handling to avoid infinite recursion.
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass

        if "_modules" not in self.__dict__:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Could not find the attribute the PyTorch way. So let's check if it's an attribute on the
        # original_module or the module further down (e.g., `modules_to_save[active_adapter]`).
        modules = self.__dict__["_modules"]
        if self.disable_adapters:
            return getattr(self.original_module, name)
        elif self._hasattr_wrapped(name, modules):
            return self._getattr_wrapped(name, modules)

        # For some reason, there is no module corresponding to the active adapter; this should normally not be
        # reached and exists as a failsafe (otherwise, a KeyError would be raised)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def update(self, adapter_name, **kwargs):
        """Called when this instance should be part of an adapter's training.
        Adds the given adapter to the list of adapters that this instance is training along with.

        Additional kwargs are expected to be the same kwargs that are also passed for initializing this class.
        """
        if adapter_name not in self._adapters:
            self._adapters.add(adapter_name)

    def _create_new_hook(self, old_hook):
        r"""
        Creates a new hook based on the old hook. Use it only if you know what you are doing !
        """
        old_hook_cls = getattr(accelerate.hooks, old_hook.__class__.__name__)
        old_hook_attr = old_hook.__dict__
        filtered_old_hook_attr = {}
        old_hook_init_signature = inspect.signature(old_hook_cls.__init__)
        for k in old_hook_attr.keys():
            if k in old_hook_init_signature.parameters:
                filtered_old_hook_attr[k] = old_hook_attr[k]
        new_hook = old_hook_cls(**filtered_old_hook_attr)
        return new_hook

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

    def _forward_wrapped(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError

    def _forward_wrapped_mixed_batch(
        self, x: torch.Tensor, active_adapter: str, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        raise NotImplementedError

    def _forward_wrapped_passthrough(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """The forward call when no adapter is involved in the forward computation, only the base model"""
        raise NotImplementedError

    def _mixed_batch_forward(
        self, input: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.

        SUPPORTED_MODULES = (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)

        module_names = ", ".join([module.__name__ for module in SUPPORTED_MODULES])

        if not isinstance(self.original_module, SUPPORTED_MODULES):
            raise TypeError(f"Mixed batching is only supported for the following modules: {module_names}.")

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []

        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        results = [0 for _ in range(len(input))]

        for i, active_adapter in enumerate(unique_adapters):
            sub_batch = input[sub_batch_indices_list[i]]

            if active_adapter == "__base__":
                output = self.original_module(sub_batch, *args, **kwargs)
            else:
                output = self._forward_wrapped_mixed_batch(sub_batch, active_adapter, *args, **kwargs)

            for index, j in enumerate(sub_batch_indices_list[i]):
                results[j] = output[index]

        return torch.stack(results)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters or any(adapter not in self._adapters for adapter in self.active_adapters):
            return self._forward_wrapped_passthrough(x, *args, **kwargs)

        if adapter_names is None:
            return self._forward_wrapped(x, *args, **kwargs)
        return self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)

    def enable_adapters(self, enabled: bool):
        """Toggle the enabling and disabling of adapters

        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
        if enabled:
            self._disable_adapters = False
        else:
            self._disable_adapters = True

    def set_adapter(self, adapter_names: Union[str, list[str]]):
        """Set the active adapter

        Args:
            adapter_name (str): The name of the adapter to set as active
        """
        if isinstance(adapter_names, str):
            self._active_adapter = adapter_names
        else:
            self._active_adapter = []
            for adapter_name in adapter_names:
                if adapter_name not in self._adapters:
                    raise ValueError(f"Adapter {adapter_name} not found in {self._adapters}")

                self._active_adapter.append(adapter_name)

    def delete_adapter(self, adapter_name: str, new_active_adapters: Optional[list[str]]) -> None:
        """Delete an adapter from the layer, set a new active adapter if necessary"""
        raise NotImplementedError

    def adapter_state_dict(self, adapter_name):
        """Return the state dict of this module for a given adapter."""
        raise NotImplementedError

    def adapter_state_dict_load_map(self, adapter_name):
        """Return a mapping from the key present in disk-loaded state dict
        and how it should be represented in the loaded model's state dict.

        The default should be a 1:1 mapping but it is important to define a mapping as it also serves as the
        ground-truth for which keys are supposed to be loaded from a saved state dict.
        """
        raise NotImplementedError

    def unload_and_optionally_merge_module(
        self, merge: bool, safe_merge: bool, adapter_names: Optional[list[str]]
    ) -> torch.nn.Module:
        """Handles unloading when called from PEFT models. Returns the wrapped module
        and handles merging onto the wrapped module if requested.
        """
        raise NotImplementedError


class ModulesToSaveWrapper(AuxiliaryTrainingWrapper):
    """Wraps a module that is supposed to be trained (i.e. `requires_grad_(True)`) and saved after training."""

    def __init__(self, module_to_save, adapter_name):
        super().__init__(module_to_save, adapter_name)

    def init_modules(self, adapter_name):
        # we treat each adapter separately, so we have multiple adapters, same (copied) module for each
        self.modules_to_save = torch.nn.ModuleDict({})

    def _error_message_name(self):
        return "modules_to_save"

    def _forward_wrapped(self, x, *args, **kwargs):
        if not self.active_adapters:
            return self._forward_wrapped_passthrough(x, *args, **kwargs)
        return self.modules_to_save[self.active_adapters[0]](x, *args, **kwargs)

    def _forward_wrapped_mixed_batch(self, x, active_adapter, *args, **kwargs):
        return self.modules_to_save[active_adapter](x, *args, **kwargs)

    def _forward_wrapped_passthrough(self, x, *args, **kwargs):
        return self.original_module(x, *args, **kwargs)

    def _hasattr_wrapped(self, name, modules):
        return self.active_adapters[0] in modules["modules_to_save"]

    def _getattr_wrapped(self, name, modules):
        return getattr(modules["modules_to_save"][self.active_adapters[0]], name)

    def update(self, adapter_name, **kwargs):
        super().update(adapter_name)

        context_manager = nullcontext()
        for _, param in self.original_module.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                import deepspeed

                context_manager = deepspeed.zero.GatheredParameters(self.original_module.parameters(), modifier_rank=0)
                break

        if adapter_name not in self.modules_to_save:
            with context_manager:
                self.modules_to_save[adapter_name] = copy.deepcopy(self.original_module)

        if hasattr(self.modules_to_save[adapter_name], "_hf_hook"):
            old_hook = self.modules_to_save[adapter_name]._hf_hook
            new_hook = self._create_new_hook(old_hook)
            remove_hook_from_module(self.modules_to_save[adapter_name])
            add_hook_to_module(self.modules_to_save[adapter_name], new_hook)

        self.original_module.requires_grad_(False)

        # note that there currently cannot be more than one active adapter for the same layer with modules to save
        # since there would be no clear way to decide which adapter's weights are the correct ones. therefore we
        # assume that there is only one active adapter. this precondition is enforced by _set_adapter.
        if adapter_name == self.active_adapter:
            self.modules_to_save[adapter_name].requires_grad_(True)

    def enable_adapters(self, enabled: bool):
        """Takes care of setting the required_grad flag on the wrapped module.
        If adapters are enabled, gradients for the module are required as well.
        """
        super().enable_adapters(enabled)

        if enabled:
            self.original_module.requires_grad_(False)
            self.modules_to_save[self.active_adapter].requires_grad_(True)
        else:
            self.original_module.requires_grad_(True)
            self.modules_to_save.requires_grad_(False)

    def set_adapter(self, adapter_names: Union[str, list[str]]):
        """Set the active adapter

        Additionally, this function will set the specified adapter to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_names (list[str], str): The name of the adapter to set as active
        """
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        if len(adapter_names) > 1:
            raise ValueError(f"Attempted to set multiple ({adapter_names}) adapters at once for modules_to_save.")

        adapter_name = adapter_names[0]

        if adapter_name not in self._adapters:
            raise ValueError(f"Adapter {adapter_name} not found in {self._adapters}")

        self.modules_to_save[self.active_adapters[0]].requires_grad_(False)
        self.modules_to_save[adapter_name].requires_grad_(True)
        self._active_adapter = adapter_name

    def delete_adapter(self, adapter_name: str, new_active_adapters: Optional[list[str]]) -> None:
        """
        Delete the adapter if present.

        This method will also set a new active adapter if the deleted adapter was the active adapter. It is important
        that the new adapter is chosen by the caller in a deterministic way, so that the same adapter is chosen on all
        layers.
        """
        if adapter_name not in self.modules_to_save:
            return

        # set new active adapter, if necessary
        # note: there can only ever be one active adapter, unlike for LoRA etc.
        if isinstance(new_active_adapters, (list, tuple)) and len(new_active_adapters) > 1:
            name = self.__class__.__name__
            raise ValueError(
                f"Attempted to set multiple ({new_active_adapters}) adapters at once for {name}, which is not allowed."
            )

        if adapter_name in self._adapters:
            self._adapters.remove(adapter_name)

        if not new_active_adapters:
            # no active adapter now
            del self.modules_to_save[adapter_name]
            self._active_adapter = []
            return

        new_active_adapter = new_active_adapters[0]
        if new_active_adapter not in self.modules_to_save:
            # a new active adapter was chosen but it seems like it has no modules_to_save
            del self.modules_to_save[adapter_name]
            self._active_adapter = []
            return

        if new_active_adapter != self.active_adapters[0]:
            self.set_adapter(new_active_adapter)
        del self.modules_to_save[adapter_name]

    def adapter_state_dict_load_map(self, adapter_name):
        # Maps the module keys as they are in the saved state dict to the in-memory state dict.
        # Must contain all keys that are supposed to be loaded.
        if adapter_name not in self._adapters:
            # In caes of multiple adapters, each bringing their own modules to save, each
            # ModulesToSaveWrapper will be queried but not every wrapper is obliged to serve the same adapters.
            return {}
        return {k: f"modules_to_save.{adapter_name}.{k}" for k in self.modules_to_save[adapter_name].state_dict()}

    def adapter_state_dict(self, adapter_name, state_dict):
        if adapter_name not in self._adapters:
            # In caes of multiple adapters, each bringing their own modules to save, each
            # ModulesToSaveWrapper will be queried but not every wrapper is obliged to serve the same adapters.
            return {}

        return {
            k: state_dict[f"modules_to_save.{adapter_name}.{k}"]
            for k in self.modules_to_save[adapter_name].state_dict()
        }

    def unload_and_optionally_merge_module(
        self, merge: bool, safe_merge: bool, adapter_names: Optional[list[str]]
    ) -> torch.nn.Module:
        """Unloading in case of `ModulesToSave` means to simply return the wrapped module.

        However, if the wrapped module is itself a tuner, we'll call merge on it before.
        """
        new_module = self.modules_to_save[self.active_adapter]

        # TODO: not sure if this is still a sensible thing to do. We would basically have to
        # do the same checks as `_unload_and_optionally_merge` to support MHA, for example.
        if hasattr(new_module, "base_layer"):
            # check if the module is itself a tuner layer
            if merge:
                new_module.merge(safe_merge=safe_merge, adapter_names=adapter_names)
            new_module = new_module.get_base_layer()

        return new_module


class TrainableTokensWrapper(AuxiliaryTrainingWrapper):
    """Wraps a module (typically an embedding layer) that is supposed to be re-trained selectively (i.e.
    solely updating a few columns) using the `TrainableTokensLayer` PEFT method.

    Supports weight-tying to another adapter when passed a `tied_adapter` which is expected to be a
    `TrainableTokensLayer`.
    """

    def __init__(
        self,
        module_to_save: torch.nn.Module,
        adapter_name: str,
        token_indices: list[int],
        tied_adapter=None,
    ) -> None:
        super().__init__(module_to_save, adapter_name, token_indices=token_indices, tied_adapter=tied_adapter)

        # unset the original_module attribute since we're using a property to remove this from the state dict.
        self.original_module = None

    @property
    def original_module(self):
        # use a property instead of an attribute to exclude this pointer from the state dict
        # to make sure that it will not be saved.
        return self.token_adapter.base_layer

    def init_modules(self, adapter_name, token_indices, tied_adapter):
        # use a local import to avoid potential circular imports
        from peft.tuners.trainable_tokens import TrainableTokensLayer

        # since super().__init__() calls update before we have a chance to initialise the adapter we would
        # need here, we do the initialization here.
        self.token_adapter = TrainableTokensLayer(self.original_module, adapter_name, token_indices, tied_adapter)

    def _error_message_name(self):
        return "trainable_token_indices"

    def _hasattr_wrapped(self, name, modules):
        return name == "weight"

    def _getattr_wrapped(self, name, modules):
        # some models query self.wte.weight.dtype, some may query the weights directly. for the first case it is not
        # necessary to do anything special but we don't know if is going to be `.dtype`. so we need to get the merged
        # weights from the adapter.
        if name == "weight":
            return modules["token_adapter"].get_merged_weights(self.token_adapter.active_adapters)

        raise RuntimeError(
            f"This code should've never been reached, probably a bad check in `_hasattr_wrapped` for {name}. "
            "Please file an issue under https://github.com/huggingface/peft/issues."
        )

    def _forward_wrapped(self, x, *args, **kwargs):
        if not self.active_adapters:
            return self._forward_wrapped_passthrough(x, *args, **kwargs)
        return self.token_adapter(x)

    def _forward_wrapped_mixed_batch(self, x, active_adapter, *args, **kwargs):
        return self.token_adapter.forward_adapters(x, [active_adapter])

    def _forward_wrapped_passthrough(self, x, *args, **kwargs):
        # the token adapter knows how to deal with disabled adapter / no active adapter, don't call original_module
        # directly
        return self.token_adapter(x, *args, **kwargs)

    def update(self, active_adapter, **kwargs):
        # TODO this does not support deepspeed/fsdp since it is missing a context manager
        # see ModulesToSaveWrapper implementation
        if active_adapter not in self._adapters:
            self.token_adapter.update_layer(active_adapter, **kwargs)

        super().update(active_adapter)

    def adapter_state_dict_load_map(self, adapter_name):
        if self.token_adapter.tied_adapter:
            return {}
        return {"token_adapter.trainable_tokens_delta": f"token_adapter.trainable_tokens_delta.{adapter_name}"}

    def adapter_state_dict(self, adapter_name, state_dict):
        if self.token_adapter.tied_adapter:
            # storing of weight-tied layers is not up to us and will be handled by
            # transformers. we're just here to keep those layers in sync during training.
            # therefore we return an empty state dict.
            return {}

        return {
            f"token_adapter.{k}": state_dict[f"token_adapter.{k}.{adapter_name}"] for k in ["trainable_tokens_delta"]
        }

    def enable_adapters(self, enabled: bool):
        """Enables/disables the underlying `TrainableTokens` adapter.
        Also handles the internal adapter disable flag.
        """
        super().enable_adapters(enabled)

        self.token_adapter.enable_adapters(enabled)

    def set_adapter(self, adapter_names: Union[str, list[str]]):
        super().set_adapter(adapter_names)
        self.token_adapter.set_adapter(adapter_names)

    def delete_adapter(self, adapter_name: str, new_active_adapters: Optional[list[str]]) -> None:
        """
        Delete the adapter if present.

        This method will also set a new active adapter if the deleted adapter was the active adapter. It is important
        that the new adapter is chosen by the caller in a deterministic way, so that the same adapter is chosen on all
        layers.
        """
        self.token_adapter.delete_adapter(adapter_name)

        # set new active adapter, if necessary
        # note: there can only ever be one active adapter, unlike for LoRA etc.
        if isinstance(new_active_adapters, (list, tuple)) and len(new_active_adapters) > 1:
            name = self.__class__.__name__
            raise ValueError(
                f"Attempted to set multiple ({new_active_adapters}) adapters at once for {name}, which is not allowed."
            )

        if adapter_name in self._adapters:
            self._adapters.remove(adapter_name)

        if not new_active_adapters:
            self._active_adapter = []
            return

        if new_active_adapters[0] not in self.token_adapter.trainable_tokens_delta:
            # a new active adapter was chosen but it seems like it has no trainable_tokens
            self._active_adapter = []
            return

        new_active_adapter = new_active_adapters[0]
        self.set_adapter(new_active_adapter)

    def unload_and_optionally_merge_module(
        self, merge: bool, safe_merge: bool, adapter_names: Optional[list[str]]
    ) -> torch.nn.Module:
        """Unloading for `TrainableTokensWrapper` means to return the wrapped module, e.g. the embedding layer and,
        if requested, merging the `TrainableTokens` adapter onto the wrapped module.
        """
        if merge:
            self.token_adapter.merge(safe_merge=safe_merge, adapter_names=adapter_names)
        return self.token_adapter.get_base_layer()


def _get_input_embeddings_name(model, default=None):
    if not hasattr(model, "get_input_embeddings"):
        return default

    input_embeddings = model.get_input_embeddings()
    for name, module in model.named_modules():
        if module is input_embeddings:
            return name

    return default


def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def _freeze_adapter(model, adapter_name):
    for n, p in model.named_parameters():
        if adapter_name in n:
            p.requires_grad = False


def _set_trainable(
    model,
    adapter_name,
    module_names,
    strict_module_check=False,
    wrapper_cls: Optional[AuxiliaryTrainingWrapper] = None,
    **wrapper_kwargs,
):
    """Wraps modules that are supposed to be re-trained either normally, i.e. marking them to require gradients and
    saving them alongside other modules, or with certain methods that go alongside PEFT methods, such as retraining
    specific token indices using selective read/write.

    Note that you need to validate beforehand if there are layers targeted by multiple wrappers, e.g. if the
    'embedding' layer is configured for both `ModulesToSaveWrapper` and `TrainableTokensWrapper` there would be
    conflicts down the line.

    The default is to wrap the module in a `ModulesToSaveWrapper` wrapper.

    If `strict_module_check` is set, this method raises an ValueError, similar to BaseTuner.inject_adapter when none of
    the requested modules in `module_names` is not found in the model.
    """
    if wrapper_cls is None:
        wrapper_cls = ModulesToSaveWrapper

    if not module_names:
        # This is useful for the case that the PEFT config does not have `modules_to_save`, e.g.
        # in the case of prompt tuning and friends.
        return

    trainable_modules = []
    found_modules = set()
    # disable removal of duplicates to support targeting tied weights
    key_list = [key for key, _ in model.named_modules(remove_duplicate=False)]

    for key in key_list:
        target_module_found = any(key.endswith(target_key) for target_key in module_names)
        if target_module_found:
            parent, target, target_name = _get_submodules(model, key)
            if isinstance(target, wrapper_cls):
                target.update(adapter_name, **wrapper_kwargs)
                target.set_adapter(target.active_adapter)
            else:
                new_module = wrapper_cls(target, adapter_name, **wrapper_kwargs)
                new_module.set_adapter(adapter_name)
                setattr(parent, target_name, new_module)
                trainable_modules.append(new_module)
            found_modules.add(target_name)

    not_found = set(module_names).difference(found_modules)
    if strict_module_check and not found_modules:
        raise ValueError(
            f"Target modules {not_found} not found in the base model. Please check the target modules and try again."
        )

    return trainable_modules


def _set_adapter(model, adapter_name):
    def check_adapter_name(adapter_name):
        if isinstance(adapter_name, str):
            return adapter_name

        # adapter_name is a list of str
        if len(adapter_name) > 1:
            raise ValueError("Only one adapter can be set at a time for modules_to_save")
        elif len(adapter_name) == 0:
            raise ValueError("Please specify at least one adapter to set")
        adapter_name = adapter_name[0]
        return adapter_name

    for module in model.modules():
        if isinstance(module, AuxiliaryTrainingWrapper):
            # only check the adapter_name if we actually encounter a AuxiliaryTrainingWrapper, otherwise we don't care
            adapter_name = check_adapter_name(adapter_name)

            # if the adapter is found in this module, set it as the active adapter, else disable the adapters of this
            # module
            if adapter_name in module._adapters:
                module.enable_adapters(True)
                module.set_adapter(adapter_name)
            else:
                module.enable_adapters(False)


def _prepare_prompt_learning_config(peft_config, model_config):
    # In case of VLM we focus on the language model portion of the model.
    if "text_config" in model_config:
        model_config = model_config["text_config"]

    if peft_config.num_layers is None:
        if "num_hidden_layers" in model_config:
            num_layers = model_config["num_hidden_layers"]
        elif "num_layers" in model_config:
            num_layers = model_config["num_layers"]
        elif "n_layer" in model_config:
            num_layers = model_config["n_layer"]
        else:
            raise ValueError("Please specify `num_layers` in `peft_config`")
        peft_config.num_layers = num_layers

    if peft_config.token_dim is None:
        if "hidden_size" in model_config:
            token_dim = model_config["hidden_size"]
        elif "n_embd" in model_config:
            token_dim = model_config["n_embd"]
        elif "d_model" in model_config:
            token_dim = model_config["d_model"]
        else:
            raise ValueError("Please specify `token_dim` in `peft_config`")
        peft_config.token_dim = token_dim

    if peft_config.num_attention_heads is None:
        if "num_attention_heads" in model_config:
            num_attention_heads = model_config["num_attention_heads"]
        elif "n_head" in model_config:
            num_attention_heads = model_config["n_head"]
        elif "num_heads" in model_config:
            num_attention_heads = model_config["num_heads"]
        elif "encoder_attention_heads" in model_config:
            num_attention_heads = model_config["encoder_attention_heads"]
        else:
            raise ValueError("Please specify `num_attention_heads` in `peft_config`")
        peft_config.num_attention_heads = num_attention_heads

    # For grouped-query attention, see #1901.
    if peft_config.peft_type == "PREFIX_TUNING" and "num_key_value_heads" in model_config:
        num_key_value_heads = model_config["num_key_value_heads"]
        peft_config.token_dim = peft_config.token_dim // peft_config.num_attention_heads * num_key_value_heads
        peft_config.num_attention_heads = num_key_value_heads

    if getattr(peft_config, "encoder_hidden_size", None) is None:
        setattr(peft_config, "encoder_hidden_size", peft_config.token_dim)

    return peft_config


def _get_no_split_modules(model) -> set[str]:
    """
    Get the modules of the model that should not be split when using device_map. We iterate through the modules to get
    the underlying `_no_split_modules`.

    Returns:
        `List[str]`: List of modules that should not be split
    """
    # After discussion in https://github.com/huggingface/transformers/pull/38141, based on:
    # https://github.com/huggingface/transformers/blob/1e921a3a9cea92b383ca4b0484ee45596bbdadc3/src/transformers/modeling_utils.py#L2677-L2704
    _no_split_modules: set[str] = set()
    if not hasattr(model, "_no_split_modules"):
        return _no_split_modules

    modules_to_check = [model]
    while len(modules_to_check) > 0:
        module = modules_to_check.pop(-1)
        # if the module does not appear in _no_split_modules, we also check the children
        if module.__class__.__name__ not in _no_split_modules:
            if isinstance(module, PreTrainedModel):
                if module._no_split_modules is not None:
                    _no_split_modules = _no_split_modules | set(module._no_split_modules)
            modules_to_check += list(module.children())
    return _no_split_modules


def fsdp_auto_wrap_policy(model):
    if hasattr(FullyShardedDataParallelPlugin, "get_module_class_from_name"):
        get_module_class_from_name = FullyShardedDataParallelPlugin.get_module_class_from_name
    else:
        from accelerate.utils.dataclasses import get_module_class_from_name
    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    from ..tuners import PrefixEncoder, PromptEmbedding, PromptEncoder

    default_transformer_cls_names_to_wrap = ",".join(_get_no_split_modules(model))
    transformer_cls_names_to_wrap = os.environ.get(
        "FSDP_TRANSFORMER_CLS_TO_WRAP", default_transformer_cls_names_to_wrap
    ).split(",")
    transformer_cls_to_wrap = {PrefixEncoder, PromptEncoder, PromptEmbedding}
    for layer_class in transformer_cls_names_to_wrap:
        if len(layer_class) == 0:
            continue
        transformer_cls = get_module_class_from_name(model, layer_class)
        if transformer_cls is None:
            raise Exception("Could not find the transformer layer class to wrap in the model.")
        else:
            transformer_cls_to_wrap.add(transformer_cls)

    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_cls_to_wrap,
    )

    auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    return auto_wrap_policy


def transpose(weight, fan_in_fan_out):
    if not fan_in_fan_out:
        return weight

    if isinstance(weight, torch.nn.Parameter):
        return torch.nn.Parameter(weight.T)
    return weight.T


def _is_valid_match(key: str, target_key: str):
    """
    Helper function to match module names target_key and key. Makes sure that either the key is exactly the target_key
    or the target_key is a submodule of key
    """
    if key.endswith(target_key):
        if len(key) > len(target_key):
            return key.endswith("." + target_key)  # must be a sub module
        return True
    return False


def _get_batch_size(input_ids: Optional[torch.Tensor], inputs_embeds: Optional[torch.Tensor]) -> int:
    """Get the batch size based on either input_ids or input_embeds

    Raises an ValueError if both are None.

    """
    if (input_ids is None) and (inputs_embeds is None):
        raise ValueError("You have to provide either input_ids or inputs_embeds")

    if input_ids is not None:
        batch_size = input_ids.shape[0]
    else:
        batch_size = inputs_embeds.shape[0]
    return batch_size


def get_quantization_config(model: torch.nn.Module, method: str):
    """
    Get the quantization config of the related quantization method
    """
    if (
        hasattr(model, "config")
        and hasattr(model.config, "quantization_config")
        and (getattr(model, "quantization_method", None) == method)
    ):
        return model.config.quantization_config
    return None


def get_auto_gptq_quant_linear(gptq_quantization_config):
    """
    Get the right AutoGPTQQuantLinear class based on the quantization config file
    """
    if gptq_quantization_config is None:
        return None

    if is_auto_gptq_available():
        from auto_gptq.utils.import_utils import dynamically_import_QuantLinear
    else:
        return None

    desc_act = gptq_quantization_config.desc_act
    group_size = gptq_quantization_config.group_size
    bits = gptq_quantization_config.bits
    if hasattr(gptq_quantization_config, "use_exllama"):
        use_exllama = gptq_quantization_config.use_exllama
    else:
        use_exllama = not gptq_quantization_config.disable_exllama
    if hasattr(gptq_quantization_config, "exllama_config"):
        exllama_version = gptq_quantization_config.exllama_config["version"]
    else:
        exllama_version = 1

    QuantLinear = dynamically_import_QuantLinear(
        use_triton=False,
        desc_act=desc_act,
        group_size=group_size,
        bits=bits,
        disable_exllama=not (use_exllama and exllama_version == 1),
        disable_exllamav2=not (use_exllama and exllama_version == 2),
    )

    return QuantLinear


def get_gptqmodel_quant_linear(gptq_quantization_config, device_map=None):
    """
    Get the right GPTQQuantLinear class based on the quantization config file
    """
    if gptq_quantization_config is None:
        return None

    if not is_gptqmodel_available():
        return None

    from gptqmodel.utils.importer import hf_select_quant_linear

    desc_act = gptq_quantization_config.desc_act
    group_size = gptq_quantization_config.group_size
    bits = gptq_quantization_config.bits
    checkpoint_format = (
        gptq_quantization_config.checkpoint_format
        if hasattr(gptq_quantization_config, "checkpoint_format")
        else "gptq"
    )
    sym = gptq_quantization_config.sym
    meta = gptq_quantization_config.meta if hasattr(gptq_quantization_config, "meta") else None

    QuantLinear = hf_select_quant_linear(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        device_map=device_map,
        checkpoint_format=checkpoint_format,
        meta=meta,
        backend="auto_trainable",
    )

    return QuantLinear


def id_tensor_storage(tensor: torch.Tensor) -> tuple[torch.device, int, int]:
    """
    Unique identifier to a tensor storage. Multiple different tensors can share the same underlying storage. For
    example, "meta" tensors all share the same storage, and thus their identifier will all be equal. This identifier is
    guaranteed to be unique and constant for this tensor's storage during its lifetime. Two tensor storages with
    non-overlapping lifetimes may have the same id.

    This method is the exact same copy of
    https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py#L282C1-L300C58 but we added
    it here manually to avoid import issue with old versions of transformers.
    """
    if tensor.device.type == "xla" and is_torch_tpu_available():
        # NOTE: xla tensors dont have storage
        # use some other unique id to distinguish.
        # this is a XLA tensor, it must be created using torch_xla's
        # device. So the following import is safe:
        import torch_xla

        unique_id = torch_xla._XLAC._xla_get_tensor_id(tensor)
    else:
        unique_id = storage_ptr(tensor)

    return tensor.device, unique_id, storage_size(tensor)


def cast_mixed_precision_params(model, dtype):
    """
    Cast all non-trainable parameters of the model to the given `dtype`. The `dtype` can be `torch.float16` or
    `torch.bfloat16` as per the mixed-precision training you are performing. The trainable parameters are cast to full
    precision. This is meant to reduce the GPU memory usage when using PEFT methods by using half-precision dtype for
    non-trainable parameters. Having the trainable parameters in full-precision preserves training stability when using
    automatic mixed-precision training.

    Args:
        model (`torch.nn.Module`):
            The model to cast the non-trainable parameters of.
        dtype (`torch.dtype`):
            The dtype to cast the non-trainable parameters to. The `dtype` can be `torch.float16` or
    `torch.bfloat16` as per the mixed-precision training you are performing.
    """
    for p in model.parameters():
        if not p.requires_grad:
            p.data = p.to(dtype)
        else:
            p.data = p.to(torch.float32)


def str_to_bool(value: str) -> int:
    """
    Converts a string representation of truth to `True` (1) or `False` (0).

    True values are `y`, `yes`, `t`, `true`, `on`, and `1`; False value are `n`, `no`, `f`, `false`, `off`, and `0`;
    """
    # same as function as in accelerate.utils, which replaces the deprecated distutils.util.strtobool
    value = value.lower()
    if value in ("y", "yes", "t", "true", "on", "1"):
        return 1
    elif value in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        raise ValueError(f"invalid truth value {value}")


def check_file_exists_on_hf_hub(repo_id: str, filename: str, **kwargs) -> Optional[bool]:
    """Check if a file exists on HF Hub, if check was not successful returns None instead of erroring.

    Respect offline mode if set.

    """
    exists: Optional[bool] = None
    if str_to_bool(os.environ.get("HF_HUB_OFFLINE", "0")):
        # user set offline mode, cannot check
        return exists

    try:
        exists = file_exists(repo_id, filename, **kwargs)
    except (HFValidationError, EntryNotFoundError):
        # error, exists stays None
        pass
    except Exception as e:
        warnings.warn(
            f"Unable to fetch remote file due to the following error {e} - silently ignoring the lookup"
            f" for the file {filename} in {repo_id}."
        )

    return exists


def get_pattern_key(pattern_keys: Sequence[str], key_to_match: str) -> str:
    """Match a substring of key_to_match in pattern keys"""
    for key in pattern_keys:
        match = re.match(rf"(.*\.)?({key})$", key_to_match)
        if not match:
            continue
        return key

    return key_to_match


def set_additional_trainable_modules(model, peft_config, model_config, adapter_name):
    """Handle the resolution of additional trainable modules (also called AuxiliaryTrainingWrapper)
    by checking the config if such modules are requested and adding them to the model.

    Currently trainable tokens and modules to save are considered additional trainable modules.
    """
    if getattr(peft_config, "modules_to_save", None) is not None:
        # this may add a new ModulesToSaveWrapper
        _set_trainable(model, adapter_name, module_names=getattr(peft_config, "modules_to_save", None))

    if getattr(peft_config, "trainable_token_indices", None) is not None:
        if isinstance(peft_config.trainable_token_indices, dict):
            target_layers = peft_config.trainable_token_indices
        else:
            layer_name = _get_input_embeddings_name(model, "embed_tokens")
            target_layers = {layer_name: peft_config.trainable_token_indices}

        modules_to_save = getattr(peft_config, "modules_to_save", None)
        if modules_to_save is not None:
            for target_layer in target_layers:
                if target_layer in modules_to_save:
                    raise ValueError(
                        "The embedding layer is already marked to be trained fully, either specify "
                        f'`modules_to_save=[..., "{target_layer}", ...]` or '
                        f"`trainable_tokens={{'{target_layer}': x}}` but not both."
                    )

        for target_layer, token_indices in target_layers.items():
            _set_trainable(
                model,
                adapter_name,
                module_names=[target_layer],
                strict_module_check=True,
                wrapper_cls=TrainableTokensWrapper,
                token_indices=token_indices,
            )

        # There might be the possibility that we have output weights that are tied to the input weights.
        # In that case we will tie any module that wants tied weights to the token adapter to make sure that
        # any modification is reflected in the tied layers as well.
        if (
            model_config.get("tie_word_embeddings", False)
            # some models may be misconfigured to have weight tying enabled but don't define tied weights keys
            and model._tied_weights_keys is not None
            and isinstance(model.get_input_embeddings(), TrainableTokensWrapper)
        ):
            # the embedding layer is modified and we want weight tying.
            module_keys = [".".join(n.split(".")[:-1]) for n in model._tied_weights_keys]

            token_adapter = model.get_input_embeddings().token_adapter
            _set_trainable(
                model,
                adapter_name,
                module_names=module_keys,
                strict_module_check=True,
                wrapper_cls=TrainableTokensWrapper,
                token_indices=token_adapter.token_indices[adapter_name],
                tied_adapter=model.get_input_embeddings().token_adapter,
            )


def create_attention_mask(
    model, *, model_input, attention_mask, past_key_values, cache_position, batch_size, sequence_length
):
    # adapted from:
    # https://github.com/huggingface/transformers/blob/cb4c56ce0dfa1350267ed28e57760986a58a9ba4/src/transformers/generation/utils.py#L644-L680
    # In PEFT, we sometimes need to re-create the attention mask. This is because some prompt learning methods insert
    # new items into the sequence, which results in the attention mask needing an update. We re-use transformers code
    # for this as much as possible.
    try:
        from transformers.masking_utils import create_masks_for_generate
    except ImportError as exc:
        raise ImportError("Your transformers version is too old, please upgrade it to > 4.52") from exc

    # Create the causal mask with fixed shape in advance, to reduce recompilations. If the function to create
    # the 4D causal mask exists, it should be present in the base model (XXXModel class) or in its decoder.
    base_model = getattr(model, model.base_model_prefix, model)
    decoder = base_model.get_decoder() if hasattr(base_model, "get_decoder") else None
    causal_mask_creation_function = getattr(base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None)
    if causal_mask_creation_function is None and decoder is not None:  # it may be in the decoder
        causal_mask_creation_function = getattr(decoder, "_prepare_4d_causal_attention_mask_with_cache_position", None)

    # If it's not defined, it means the model uses the new general mask API
    if causal_mask_creation_function is None:  # can't be found
        token_type_ids = getattr(model_input, "token_type_ids", None)
        # Some models may overwrite the general one
        causal_mask_creation_function = getattr(model, "create_masks_for_generate", create_masks_for_generate)
        attention_mask = causal_mask_creation_function(
            config=model.config,
            # we only need batch size, seq_length and dtype here - we don't care about the values of the embeddings
            input_embeds=torch.empty((batch_size, sequence_length), dtype=model.dtype),
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            token_type_ids=token_type_ids,
        )
    else:
        attention_mask = causal_mask_creation_function(
            attention_mask,
            sequence_length=sequence_length,
            target_length=past_key_values.get_max_cache_shape(),
            dtype=model.dtype,
            cache_position=cache_position,
            batch_size=batch_size,
            config=model.config,
            past_key_values=past_key_values,
        )
    return attention_mask
