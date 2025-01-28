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
from __future__ import annotations

import math
from operator import attrgetter
from typing import Optional

import torch

from peft.config import PeftConfig
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING, PEFT_TYPE_TO_PREFIX_MAPPING
from peft.tuners.lora import Conv2d, Linear, LoraConfig, LoraLayer

from .other import get_pattern_key, infer_device
from .peft_types import PeftType
from .save_and_load import _insert_adapter_name_into_state_dict, load_peft_weights


# so far only LoRA is supported
CONFIG_KEYS_TO_CHECK = {PeftType.LORA: ["use_rslora", "lora_dropout", "alpha_pattern", "use_dora"]}


def _update_scaling(lora_module, adapter_name, scaling=None):
    """
    Update the value of the scalings of the LoRA module.

    Takes into consideration that scalings can be tensors from prepare_model_for_compiled_hotswap.
    """
    if lora_module.scaling[adapter_name] == scaling:
        return

    if isinstance(lora_module.scaling[adapter_name], torch.Tensor):
        lora_module.scaling[adapter_name].fill_(scaling)
    elif isinstance(lora_module.scaling[adapter_name], (float, int)):
        lora_module.scaling[adapter_name] = scaling
    else:
        raise ValueError(
            "Something went wrong when trying to set the new scale value, expected to find the old value to be of type "
            f"float or torch.Tensor, got {type(lora_module.scaling[adapter_name])} instead."
        )


def _convert_scalings_to_tensor(model):
    """
    Convert the LoRA scaling values into torch.tensors to prevent recompilation if they change.
    """
    for module in model.modules():
        if not isinstance(module, LoraLayer):
            continue

        scaling = module.scaling
        for key, val in scaling.items():
            if isinstance(val, float):
                scaling[key] = torch.tensor(val, device=module.weight.device)
            elif not isinstance(val, torch.Tensor):
                raise ValueError(
                    "Something went wrong while trying to convert the scalings, expected to find values of type float "
                    f"but found {type(val)} instead."
                )


def _pad_lora_weights(model, target_rank):
    """
    Pad LoRA weights in a state dict to a target rank while preserving the original behavior.

    Args:
      state_dict (dict): The state dict containing LoRA weights
      target_rank (int): The target rank to pad to

    Returns: new_state_dict: A new state dict with padded LoRA weights
    """
    for module in model.modules():
        if not isinstance(module, (Conv2d, Linear)):
            continue

        is_conv = isinstance(module, Conv2d)

        # LoRA A
        for adapter_name, lora_module in module.lora_A.items():
            weight = lora_module.weight
            original_rank = weight.size(0)

            if original_rank == target_rank:
                continue

            if original_rank > target_rank:
                raise ValueError(
                    f"Trying to pad the adapter to the target rank {target_rank}, but the original rank is larger "
                    f"({original_rank}), which is not possible. Please choose a target rank that is greater or equal "
                    "to the largest rank of the adapter."
                )

            if is_conv:
                padded = torch.zeros(
                    target_rank,
                    weight.size(1),
                    weight.size(2),
                    weight.size(3),
                    device=weight.device,
                    dtype=weight.dtype,
                )
                padded[:original_rank, :, :, :] = weight
                new_layer = torch.nn.Conv2d(
                    weight.size(1),
                    target_rank,
                    kernel_size=lora_module.kernel_size,
                    stride=lora_module.stride,
                    padding=lora_module.padding,
                    bias=lora_module.bias,
                )
            else:
                padded = torch.zeros(target_rank, weight.size(1), device=weight.device, dtype=weight.dtype)
                padded[:original_rank, :] = weight
                new_layer = torch.nn.Linear(weight.size(1), target_rank, bias=lora_module.bias)

            if new_layer.weight.shape != padded.shape:
                raise ValueError(
                    "Something went wrong when trying to pad the LoRA weights, the new shape should be "
                    f"{padded.shape} but {new_layer.weight.shape} was found. Please open an issue on PEFT "
                    "(https://github.com/huggingface/peft/issues) and report this error."
                )

            new_layer.weight.data = padded
            if lora_module.bias:
                new_layer.bias.data = lora_module.bias.data
            module.lora_A[adapter_name] = new_layer

        # LoRA B
        for adapter_name, lora_module in module.lora_B.items():
            weight = lora_module.weight
            original_rank = weight.size(1)

            if original_rank == target_rank:
                continue

            if original_rank > target_rank:
                # TODO: is this necessary or can we just continue???
                raise ValueError(
                    f"Trying to pad the adapter to the target rank {target_rank}, but the original rank is larger "
                    f"({original_rank}), which is not possible. Please choose a target rank that is greater or equal "
                    "to the largest rank of the adapter."
                )

            if is_conv:
                padded = torch.zeros(
                    weight.size(0),
                    target_rank,
                    weight.size(2),
                    weight.size(3),
                    device=weight.device,
                    dtype=weight.dtype,
                )
                padded[:, :original_rank, :, :] = weight
                new_layer = torch.nn.Conv2d(
                    target_rank,
                    weight.size(0),
                    kernel_size=lora_module.kernel_size,
                    stride=lora_module.stride,
                    padding=lora_module.padding,
                    bias=lora_module.bias,
                )
                new_layer.weight.data = padded
            else:
                padded = torch.zeros(weight.size(0), target_rank, device=weight.device, dtype=weight.dtype)
                padded[:, :original_rank] = weight
                new_layer = torch.nn.Linear(target_rank, weight.size(0), bias=lora_module.bias)

            if new_layer.weight.shape != padded.shape:
                raise ValueError(
                    "Something went wrong when trying to pad the LoRA weights, the new shape should be "
                    f"{padded.shape} but {new_layer.weight.shape} was found. Please open an issue on PEFT "
                    "(https://github.com/huggingface/peft/issues) and report this error."
                )

            new_layer.weight.data = padded
            if lora_module.bias:
                new_layer.bias.data = lora_module.bias.data
            module.lora_B[adapter_name] = new_layer


def prepare_model_for_compiled_hotswap(
    model: torch.nn.Module,
    *,
    target_rank: Optional[int] = None,
    config: Optional[LoraConfig | dict[str, LoraConfig]] = None,
) -> None:
    """
    Helper function that prepares the model so that it can later be compiled and then used with hot-swapping.

    It is necessary to call this function on the model for hot-swapping to work if

    - the different LoRA adapters have different ranks and/or different alpha values (i.e. scalings)
    - you plan to torch.compile the model and want to avoid re-compilation

    It is important to call this function *after* the first LoRA adapter has been loaded (i.e. the one that will be
    swapped out) but *before* the model is compiled.

    Even with this function, hot-swapping LoRA adapters that target different layers is still not supported.

    Note: This function modifies the model in-place. If you want to restore the model to its initial state, you will
    have to reload it.

    Args:
        model (`nn.Module`):
            The model with the loaded adapter, before compilation.
        target_rank (`int`, *optional*):
            The target rank to pad the LoRA weights to. Should be the maximum rank among all LoRA adapters that will be
            hot-swapped. If not specified, the target ranks will not be changed.
        config (`LoraConfig` or `dict[str, LoraConfig]`, *optional*):
            Optionally pass the `LoraConfig`s of the LoRA adapters. If passed, the rank in the configs will be updated
            to `target_rank`.
    """
    is_compiled = hasattr(model, "_orig_mod")
    if is_compiled:
        raise ValueError("Call prepare_model_for_compiled_hotswap *before* compiling the model")

    _convert_scalings_to_tensor(model)
    if target_rank is not None:
        _pad_lora_weights(model, target_rank=target_rank)

    if not config:
        return
    if target_rank is None:
        return

    if not isinstance(config, dict):
        config = {"dummy": config}

    for lora_config in config.values():
        lora_config.r = target_rank
        if lora_config.rank_pattern:
            for key in lora_config.rank_pattern:
                lora_config.rank_pattern[key] = target_rank


def hotswap_adapter_from_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    adapter_name: str,
    config: LoraConfig,
    parameter_prefix: str = "lora_",
):
    """
    Swap out the adapter weights from the model with the weights from state_dict.

    As of now, only LoRA is supported.

    This is a low-level function that assumes that the adapters have been checked for compatibility and that the
    state_dict has been correctly mapped to work with PEFT. For a high level function that performs this work for you,
    use `hotswap_adapter` instead.

    Args:
        model (`nn.Module`):
            The model with the loaded adapter.
        state_dict (`dict[str, torch.Tensor]`):
            The state dict of the new adapter, which needs to be compatible (targeting same modules etc.).
        adapter_name (`str`):
            The name of the adapter that should be hot-swapped, e.g. `"default"`. The name will remain the same after
            swapping.
        config (`LoraConfig`):
            The config of the LoRA adapter. This is used to determine the scaling and rank of the adapter.
        parameter_prefix (`str`, *optional*, defaults to `"lora_"`)
            The prefix used to identify the adapter's keys in the state dict. For LoRA, this would be `"lora_"` (the
            default).

    Raises:
        RuntimeError
            If the old and the new adapter are not compatible, a RuntimeError is raised.

    """
    # Ensure that all the keys of the new adapter correspond exactly to the keys of the old adapter, otherwise
    # hot-swapping is not possible

    is_compiled = hasattr(model, "_orig_mod")
    # TODO: there is probably a more precise way to identify the adapter keys
    missing_keys = {k for k in model.state_dict() if (parameter_prefix in k) and (adapter_name in k)}
    unexpected_keys = set()

    # first: dry run, not swapping anything
    for key, new_val in state_dict.items():
        try:
            old_val = attrgetter(key)(model)
        except AttributeError:
            unexpected_keys.add(key)
            continue

        if is_compiled:
            missing_keys.remove("_orig_mod." + key)
        else:
            missing_keys.remove(key)

    if missing_keys or unexpected_keys:
        msg = "Hot swapping the adapter did not succeed."
        if missing_keys:
            msg += f" Missing keys: {', '.join(sorted(missing_keys))}."
        if unexpected_keys:
            msg += f" Unexpected keys: {', '.join(sorted(unexpected_keys))}."
        raise RuntimeError(msg)

    # actual swapping
    for key, new_val in state_dict.items():
        module_name = ".".join(key.split(".")[:-3])
        module = model.get_submodule(module_name)

        # swap alpha/scaling
        r_key = get_pattern_key(config.rank_pattern.keys(), key)
        alpha_key = get_pattern_key(config.alpha_pattern.keys(), key)
        rank = config.rank_pattern.get(r_key, config.r)
        alpha = config.alpha_pattern.get(alpha_key, config.lora_alpha)
        if config.use_rslora:
            scaling = alpha / math.sqrt(rank)
        else:
            scaling = alpha / rank
        _update_scaling(module, adapter_name=adapter_name, scaling=scaling)

        # swap actual weights
        # no need to account for potential _orig_mod in key here, as torch handles that
        old_val = attrgetter(key)(model)
        if not is_compiled:
            torch.utils.swap_tensors(old_val, new_val)
            continue

        # Compiled models don't work with swap_tensors because there are weakrefs for the tensor. It is unclear if
        # this workaround could not cause trouble but the tests indicate that it works.
        if old_val.shape == new_val.shape:
            old_val.data = new_val.data
        else:
            if old_val.dim() != 2:
                # TODO conv2d
                raise NotImplementedError
            if old_val.shape[0] > new_val.shape[0]:
                old_val.data.fill_(0)
                old_val.data[: new_val.shape[0]] = new_val.data
            elif old_val.shape[1] > new_val.shape[1]:
                old_val.data.fill_(0)
                old_val.data[:, : new_val.shape[1]] = new_val.data
            else:
                raise ValueError(
                    f"Incompatible shapes found for LoRA weights {key}: {old_val.shape} vs {new_val.shape}. Please "
                    "ensure that all ranks are padded to the largest rank among all LoRA adapters by using "
                    "peft.utils.hotswap.prepare_model_for_compiled_hotswap."
                )


def _check_hotswap_configs_compatible(config0: PeftConfig, config1: PeftConfig) -> None:
    """
    Check if two configs are compatible for hot-swapping.

    Only LoRA parameters are checked for now.

    To hot-swap two adapters, their configs must be compatible. Otherwise, the results could be false. E.g. if they use
    different alpha values, after hot-swapping, the alphas from the first adapter would still be used with the weights
    from the 2nd adapter, which would result in incorrect behavior. There is probably a way to swap these values as
    well, but that's not implemented yet, and we need to be careful not to trigger re-compilation if the model is
    compiled (so no modification of the dict).

    """

    if config0.peft_type != config1.peft_type:
        msg = f"Incompatible PEFT types found: {config0.peft_type.value} and {config1.peft_type.value}"
        raise ValueError(msg)

    if config0.peft_type not in CONFIG_KEYS_TO_CHECK:
        msg = (
            f"Hotswapping only supports {', '.join(CONFIG_KEYS_TO_CHECK.keys())} but "
            f"{config0.peft_type.value} was passed."
        )
        raise ValueError(msg)
    config_keys_to_check = CONFIG_KEYS_TO_CHECK[config0.peft_type]

    # TODO: This is a very rough check only for LoRA at the moment. Also, there might be some options that don't
    # necessarily require an error.
    config0 = config0.to_dict()
    config1 = config1.to_dict()
    sentinel = object()
    for key in config_keys_to_check:
        val0 = config0.get(key, sentinel)
        val1 = config1.get(key, sentinel)
        if val0 != val1:
            raise ValueError(f"Configs are incompatible: for {key}, {val0} != {val1}")


def hotswap_adapter(model, model_name_or_path, adapter_name, torch_device=None, **kwargs):
    """Substitute old adapter data with new adapter data, keeping the rest the same.

    As of now, only LoRA is supported.

    This function is useful when you want to replace the loaded adapter with a new adapter. The adapter name will
    remain the same, but the weights and other parameters will be swapped out.

    If the adapters are incomptabile, e.g. targeting different layers or having different alpha values, an error will
    be raised.

    Example:

    ```py
    >>> import torch
    >>> from transformers import AutoModelForCausalLM
    >>> from peft import PeftModel
    >>> from peft.utils.hotswap import hotswap_adapter

    >>> model_id = ...
    >>> inputs = ...
    >>> device = ...
    >>> model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    >>> # load lora 0
    >>> model = PeftModel.from_pretrained(model, "path-adapter-0")
    >>> model = torch.compile(model)  # optionally compile the model
    >>> with torch.inference_mode():
    ...     output_adapter_0 = model(inputs)

    >>> # replace the "default" lora adapter with the new one
    >>> hotswap_adapter(model, "path-adapter-1", adapter_name="default", torch_device=device)
    >>> with torch.inference_mode():
    ...     output_adapter_1 = model(inputs).logits
    ```

    Args:
        model ([`~PeftModel`]):
            The PEFT model with the loaded adapter.
        model_name_or_path (`str`):
            The name or path of the model to load the new adapter from.
        adapter_name (`str`):
            The name of the adapter to swap, e.g. `"default"`. The name will stay the same after swapping.
        torch_device: (`str`, *optional*, defaults to None):
            The device to load the new adapter onto.
        **kwargs (`optional`):
            Additional keyword arguments used for loading the config and weights.

    """
    if torch_device is None:
        torch_device = infer_device()

    ############################
    # LOAD CONFIG AND VALIDATE #
    ############################

    config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[
        PeftConfig._get_peft_type(
            model_name_or_path,
            subfolder=kwargs.get("subfolder", None),
            revision=kwargs.get("revision", None),
            cache_dir=kwargs.get("cache_dir", None),
            use_auth_token=kwargs.get("use_auth_token", None),
            token=kwargs.get("token", None),
        )
    ]
    config = config_cls.from_pretrained(model_name_or_path, **kwargs)
    # config keys that could affect the model output besides what is determined by the state_dict
    _check_hotswap_configs_compatible(model.active_peft_config, config)

    state_dict = load_peft_weights(model_name_or_path, device=torch_device, **kwargs)

    ###########################
    # LOAD & REMAP STATE_DICT #
    ###########################

    parameter_prefix = PEFT_TYPE_TO_PREFIX_MAPPING[config.peft_type]
    peft_model_state_dict = _insert_adapter_name_into_state_dict(
        state_dict, adapter_name=adapter_name, parameter_prefix=parameter_prefix
    )

    hotswap_adapter_from_state_dict(
        model=model,
        state_dict=peft_model_state_dict,
        adapter_name=adapter_name,
        parameter_prefix=parameter_prefix,
        config=config,
    )
