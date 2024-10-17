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

from operator import attrgetter

import torch

from peft.config import PeftConfig
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING

from .constants import PEFT_TYPE_TO_PREFIX_MAPPING
from .other import infer_device
from .peft_types import PeftType
from .save_and_load import _insert_adapter_name_into_state_dict, load_peft_weights


# so far only LoRA is supported
CONFIG_KEYS_TO_CHECK = {PeftType.LORA: ["lora_alpha", "use_rslora", "lora_dropout", "alpha_pattern", "use_dora"]}


def hotswap_adapter_from_state_dict(model, state_dict, adapter_name, parameter_prefix="lora_"):
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
        # no need to account for potential _orig_mod in key here, as torch handles that
        old_val = attrgetter(key)(model)
        if is_compiled:
            # Compiled models don't work with swap_tensors because there are weakrefs for the tensor. It is unclear if
            # this workaround could not cause trouble but the tests indicate that it works.
            old_val.data = new_val.data
        else:
            torch.utils.swap_tensors(old_val, new_val)


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
    )
