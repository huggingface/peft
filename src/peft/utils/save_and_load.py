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

import os
import warnings
from typing import Optional

import torch
from huggingface_hub import file_exists, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors.torch import load_file as safe_load_file

from .other import (
    EMBEDDING_LAYER_NAMES,
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    check_file_exists_on_hf_hub,
    infer_device,
)
from .peft_types import PeftType


def has_valid_embedding_base_layer(layer):
    """Check if the layer has an embedding base layer"""
    return hasattr(layer, "base_layer") and isinstance(layer.base_layer, (torch.nn.Linear, torch.nn.Embedding))


def get_embedding_layer_name(model, layer, is_embedding_in_target_modules):
    """Get the name of the embedding module for a given layer."""
    for name, module in model.named_modules():
        if (not is_embedding_in_target_modules and module == layer) or module == getattr(layer, "base_layer", None):
            return name
    return None


def get_peft_model_state_dict(
    model, state_dict=None, adapter_name="default", unwrap_compiled=False, save_embedding_layers="auto"
):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
            the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the passed model will be used.
        adapter_name (`str`, *optional*, defaults to `"default"`):
            The name of the adapter whose state dict should be returned.
        unwrap_compiled (`bool`, *optional*, defaults to `False`):
            Whether to unwrap the model if torch.compile was used.
        save_embedding_layers (`Union[bool, str]`, , *optional*, defaults to `auto`):
            If `True`, save the embedding layers in addition to adapter weights. If `auto`, checks the common embedding
            layers `peft.utils.other.EMBEDDING_LAYER_NAMES` in config's `target_modules` when available. Based on it
            sets the boolean flag. This only works for 🤗 transformers models.
    """
    if unwrap_compiled:
        model = getattr(model, "_orig_mod", model)

    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()
    if config.peft_type in (PeftType.LORA, PeftType.ADALORA):
        # to_return = lora_state_dict(model, bias=model.peft_config.bias)
        # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
        # to be used directly with the state dict which is necessary when using DeepSpeed or FSDP
        bias = config.bias
        if bias == "none":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k}
        elif bias == "all":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k}
        elif bias == "lora_only":
            to_return = {}
            for k in state_dict:
                if "lora_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("lora_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
        to_return = {k: v for k, v in to_return.items() if (("lora_" in k and adapter_name in k) or ("bias" in k))}
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                rank_pattern = {k.replace(f".{adapter_name}", ""): v for k, v in rank_pattern.items()}
                config.rank_pattern = rank_pattern
                to_return = model.resize_state_dict_by_rank_pattern(rank_pattern, to_return, adapter_name)

    elif config.peft_type == PeftType.BOFT:
        bias = config.bias
        if bias == "none":
            to_return = {k: state_dict[k] for k in state_dict if "boft_" in k}
        elif bias == "all":
            to_return = {k: state_dict[k] for k in state_dict if "boft_" in k or "bias" in k}
        elif bias == "boft_only":
            to_return = {}
            for k in state_dict:
                if "boft_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("boft_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError

    elif config.peft_type == PeftType.LOHA:
        to_return = {k: state_dict[k] for k in state_dict if "hada_" in k}

    elif config.peft_type == PeftType.LOKR:
        to_return = {k: state_dict[k] for k in state_dict if "lokr_" in k}

    elif config.peft_type == PeftType.ADAPTION_PROMPT:
        to_return = {k: state_dict[k] for k in state_dict if k.split(".")[-1].startswith("adaption_")}

    elif config.is_prompt_learning:
        to_return = {}
        if config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
            to_return["prefix_task_cols"] = model.prompt_encoder[adapter_name].prefix_task_cols
            to_return["prefix_task_rows"] = model.prompt_encoder[adapter_name].prefix_task_rows
            prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
        else:
            if config.inference_mode:
                prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
            else:
                prompt_embeddings = model.get_prompt_embedding_to_save(adapter_name)
        to_return["prompt_embeddings"] = prompt_embeddings

    elif config.peft_type == PeftType.IA3:
        to_return = {k: state_dict[k] for k in state_dict if "ia3_" in k}

    elif config.peft_type == PeftType.OFT:
        to_return = {k: state_dict[k] for k in state_dict if "oft_" in k}

    elif config.peft_type == PeftType.POLY:
        to_return = {k: state_dict[k] for k in state_dict if "poly_" in k}

    elif config.peft_type == PeftType.LN_TUNING:
        to_return = {k: state_dict[k] for k in state_dict if "ln_tuning_" in k}

    elif config.peft_type == PeftType.VERA:
        to_return = {k: state_dict[k] for k in state_dict if "vera_lambda_" in k}
        if config.save_projection:
            # TODO: adding vera_A and vera_B to `self.get_base_layer` would
            # make name to match here difficult to predict.
            if f"base_model.vera_A.{adapter_name}" not in state_dict:
                raise ValueError(
                    "Model was initialised to not save vera_A and vera_B but config now specifies to save projection!"
                    " Set `config.save_projection` to `False`."
                )
            to_return["base_model.vera_A." + adapter_name] = state_dict["base_model.vera_A." + adapter_name]
            to_return["base_model.vera_B." + adapter_name] = state_dict["base_model.vera_B." + adapter_name]

    else:
        raise ValueError(f"Unknown PEFT type passed: {config.peft_type}")

    if getattr(model, "modules_to_save", None) is not None:
        for key, value in state_dict.items():
            if any(f"{module_name}.modules_to_save.{adapter_name}" in key for module_name in model.modules_to_save):
                to_return[key.replace("modules_to_save.", "")] = value

    # check the common embedding layers in `target_modules` to reset `save_embedding_layers` if necessary
    is_embedding_in_target_modules = False
    if (
        save_embedding_layers == "auto"
        and hasattr(config, "target_modules")
        and any(k in config.target_modules for k in EMBEDDING_LAYER_NAMES)
    ):
        warnings.warn("Setting `save_embedding_layers` to `True` as embedding layers found in `target_modules`.")
        save_embedding_layers = is_embedding_in_target_modules = True
    elif save_embedding_layers == "auto":
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
        model_id = getattr(config, "base_model_name_or_path", None)

        # For some models e.g. diffusers the text config file is stored in a subfolder
        # we need to make sure we can download that config.
        has_remote_config = False

        # ensure that this check is not performed in HF offline mode, see #1452
        if model_id is not None:
            exists = check_file_exists_on_hf_hub(model_id, "config.json")
            if exists is None:
                # check failed, could not determine if it exists or not
                warnings.warn(
                    f"Could not find a config file in {model_id} - will assume that the vocabulary was not modified."
                )
                has_remote_config = False
            else:
                has_remote_config = exists

        # check if the vocab size of the base model is different from the vocab size of the finetuned model
        if (
            vocab_size
            and model_id
            and has_remote_config
            and (vocab_size != model.config.__class__.from_pretrained(model_id).vocab_size)
        ):
            warnings.warn(
                "Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning."
            )
            save_embedding_layers = True
        else:
            save_embedding_layers = False

    if save_embedding_layers and hasattr(model, "get_input_embeddings"):
        for layer in [model.get_input_embeddings(), model.get_output_embeddings()]:
            if not is_embedding_in_target_modules or has_valid_embedding_base_layer(layer):
                # support from version >= 0.6.2
                embedding_module_name = get_embedding_layer_name(model, layer, is_embedding_in_target_modules)
                if embedding_module_name:
                    to_return.update({k: v for k, v in state_dict.items() if embedding_module_name in k})
    elif save_embedding_layers:
        warnings.warn("Could not identify embedding layer(s) because the model is not a 🤗 transformers model.")

    to_return = {k.replace(f".{adapter_name}", ""): v for k, v in to_return.items()}
    return to_return


def _find_mismatched_keys(
    model: torch.nn.Module, peft_model_state_dict: dict[str, torch.Tensor], ignore_mismatched_sizes: bool = False
) -> tuple[dict[str, torch.Tensor], list[tuple[str, tuple[int, ...], tuple[int, ...]]]]:
    if not ignore_mismatched_sizes:
        return peft_model_state_dict, []

    mismatched = []
    state_dict = model.state_dict()
    for key, tensor in peft_model_state_dict.items():
        if key not in state_dict:
            continue

        # see https://github.com/huggingface/transformers/blob/09f9f566de83eef1f13ee83b5a1bbeebde5c80c1/src/transformers/modeling_utils.py#L3858-L3864
        if (state_dict[key].shape[-1] == 1) and (state_dict[key].numel() * 2 == tensor.numel()):
            # This skips size mismatches for 4-bit weights. Two 4-bit values share an 8-bit container, causing size
            # differences. Without matching with module type or paramter type it seems like a practical way to detect
            # valid 4bit weights.
            continue

        if state_dict[key].shape != tensor.shape:
            mismatched.append((key, tensor.shape, state_dict[key].shape))

    for key, _, _ in mismatched:
        del peft_model_state_dict[key]

    return peft_model_state_dict, mismatched


def set_peft_model_state_dict(
    model, peft_model_state_dict, adapter_name="default", ignore_mismatched_sizes: bool = False
):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]):
            The Peft model.
        peft_model_state_dict (`dict`):
            The state dict of the Peft model.
        adapter_name (`str`, *optional*, defaults to `"default"`):
            The name of the adapter whose state dict should be set.
        ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
            Whether to ignore mismatched in the state dict.
    """
    config = model.peft_config[adapter_name]
    state_dict = {}
    if getattr(model, "modules_to_save", None) is not None:
        for key, value in peft_model_state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                for module_name in model.modules_to_save:
                    if module_name in key:
                        key = key.replace(module_name, f"{module_name}.modules_to_save.{adapter_name}")
                        break
            state_dict[key] = value
    else:
        state_dict = peft_model_state_dict

    if config.peft_type in (
        PeftType.LORA,
        PeftType.LOHA,
        PeftType.LOKR,
        PeftType.ADALORA,
        PeftType.IA3,
        PeftType.OFT,
        PeftType.POLY,
        PeftType.LN_TUNING,
        PeftType.BOFT,
        PeftType.VERA,
    ):
        peft_model_state_dict = {}
        parameter_prefix = {
            PeftType.IA3: "ia3_",
            PeftType.LORA: "lora_",
            PeftType.ADALORA: "lora_",
            PeftType.LOHA: "hada_",
            PeftType.LOKR: "lokr_",
            PeftType.OFT: "oft_",
            PeftType.POLY: "poly_",
            PeftType.BOFT: "boft_",
            PeftType.LN_TUNING: "ln_tuning_",
            PeftType.VERA: "vera_lambda_",
        }[config.peft_type]
        for k, v in state_dict.items():
            if parameter_prefix in k:
                suffix = k.split(parameter_prefix)[1]
                if "." in suffix:
                    suffix_to_replace = ".".join(suffix.split(".")[1:])
                    k = k.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
                else:
                    k = f"{k}.{adapter_name}"
                peft_model_state_dict[k] = v
            else:
                peft_model_state_dict[k] = v

        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                model.resize_modules_by_rank_pattern(rank_pattern, adapter_name)
        elif config.peft_type == PeftType.VERA:
            if config.save_projection and "base_model.vera_A" not in peft_model_state_dict:
                raise ValueError(
                    "Specified to load vera_A and vera_B from state dictionary however they were not present!"
                )
            elif not config.save_projection and "base_model.vera_A" in peft_model_state_dict:
                warnings.warn(
                    "Specified to not load vera_A and vera_B from state dictionary however they are present in state"
                    " dictionary! Consider using them to ensure checkpoint loading is correct on all platforms using"
                    " `peft_config.save_projection = True`"
                )
            elif not config.save_projection:  # and no vera_A in state dictionary
                warnings.warn(
                    "Specified to not load vera_A and vera_B from state dictionary. This means we will be relying on"
                    " PRNG initialisation to restore these projections using `config.projection_prng_key`, which may"
                    " not be accurate on all system configurations."
                )
    elif config.is_prompt_learning or config.peft_type == PeftType.ADAPTION_PROMPT:
        peft_model_state_dict = state_dict
    else:
        raise NotImplementedError

    peft_model_state_dict, mismatched_keys = _find_mismatched_keys(
        model, peft_model_state_dict, ignore_mismatched_sizes=ignore_mismatched_sizes
    )
    load_result = model.load_state_dict(peft_model_state_dict, strict=False)
    if config.is_prompt_learning:
        model.prompt_encoder[adapter_name].embedding.load_state_dict(
            {"weight": peft_model_state_dict["prompt_embeddings"]}, strict=True
        )

    if config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
        model.prompt_encoder[adapter_name].load_state_dict(peft_model_state_dict, strict=False)

    if mismatched_keys:
        # see https://github.com/huggingface/transformers/blob/09f9f566de83eef1f13ee83b5a1bbeebde5c80c1/src/transformers/modeling_utils.py#L4039
        mismatched_warning = "\n".join(
            [
                f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                for key, shape1, shape2 in mismatched_keys
            ]
        )
        msg = (
            f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint "
            f"and are being ignored because you passed `ignore_mismatched_sizes=True`: {mismatched_warning}."
        )
        warnings.warn(msg)
    return load_result


def load_peft_weights(model_id: str, device: Optional[str] = None, **hf_hub_download_kwargs) -> dict:
    r"""
    A helper method to load the PEFT weights from the HuggingFace Hub or locally

    Args:
        model_id (`str`):
            The local path to the adapter weights or the name of the adapter to load from the HuggingFace Hub.
        device (`str`):
            The device to load the weights onto.
        hf_hub_download_kwargs (`dict`):
            Additional arguments to pass to the `hf_hub_download` method when loading from the HuggingFace Hub.
    """
    path = (
        os.path.join(model_id, hf_hub_download_kwargs["subfolder"])
        if hf_hub_download_kwargs.get("subfolder", None) is not None
        else model_id
    )

    if device is None:
        device = infer_device()

    if os.path.exists(os.path.join(path, SAFETENSORS_WEIGHTS_NAME)):
        filename = os.path.join(path, SAFETENSORS_WEIGHTS_NAME)
        use_safetensors = True
    elif os.path.exists(os.path.join(path, WEIGHTS_NAME)):
        filename = os.path.join(path, WEIGHTS_NAME)
        use_safetensors = False
    else:
        token = hf_hub_download_kwargs.get("token", None)
        if token is None:
            token = hf_hub_download_kwargs.get("use_auth_token", None)

        hub_filename = (
            os.path.join(hf_hub_download_kwargs["subfolder"], SAFETENSORS_WEIGHTS_NAME)
            if hf_hub_download_kwargs.get("subfolder", None) is not None
            else SAFETENSORS_WEIGHTS_NAME
        )
        has_remote_safetensors_file = file_exists(
            repo_id=model_id,
            filename=hub_filename,
            revision=hf_hub_download_kwargs.get("revision", None),
            repo_type=hf_hub_download_kwargs.get("repo_type", None),
            token=token,
        )
        use_safetensors = has_remote_safetensors_file

        if has_remote_safetensors_file:
            # Priority 1: load safetensors weights
            filename = hf_hub_download(
                model_id,
                SAFETENSORS_WEIGHTS_NAME,
                **hf_hub_download_kwargs,
            )
        else:
            try:
                filename = hf_hub_download(model_id, WEIGHTS_NAME, **hf_hub_download_kwargs)
            except EntryNotFoundError:
                raise ValueError(
                    f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {WEIGHTS_NAME} or {SAFETENSORS_WEIGHTS_NAME} is present at {model_id}."
                )

    if use_safetensors:
        if hasattr(torch.backends, "mps") and (device == torch.device("mps")):
            adapters_weights = safe_load_file(filename, device="cpu")
        else:
            adapters_weights = safe_load_file(filename, device=device)
    else:
        adapters_weights = torch.load(filename, map_location=torch.device(device))

    return adapters_weights
