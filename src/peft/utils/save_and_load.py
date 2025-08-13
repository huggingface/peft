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
import re
import warnings
from typing import Optional

import huggingface_hub
import torch
from huggingface_hub import file_exists, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
from safetensors.torch import load_file as safe_load_file
from transformers.utils import http_user_agent

from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING

from .other import (
    EMBEDDING_LAYER_NAMES,
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    AuxiliaryTrainingWrapper,
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

    # TUNER SPECIFIC CODE
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

        if config.use_dora:
            # Here we take care of a refactor of DoRA which changed lora_magnitude_vector from a ParameterDict to a
            # ModuleDict with a DoraLayer instance. The old parameter is now the "weight" attribute of that layer. Since
            # we want the state_dict format not to change, we remove the "weight" part.
            new_dora_suffix = f"lora_magnitude_vector.{adapter_name}.weight"

            def renamed_dora_weights(k):
                if k.endswith(new_dora_suffix):
                    k = k[:-7]  # remove ".weight"
                return k

            to_return = {renamed_dora_weights(k): v for k, v in to_return.items()}

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

    elif config.peft_type == PeftType.VERA:
        vera_prefix = PEFT_TYPE_TO_PREFIX_MAPPING[config.peft_type]
        to_return = {k: state_dict[k] for k in state_dict if vera_prefix in k}
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
    elif config.peft_type == PeftType.XLORA:
        to_return = {k: state_dict[k] for k in state_dict if "internal_xlora_classifier" in k}
    elif config.peft_type == PeftType.VBLORA:
        to_return = {}
        # choose the most efficient dtype for indices
        if config.num_vectors < 2**8:
            indices_dtype = torch.uint8
        elif config.num_vectors < 2**15:
            indices_dtype = torch.int16
        elif config.num_vectors < 2**31:
            indices_dtype = torch.int32
        else:
            indices_dtype = torch.int64
        if config.save_only_topk_weights:
            # in save_only_topk_weights mode, we save topk_indices and topk_weights for parameter efficiency
            for k in state_dict:
                if "vblora_logits" in k:
                    logits, indices = state_dict[k].topk(config.topk)
                    to_return.update({k + "_topk_indices": indices.to(dtype=indices_dtype)})
                    to_return.update({k + "_topk_weights": torch.softmax(logits, dim=-1)[:, :, :-1].contiguous()})
        else:
            to_return = {k: state_dict[k] for k in state_dict if "vblora_logits" in k}
        to_return["base_model.vblora_vector_bank." + adapter_name] = state_dict[
            "base_model.vblora_vector_bank." + adapter_name
        ]
    elif config.peft_type in list(PeftType):
        prefix = PEFT_TYPE_TO_PREFIX_MAPPING[config.peft_type]
        to_return = {k: state_dict[k] for k in state_dict if prefix in k}
    else:
        raise ValueError(f"Unknown PEFT type passed: {config.peft_type}")

    # ADDITIONAL TRAINING MODULES / MODULES_TO_SAVE
    for name, module in model.named_modules():
        if isinstance(module, AuxiliaryTrainingWrapper):
            # Compute the module-relative state dict to make it easier for the adapter to fetch the appropriate
            # keys that the module thinks need to be saved. We cannot rely on `.state_dict()` internally of the
            # module since accelerators like DeepSpeed require special handling which is done for the model
            # state dict from above but most likely not in the module itself. See #2450.
            module_state_dict = {
                k.removeprefix(f"{name}."): v for k, v in state_dict.items() if k.startswith(f"{name}.")
            }
            to_return.update(
                {f"{name}.{k}": v for k, v in module.adapter_state_dict(adapter_name, module_state_dict).items()}
            )

    # DEAL WITH EMBEDDINGS
    # check the common embedding layers in `target_modules` to reset `save_embedding_layers` if necessary
    is_embedding_in_target_modules = False
    if (
        save_embedding_layers == "auto"
        and hasattr(config, "target_modules")
        and any(k in config.target_modules for k in EMBEDDING_LAYER_NAMES)
        and config.peft_type != PeftType.TRAINABLE_TOKENS
    ):
        warnings.warn("Setting `save_embedding_layers` to `True` as embedding layers found in `target_modules`.")
        save_embedding_layers = is_embedding_in_target_modules = True
    elif save_embedding_layers == "auto":
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
        model_id = getattr(config, "base_model_name_or_path", None)

        # For some models e.g. diffusers the text config file is stored in a subfolder
        # we need to make sure we can download that config.
        has_base_config = False

        # ensure that this check is not performed in HF offline mode, see #1452
        if model_id is not None:
            local_config_exists = os.path.exists(os.path.join(model_id, "config.json"))
            exists = local_config_exists or check_file_exists_on_hf_hub(model_id, "config.json")
            if exists is None:
                # check failed, could not determine if it exists or not
                warnings.warn(
                    f"Could not find a config file in {model_id} - will assume that the vocabulary was not modified."
                )
                has_base_config = False
            else:
                has_base_config = exists

        # check if the vocab size of the base model is different from the vocab size of the finetuned model
        if (
            vocab_size
            and model_id
            and has_base_config
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

    # REMOVE ADAPTER NAME
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
            # differences. Without matching with module type or parameter type it seems like a practical way to detect
            # valid 4bit weights.
            continue

        if state_dict[key].shape != tensor.shape:
            mismatched.append((key, tensor.shape, state_dict[key].shape))

    for key, _, _ in mismatched:
        del peft_model_state_dict[key]

    return peft_model_state_dict, mismatched


def _insert_adapter_name_into_state_dict(
    state_dict: dict[str, torch.Tensor], adapter_name: str, parameter_prefix: str
) -> dict[str, torch.Tensor]:
    """Utility function to remap the state_dict keys to fit the PEFT model by inserting the adapter name."""
    peft_model_state_dict = {}
    for key, val in state_dict.items():
        if parameter_prefix in key:
            suffix = key.split(parameter_prefix)[1]
            if "." in suffix:
                suffix_to_replace = ".".join(suffix.split(".")[1:])
                key = key.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
            else:
                key = f"{key}.{adapter_name}"
            peft_model_state_dict[key] = val
        else:
            peft_model_state_dict[key] = val
    return peft_model_state_dict


def set_peft_model_state_dict(
    model,
    peft_model_state_dict,
    adapter_name="default",
    ignore_mismatched_sizes: bool = False,
    low_cpu_mem_usage: bool = False,
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
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            This argument must be `True` if the `model` was loaded with adapter weights on the meta device, e.g. after
            calling `inject_adapter_in_model` with `low_cpu_mem_usage=True`. Otherwise, leave it as `False`.

    """
    config = model.peft_config[adapter_name]
    state_dict = peft_model_state_dict

    # handle auxiliary training wrappers such as ModulesToSaveWrapper and TrainableTokensWrapper by getting each of
    # them and translating saved state dict key (which does not include the adapter name) to loaded state dict key
    # (which includes the adapter name).
    for name, module in model.named_modules():
        if isinstance(module, AuxiliaryTrainingWrapper):
            # Not every module has a 1:1 mapping. ModulesToSaveWrapper, for example, removes the
            # `modules_to_save.{adapter_name}.` prefix. This prefix must be restored when loading the model from the
            # saved state dict which is why we fetch a load key map from the wrapper.
            key_map = module.adapter_state_dict_load_map(adapter_name)
            for k in key_map:
                lookup_key = f"{name}.{k}"
                store_key = f"{name}.{key_map[k]}"

                state_dict[store_key] = peft_model_state_dict[lookup_key]

                # delete the old key from the previous `state_dict = peft_model_state_dict` statement.
                del state_dict[lookup_key]

    if config.is_prompt_learning or config.peft_type == PeftType.ADAPTION_PROMPT:
        peft_model_state_dict = state_dict
    elif config.peft_type == PeftType.XLORA:
        peft_model_state_dict = state_dict
    elif config.peft_type in PEFT_TYPE_TO_PREFIX_MAPPING:
        peft_model_state_dict = {}
        parameter_prefix = PEFT_TYPE_TO_PREFIX_MAPPING[config.peft_type]
        if config.peft_type == PeftType.VBLORA and config.save_only_topk_weights:
            num_vectors, _ = model.vblora_vector_bank[adapter_name].shape
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                # in save_only_topk_weights mode, only topk_indices and topk_weights are saved
                # note that topk_indices and topk_weights serve as an efficient representation of the logits
                # so we need to recover the logits from the topk_indices and topk_weights
                if "_topk_indices" in k:
                    v = state_dict[k].to(torch.long)
                    original_key = k.replace("_topk_indices", "")
                    # find the corresponding topk_weights from the state_dict
                    topk_weights = state_dict[k.replace("_topk_indices", "_topk_weights")]
                    # as we only save the first k-1 topk_weights, here we recover the last one
                    topk_weights = torch.cat([topk_weights, 1 - topk_weights.sum(-1, keepdim=True)], dim=-1)
                    # convert the weights to logits
                    topk_logits = torch.log(topk_weights)
                    matrix = (
                        torch.zeros([*(topk_logits.shape[:-1]), num_vectors])
                        .fill_(float("-inf"))
                        .to(topk_logits.device)
                        .scatter(-1, v, topk_logits)
                    )
                    # add logits to the state_dict
                    state_dict[original_key] = matrix
                    # delete the topk_indices and topk_weights from the state_dict
                    del state_dict[k]
                    del state_dict[k.replace("_topk_indices", "_topk_weights")]

        peft_model_state_dict = _insert_adapter_name_into_state_dict(
            state_dict, adapter_name=adapter_name, parameter_prefix=parameter_prefix
        )

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
        elif config.peft_type == PeftType.LORA:
            # Here we take care of a refactor of DoRA which changed lora_magnitude_vector from a ParameterDict to a
            # ModuleDict with a DoraLayer instance. The old parameter is now the "weight" attribute of that layer.
            old_dora_suffix = f"lora_magnitude_vector.{adapter_name}"

            def renamed_dora_weights(k):
                if k.endswith(old_dora_suffix):
                    k = k + ".weight"
                return k

            peft_model_state_dict = {renamed_dora_weights(k): v for k, v in peft_model_state_dict.items()}
        elif config.peft_type == PeftType.OFT:
            if any(".oft_r." in key for key in peft_model_state_dict):
                raise ValueError(
                    "Trying to load old OFT checkpoint, which is no longer supported. Please install PEFT <= v0.15.2 to load it or train a new OFT adapter."
                )
    else:
        raise NotImplementedError

    peft_model_state_dict, mismatched_keys = _find_mismatched_keys(
        model, peft_model_state_dict, ignore_mismatched_sizes=ignore_mismatched_sizes
    )
    if low_cpu_mem_usage:
        load_result = model.load_state_dict(peft_model_state_dict, strict=False, assign=True)
        # ensure that the correct device is set
        for module in model.modules():
            if hasattr(module, "_move_adapter_to_device_of_base_layer"):
                module._move_adapter_to_device_of_base_layer(adapter_name)
    else:
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


# TODO: remove this function, use vanilla torch.load as soon as torch < 2.6.0 is no longer supported
def torch_load(*args, weights_only=True, **kwargs):
    """Call torch.load and handle weights_only.

    Defaults to weights_only=True to anticipate upcoming switch on the PyTorch side.

    """
    return torch.load(*args, weights_only=weights_only, **kwargs)


def load_peft_weights(
    model_id: str, device: Optional[str] = None, key_mapping: Optional[dict[str, str]] = None, **hf_hub_download_kwargs
) -> dict:
    r"""
    A helper method to load the PEFT weights from the HuggingFace Hub or locally

    Args:
        model_id (`str`):
            The local path to the adapter weights or the name of the adapter to load from the HuggingFace Hub.
        device (`str`):
            The device to load the weights onto.
        key_mapping (dict, *optional*, defaults to None)
            Extra mapping of PEFT `state_dict` keys applied before loading the `state_dict`. When this mapping is
            applied, the PEFT-specific `"base_model.model"` prefix is removed beforehand and the adapter name (e.g.
            `"default"`) is not inserted yet. Only pass this argument if you know what you're doing.
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

    def get_hub_filename(use_safetensors=True):
        weights_name = SAFETENSORS_WEIGHTS_NAME if use_safetensors else WEIGHTS_NAME
        return (
            os.path.join(hf_hub_download_kwargs["subfolder"], weights_name)
            if hf_hub_download_kwargs.get("subfolder", None) is not None
            else weights_name
        )

    if "user_agent" not in hf_hub_download_kwargs:
        hf_hub_download_kwargs["user_agent"] = http_user_agent()

    if os.path.exists(os.path.join(path, SAFETENSORS_WEIGHTS_NAME)):
        filename = os.path.join(path, SAFETENSORS_WEIGHTS_NAME)
        use_safetensors = True
    elif os.path.exists(os.path.join(path, WEIGHTS_NAME)):
        filename = os.path.join(path, WEIGHTS_NAME)
        use_safetensors = False
    elif huggingface_hub.constants.HF_HUB_OFFLINE:
        # if in offline mode, check if we can find the adapter file locally
        hub_filename = get_hub_filename(use_safetensors=True)
        hf_hub_download_kwargs.pop("local_files_only", None)
        try:
            filename = hf_hub_download(model_id, hub_filename, local_files_only=True, **hf_hub_download_kwargs)
            use_safetensors = True
        except LocalEntryNotFoundError:
            # Could not find safetensors, try pickle. If this also fails, it's fine to let the error be raised here, as
            # it means that the user tried to load a non-cached model in offline mode.
            hub_filename = get_hub_filename(use_safetensors=False)
            filename = hf_hub_download(model_id, hub_filename, local_files_only=True, **hf_hub_download_kwargs)
            use_safetensors = False
    else:
        token = hf_hub_download_kwargs.get("token", None)
        if token is None:
            token = hf_hub_download_kwargs.get("use_auth_token", None)

        hub_filename = get_hub_filename(use_safetensors=True)
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
        adapters_weights = torch_load(filename, map_location=torch.device(device))

    if not key_mapping:
        remapped_adapters_weights = adapters_weights
    else:
        # See discussion in https://github.com/huggingface/transformers/pull/38627
        # Remap adapter weight names according to the provided key_mapping.
        remapped_adapters_weights = {}
        for key, val in adapters_weights.items():
            if key.startswith("base_model.model."):
                prefix = "base_model.model."
            elif key.startswith("base_model."):
                prefix = "base_model."
            else:
                raise ValueError(
                    "An error occurred while trying to load a PEFT state_dict with key_mapping. This should not "
                    "happen. Please open an issue on https://github.com/huggingface/peft/issues and report the error."
                )

            key = key.removeprefix(prefix)  # the key map assumes that there is no prefix
            for pattern, replacement in key_mapping.items():
                key_new, n_replace = re.subn(pattern, replacement, key)
                # Early exit of the loop
                if n_replace > 0:
                    key = key_new
                    break
            key_with_prefix = f"{prefix}{key}"
            remapped_adapters_weights[key_with_prefix] = val

    return remapped_adapters_weights
