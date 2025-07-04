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

import collections
import copy
import inspect
import os
import warnings
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import packaging.version
import torch
import transformers
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
from accelerate.utils import get_balanced_memory, named_module_tensors
from huggingface_hub import HfFileSystem, ModelCard, ModelCardData, hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file as safe_save_file
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import Cache, DynamicCache, EncoderDecoderCache, HybridCache, PreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput, SequenceClassifierOutput, TokenClassifierOutput
from transformers.utils import PushToHubMixin

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils.constants import DUMMY_MODEL_CONFIG
from peft.utils.integrations import init_empty_weights
from peft.utils.other import create_attention_mask, set_additional_trainable_modules

from . import __version__
from .config import PeftConfig
from .mapping import PEFT_TYPE_TO_CONFIG_MAPPING, PEFT_TYPE_TO_PREFIX_MAPPING, PEFT_TYPE_TO_TUNER_MAPPING
from .utils import (
    SAFETENSORS_WEIGHTS_NAME,
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    WEIGHTS_NAME,
    PeftType,
    TaskType,
    _get_batch_size,
    _prepare_prompt_learning_config,
    _set_adapter,
    _set_trainable,
    get_peft_model_state_dict,
    id_tensor_storage,
    infer_device,
    load_peft_weights,
    map_cache_to_layer_device_map,
    set_peft_model_state_dict,
    shift_tokens_right,
)


class PeftModel(PushToHubMixin, torch.nn.Module):
    """
    Base model encompassing various Peft methods.

    Args:
        model ([`~transformers.PreTrainedModel`]): The base transformer model used for Peft.
        peft_config ([`PeftConfig`]): The configuration of the Peft model.
        adapter_name (`str`,  *optional*): The name of the adapter, defaults to `"default"`.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter weights
            using float16 and bfloat16 to float32, as this is typically required for stable training, and only affect
            select PEFT tuners.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading loading process.

            <Tip>

            Don't use `low_cpu_mem_usage=True` when creating a new PEFT adapter for training.

            </Tip>

    **Attributes**:
        - **base_model** ([`torch.nn.Module`]) -- The base transformer model used for Peft.
        - **peft_config** ([`PeftConfig`]) -- The configuration of the Peft model.
        - **modules_to_save** (`list` of `str`) -- The list of sub-module names to save when
            saving the model.
        - **prompt_encoder** ([`PromptEncoder`]) -- The prompt encoder used for Peft if
            using [`PromptLearningConfig`].
        - **prompt_tokens** (`torch.Tensor`) -- The virtual prompt tokens used for Peft if
            using [`PromptLearningConfig`].
        - **transformer_backbone_name** (`str`) -- The name of the transformer
            backbone in the base model if using [`PromptLearningConfig`].
        - **word_embeddings** (`torch.nn.Embedding`) -- The word embeddings of the transformer backbone
            in the base model if using [`PromptLearningConfig`].
    """

    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        adapter_name: str = "default",
        autocast_adapter_dtype: bool = True,
        low_cpu_mem_usage: bool = False,
    ) -> None:
        super().__init__()
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type
        # These args are special PEFT arguments that users can pass. They need to be removed before passing them to
        # forward.
        self.special_peft_forward_args = {"adapter_names"}

        self._is_prompt_learning = peft_config.is_prompt_learning
        if self._is_prompt_learning:
            self._peft_config = {adapter_name: peft_config}
            self.base_model = model
            self.add_adapter(adapter_name, peft_config, low_cpu_mem_usage=low_cpu_mem_usage)
        else:
            self._peft_config = None
            cls = PEFT_TYPE_TO_TUNER_MAPPING[peft_config.peft_type]
            ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
            with ctx():
                self.base_model = cls(model, {adapter_name: peft_config}, adapter_name)

        if hasattr(self.base_model, "_cast_adapter_dtype"):
            self.base_model._cast_adapter_dtype(
                adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype
            )

        if getattr(model, "is_gradient_checkpointing", True):
            model = self.prepare_model_for_gradient_checkpointing(model)

        # the `pretraining_tp` is set for some models to simulate Tensor Parallelism during inference to avoid
        # numerical differences, https://github.com/pytorch/pytorch/issues/76232 - to avoid any unexpected
        # behavior we disable that in this line.
        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pretraining_tp"):
            self.base_model.config.pretraining_tp = 1

    @property
    def peft_config(self) -> dict[str, PeftConfig]:
        if self._is_prompt_learning:
            return self._peft_config
        return self.base_model.peft_config

    @property
    def active_adapters(self) -> list[str]:
        try:
            adapters = self.base_model.active_adapters
            if not isinstance(adapters, list):
                # Base model is probably a transformers model, see:
                # https://github.com/huggingface/transformers/pull/30790#issuecomment-2253808249
                # Unfortunately, transformers models also have an active_adapters method but it's 1) not a property and
                # 2) calling it fails because the base model (usually) has no loaded adapter. The base model can be a
                # transformers model for prompt learning, where the base model is not wrapped in a LoraModel or similar.
                adapters = self.active_adapter
                if isinstance(adapters, str):
                    adapters = [adapters]
        except AttributeError:
            adapters = self.active_adapter
            if isinstance(adapters, str):
                adapters = [adapters]
        return adapters

    @peft_config.setter
    def peft_config(self, value: dict[str, PeftConfig]):
        if self._is_prompt_learning:
            self._peft_config = value
        else:
            self.base_model.peft_config = value

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        selected_adapters: Optional[list[str]] = None,
        save_embedding_layers: Union[str, bool] = "auto",
        is_main_process: bool = True,
        path_initial_model_for_weight_conversion: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        r"""
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        reloaded using the [`PeftModel.from_pretrained`] class method, and also used by the [`PeftModel.push_to_hub`]
        method.

        Args:
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            safe_serialization (`bool`, *optional*):
                Whether to save the adapter files in safetensors format, defaults to `True`.
            selected_adapters (`List[str]`,  *optional*):
                A list of adapters to be saved. If `None`, will default to all adapters.
            save_embedding_layers (`Union[bool, str]`, *optional*, defaults to `"auto"`):
                If `True`, save the embedding layers in addition to adapter weights. If `auto`, checks the common
                embedding layers `peft.utils.other.EMBEDDING_LAYER_NAMES` in config's `target_modules` when available.
                and automatically sets the boolean flag. This only works for 🤗 transformers models.
            is_main_process (`bool`, *optional*):
                Whether the process calling this is the main process or not. Will default to `True`. Will not save the
                checkpoint if not on the main process, which is important for multi device setups (e.g. DDP).
            path_initial_model_for_weight_conversion (`str, *optional*`):
                The path to the initialized adapter, which is obtained after initializing the model with
                PiSSA/CorDA/OLoRA and before performing any training. When `path_initial_model_for_weight_conversion`
                is not None, the difference in adapter before and after fine-tuning is calculated. This difference can
                be represented as the parameters of a standard LoRA adapter. Using this converted adapter does not
                require changes to the base model, thus conveniently allowing the use of multiple PiSSA/CorDA/OLoRA
                adapters with LoRA adapters, and the activation or deactivation of any adapters. Note that this
                conversion is not supported if `rslora` is used in combination with `rank_pattern` or `alpha_pattern`.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the `push_to_hub` method.

        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        if selected_adapters is None:
            selected_adapters = list(self.peft_config.keys())
        else:
            if any(
                selected_adapter_name not in list(self.peft_config.keys())
                for selected_adapter_name in selected_adapters
            ):
                raise ValueError(
                    f"You passed an invalid `selected_adapters` arguments, current supported adapter names are"
                    f" {list(self.peft_config.keys())} - got {selected_adapters}."
                )

        def save_mutated_as_lora(peft_config, path_initial_model_for_weight_conversion, output_state_dict, kwargs):
            if peft_config.use_rslora and (peft_config.rank_pattern or peft_config.alpha_pattern):
                msg = (
                    "Passing `path_initial_model_for_weight_conversion` to `save_pretrained` is not supported when "
                    "using `rank_pattern` or `alpha_pattern` at the same time as `use_rslora=True`."
                )
                raise ValueError(msg)

            if not any(
                str(peft_config.init_lora_weights).lower().startswith(prefix)
                for prefix in ["pissa", "corda", "olora", "true"]
            ):
                warnings.warn(
                    "`path_initial_model_for_weight_conversion` only works for converting a PiSSA/CorDA/OLoRA adapter to "
                    "a LoRA adapter"
                )
            initial_adapter_name = os.path.basename(path_initial_model_for_weight_conversion)
            try:
                self.load_adapter(
                    os.path.dirname(path_initial_model_for_weight_conversion),
                    subfolder=initial_adapter_name,
                    adapter_name=initial_adapter_name,
                )
                is_pissa = str(self.peft_config[initial_adapter_name].init_lora_weights).lower().startswith("pissa")
                is_corda = str(self.peft_config[initial_adapter_name].init_lora_weights).lower() == "corda"
                is_olora = str(self.peft_config[initial_adapter_name].init_lora_weights).lower() == "olora"
                if is_pissa or is_corda or is_olora:
                    raise ValueError(
                        "The `init_lora_weights` parameter of the initial adapter should be set to `True`. "
                        "Otherwise, `self.load_adapter` will subtract the decomposed values again based on the "
                        "residual model."
                    )
                output_state_dict = self.base_model.subtract_mutated_init(
                    output_state_dict, initial_adapter_name, kwargs
                )
            finally:
                self.delete_adapter(initial_adapter_name)
            return output_state_dict

        if is_main_process:
            os.makedirs(save_directory, exist_ok=True)
            self.create_or_update_model_card(save_directory)

        for adapter_name in selected_adapters:
            peft_config = self.peft_config[adapter_name]
            # save only the trainable weights
            output_state_dict = get_peft_model_state_dict(
                self,
                state_dict=kwargs.get("state_dict", None),
                adapter_name=adapter_name,
                save_embedding_layers=save_embedding_layers,
            )
            output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
            os.makedirs(output_dir, exist_ok=True)

            if is_main_process and safe_serialization:
                # Section copied from: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2111-L2134
                # Safetensors does not allow tensor aliasing.
                # We're going to remove aliases before saving
                ptrs = collections.defaultdict(list)
                for name, tensor in output_state_dict.items():
                    # Sometimes in the state_dict we have non-tensor objects.
                    # e.g. in bitsandbytes we have some `str` objects in the state_dict
                    if isinstance(tensor, torch.Tensor):
                        ptrs[id_tensor_storage(tensor)].append(name)
                    else:
                        # In the non-tensor case, fall back to the pointer of the object itself
                        ptrs[id(tensor)].append(name)

                # These are all the pointers of shared tensors.
                shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}

                for _, names in shared_ptrs.items():
                    # Here we just clone the shared tensors to avoid tensor aliasing which is
                    # not supported in safetensors.
                    for shared_tensor_name in names[1:]:
                        output_state_dict[shared_tensor_name] = output_state_dict[shared_tensor_name].clone()
                if path_initial_model_for_weight_conversion is not None:
                    peft_config = copy.deepcopy(peft_config)
                    peft_config.init_lora_weights = True
                    peft_config.save_pretrained(path_initial_model_for_weight_conversion)
                    output_state_dict = save_mutated_as_lora(
                        peft_config, path_initial_model_for_weight_conversion, output_state_dict, kwargs
                    )
                safe_save_file(
                    output_state_dict,
                    os.path.join(output_dir, SAFETENSORS_WEIGHTS_NAME),
                    metadata={"format": "pt"},
                )
            elif is_main_process:
                if path_initial_model_for_weight_conversion is not None:
                    peft_config = copy.deepcopy(peft_config)
                    peft_config.init_lora_weights = True
                    peft_config.save_pretrained(path_initial_model_for_weight_conversion)
                    output_state_dict = save_mutated_as_lora(
                        peft_config, path_initial_model_for_weight_conversion, output_state_dict, kwargs
                    )
                torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            # save the config and change the inference mode to `True`
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    self.base_model.__dict__.get("name_or_path", None)
                    if peft_config.is_prompt_learning
                    else self.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True

            if peft_config.task_type is None:
                # deal with auto mapping
                base_model_class = self._get_base_model_class(
                    is_prompt_tuning=peft_config.is_prompt_learning,
                )
                parent_library = base_model_class.__module__

                auto_mapping_dict = {
                    "base_model_class": base_model_class.__name__,
                    "parent_library": parent_library,
                }
            else:
                auto_mapping_dict = None

            if is_main_process:
                if path_initial_model_for_weight_conversion is not None:
                    peft_config.init_lora_weights = True
                    peft_config.r *= 2
                    if not peft_config.use_rslora:
                        peft_config.lora_alpha *= 2
                    else:
                        # with rslora, we have scaling = alpha / sqrt(r), we thus adjust alpha to keep the same scaling
                        peft_config.lora_alpha *= 2**0.5

                    if peft_config.rank_pattern:
                        peft_config.rank_pattern = {key: 2 * val for key, val in peft_config.rank_pattern.items()}
                    if peft_config.alpha_pattern:
                        peft_config.alpha_pattern = {key: 2 * val for key, val in peft_config.alpha_pattern.items()}

                peft_config.save_pretrained(output_dir, auto_mapping_dict=auto_mapping_dict)
            peft_config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(
        cls,
        model: torch.nn.Module,
        model_id: Union[str, os.PathLike],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        autocast_adapter_dtype: bool = True,
        ephemeral_gpu_offload: bool = False,
        low_cpu_mem_usage: bool = False,
        key_mapping: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> PeftModel:
        r"""
        Instantiate a PEFT model from a pretrained model and loaded PEFT weights.

        Note that the passed `model` may be modified inplace.

        Args:
            model ([`torch.nn.Module`]):
                The model to be adapted. For 🤗 Transformers models, the model should be initialized with the
                [`~transformers.PreTrainedModel.from_pretrained`].
            model_id (`str` or `os.PathLike`):
                The name of the PEFT configuration to use. Can be either:
                    - A string, the `model id` of a PEFT configuration hosted inside a model repo on the Hugging Face
                      Hub.
                    - A path to a directory containing a PEFT configuration file saved using the `save_pretrained`
                      method (`./my_peft_config_directory/`).
            adapter_name (`str`, *optional*, defaults to `"default"`):
                The name of the adapter to be loaded. This is useful for loading multiple adapters.
            is_trainable (`bool`, *optional*, defaults to `False`):
                Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and can only be
                used for inference.
            config ([`~peft.PeftConfig`], *optional*):
                The configuration object to use instead of an automatically loaded configuration. This configuration
                object is mutually exclusive with `model_id` and `kwargs`. This is useful when configuration is already
                loaded before calling `from_pretrained`.
            autocast_adapter_dtype (`bool`, *optional*):
                Whether to autocast the adapter dtype. Defaults to `True`. Only relevant for specific adapter types.
            ephemeral_gpu_offload (`bool`, *optional*):
                Whether to use ephemeral GPU offloading for partially loaded modules. Defaults to `False`. This is
                useful when parts of the model and/or components (such as adapters) are kept in CPU memory until they
                are needed. Rather than perform expensive operations on small data, the data is transferred to the GPU
                on-demand, the operation(s) performed, and the results moved back to CPU memory. This brings a slight
                momentary VRAM overhead but gives orders of magnitude speedup in certain cases.
            low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
                Create empty adapter weights on meta device before loading the saved weights. Useful to speed up the
                process.
            torch_device (`str`, *optional*, defaults to None):
                The device to load the adapter on. If `None`, the device will be inferred.
            key_mapping (dict, *optional*, defaults to None)
                Extra mapping of PEFT `state_dict` keys applied before loading the `state_dict`. When this mapping is
                applied, the PEFT-specific `"base_model.model"` prefix is removed beforehand and the adapter name (e.g.
                `"default"`) is not inserted yet. Only pass this argument if you know what you're doing.
            kwargs: (`optional`):
                Additional keyword arguments passed along to the specific PEFT configuration class.
        """
        from .auto import MODEL_TYPE_TO_PEFT_MODEL_MAPPING
        from .tuners import XLoraConfig, XLoraModel

        # load the config
        if config is None:
            config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(
                    model_id,
                    subfolder=kwargs.get("subfolder", None),
                    revision=kwargs.get("revision", None),
                    cache_dir=kwargs.get("cache_dir", None),
                    use_auth_token=kwargs.get("use_auth_token", None),
                    token=kwargs.get("token", None),
                )
            ].from_pretrained(model_id, **kwargs)
        elif isinstance(config, PeftConfig):
            config.inference_mode = not is_trainable
        else:
            raise ValueError(f"The input config must be a PeftConfig, got {config.__class__}")

        # See discussion in https://github.com/huggingface/transformers/pull/38627
        # Some transformers models can have a _checkpoint_conversion_mapping dict that is used to map state_dicts
        # stemming from updated model architectures so that they still correspond to the initial architecture. When
        # loading a PEFT state_dict created with the initial architecture on a model with the new architecture, we need
        # to map it too according to the same rules. Note that we skip prompt learning methods. This is because they
        # don't have the "base_model.model." prefix, which we need to remove before mapping. Instead just using
        # "base_model.". This could be fine, we could only remove "base_model.", However, the subsequent sub-module
        # could also be called "model", resulting in what looks like "base_model.model.". To avoid this confusion, we
        # skip prompt learning. Since it applies itself directly to the pre-trained model (unlike LoRA et al that target
        # sub-modules), skipping should be fine.
        if (key_mapping is None) and (not config.is_prompt_learning):
            key_mapping = getattr(model, "_checkpoint_conversion_mapping", {})

        # Runtime configuration, if supported
        if hasattr(config, "runtime_config"):
            config.runtime_config.ephemeral_gpu_offload = ephemeral_gpu_offload
        else:
            if ephemeral_gpu_offload:
                warnings.warn("Ephemeral GPU offloading is not supported for this model. Ignoring.")

        if hasattr(model, "hf_device_map"):
            weight_map = dict(named_module_tensors(model, recurse=True))

            # recreate the offload_index for disk-offloaded modules: we need to know the location in storage of each weight
            # before the offload hook is removed from the model
            disk_modules = set()
            index = None
            for name, module in model.named_modules():
                if hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "original_devices"):
                    if hasattr(module._hf_hook.weights_map, "dataset"):
                        index = module._hf_hook.weights_map.dataset.index
                    for key in module._hf_hook.original_devices.keys():
                        if module._hf_hook.original_devices[key] == torch.device("meta"):
                            disk_modules.add(str(name) + "." + str(key))

            if disk_modules and not kwargs.get("use_safetensors", True):
                raise ValueError("Disk offloading currently only supported for safetensors")

            if index:
                offload_index = {
                    p: {
                        "safetensors_file": index[p]["safetensors_file"],
                        "weight_name": p,
                        "dtype": str(weight_map[p].dtype).replace("torch.", ""),
                    }
                    for p in weight_map.keys()
                    if p in disk_modules
                }
                kwargs["offload_index"] = offload_index

        if (getattr(model, "hf_device_map", None) is not None) and len(
            set(model.hf_device_map.values()).intersection({"cpu", "disk"})
        ) > 0:
            remove_hook_from_submodules(model)

        if config.is_prompt_learning and is_trainable:
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
        else:
            config.inference_mode = not is_trainable
        if isinstance(getattr(model, "base_model", None), XLoraModel):
            if not isinstance(config, XLoraConfig):
                raise TypeError(f"Expected 'XLoraConfig', got '{type(config)}' instead.")
            if "adapters" in kwargs:
                config.adapters = kwargs["adapters"]
            else:
                # If the path is on HF hub, then we get the adapter names to create a subfolders list which tells
                # `load_adapter` where the adapters are.
                if not os.path.exists(model_id):
                    s = HfFileSystem()

                    # The names of the adapters which must be in folders
                    adapter_names = [
                        file["name"][len(model_id) + 1 :] for file in s.ls(model_id) if file["type"] == "directory"
                    ]
                    # Prepare a dict of adapter paths, which really just point to the hf id; we will use the subfolders
                    adapter_paths = {}
                    for adapter_name in adapter_names:
                        adapter_paths[adapter_name] = os.path.join(model_id, model_id)
                    config.adapters = adapter_paths
                    config._subfolders = adapter_names
                else:
                    if "adapters" not in kwargs:
                        raise ValueError("If model_id is a local path, then `adapters` must be passed in kwargs.")

        if config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
            model = cls(
                model,
                config,
                adapter_name,
                autocast_adapter_dtype=autocast_adapter_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )
        else:
            model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](
                model,
                config,
                adapter_name,
                autocast_adapter_dtype=autocast_adapter_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )

        load_result = model.load_adapter(
            model_id,
            adapter_name,
            is_trainable=is_trainable,
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            key_mapping=key_mapping,
            **kwargs,
        )

        # 1. Remove VB-LoRA vector bank, since it's a shared parameter set via the VBLoRAModel
        # 2. Remove the prompt encoder, as it does not need to be part of the checkpoint
        missing_keys = [
            k for k in load_result.missing_keys if "vblora_vector_bank" not in k and "prompt_encoder" not in k
        ]
        if missing_keys:
            # Let's warn here since (in contrast to load_adapter) we don't return the load result, so it could be quite
            # difficult for users to even notice that something might have gone wrong here. As we filter out non PEFT
            # keys from the missing keys, this gives no false positives.

            # careful: if the wording of the warning is changed, adjust the unit tests accordingly!
            warn_message = f"Found missing adapter keys while loading the checkpoint: {missing_keys}."

            prefix = PEFT_TYPE_TO_PREFIX_MAPPING.get(config.peft_type)
            if prefix and adapter_name in prefix:
                warn_message += (
                    f"Adapter name {adapter_name} should not be contained in the prefix {prefix}."
                    "This could be the potential reason for missing adapter keys."
                )

            warnings.warn(warn_message)

        return model

    def _setup_prompt_encoder(self, adapter_name: str):
        config = self.peft_config[adapter_name]
        if not hasattr(self, "prompt_encoder"):
            self.prompt_encoder = torch.nn.ModuleDict({})
            self.prompt_tokens = {}
        transformer_backbone = None
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False
            if isinstance(module, PreTrainedModel):
                # Make sure to freeze Tranformers model
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name
        if transformer_backbone is None:
            transformer_backbone = self.base_model

        if config.num_transformer_submodules is None:
            config.num_transformer_submodules = 2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1

        # determine the word embeddings
        word_embeddings = None
        try:
            # First try to find the word embeddings based on the module name, this should work for models like Bert,
            # Roberta, Deberta, etc.
            word_embeddings = self.base_model.get_submodule("embeddings.word_embeddings")
        except AttributeError:
            pass

        if word_embeddings is None:
            # Word embeddings could not be determined. Next try to guess them by checking which parameter has the size
            # of the vocab.
            for named_param, value in list(transformer_backbone.named_parameters()):
                # for ZeRO-3, the tensor is sharded across accelerators and deepspeed modifies it to a tensor with shape
                # [0] the actual unsharded shape is stored in "ds_shape" attribute special handling is needed in case
                # the model is initialized in deepspeed.zero.Init() context or HfDeepSpeedConfig has been called before
                # For reference refer to issue: https://github.com/huggingface/peft/issues/996
                deepspeed_distributed_tensor_shape = getattr(value, "ds_shape", None)

                # Handle VLM case with separate text and vision configs
                if hasattr(self.base_model.config, "get_text_config"):
                    vocab_size = self.base_model.config.get_text_config().vocab_size
                # below: for older transformers versions before get_text_config was added
                elif "text_config" in self.base_model.config:
                    vocab_size = self.base_model.config.text_config.vocab_size
                else:
                    vocab_size = self.base_model.config.vocab_size

                if value.shape[0] == vocab_size or (
                    deepspeed_distributed_tensor_shape is not None
                    and deepspeed_distributed_tensor_shape[0] == vocab_size
                ):
                    word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
                    break

        self.word_embeddings = word_embeddings
        model_cls = PEFT_TYPE_TO_TUNER_MAPPING[config.peft_type]

        if config.peft_type in (PeftType.PROMPT_TUNING, PeftType.MULTITASK_PROMPT_TUNING, PeftType.CPT):
            prompt_encoder = model_cls(config, self.word_embeddings)
        elif config.peft_type == PeftType.P_TUNING:
            prompt_encoder = model_cls(config)
        elif config.peft_type == PeftType.PREFIX_TUNING:
            # prefix tuning now uses Cache but that won't work with gradient checkpointing
            if any(getattr(module, "gradient_checkpointing", False) for module in self.get_base_model().modules()):
                raise ValueError("Prefix tuning does not work with gradient checkpointing.")
            prompt_encoder = model_cls(config)
        else:
            raise ValueError("Not supported")

        prompt_encoder = prompt_encoder.to(self.device)
        self.prompt_encoder.update(torch.nn.ModuleDict({adapter_name: prompt_encoder}))
        self.prompt_tokens[adapter_name] = torch.arange(
            config.num_virtual_tokens * config.num_transformer_submodules
        ).long()

    def prepare_model_for_gradient_checkpointing(self, model: PreTrainedModel):
        r"""
        Prepares the model for gradient checkpointing if necessary
        """
        self._prepare_model_for_gradient_checkpointing(model)

    def _prepare_model_for_gradient_checkpointing(self, model: PreTrainedModel):
        if not (
            getattr(model, "is_loaded_in_8bit", False)
            or getattr(model, "is_loaded_in_4bit", False)
            or getattr(model, "is_quantized", False)
        ):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            elif hasattr(model, "get_input_embeddings"):

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        return model

    def get_prompt_embedding_to_save(self, adapter_name: str) -> torch.Tensor:
        """
        Returns the prompt embedding to save when saving the model. Only applicable when using a prompt learning
        method.
        """
        prompt_encoder = self.prompt_encoder[adapter_name]
        prompt_tokens = (
            self.prompt_tokens[adapter_name].unsqueeze(0).expand(1, -1).to(prompt_encoder.embedding.weight.device)
        )
        peft_type = self.peft_config[adapter_name].peft_type
        if self.peft_config[adapter_name].peft_type == PeftType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : self.peft_config[adapter_name].num_virtual_tokens]

        if self.peft_config[adapter_name].peft_type == PeftType.MULTITASK_PROMPT_TUNING:
            prompt_embedding_cls = PEFT_TYPE_TO_TUNER_MAPPING[peft_type]
            prompt_embeddings = super(prompt_embedding_cls, prompt_encoder).forward(prompt_tokens)
        else:
            prompt_embeddings = prompt_encoder(prompt_tokens)

        return prompt_embeddings[0].detach().cpu()

    def get_prompt(
        self, batch_size: int, task_ids: Optional[torch.Tensor] = None, max_cache_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Returns the virtual prompts to use for Peft. Only applicable when using a prompt learning method.
        """
        peft_config = self.active_peft_config
        prompt_encoder = self.prompt_encoder[self.active_adapter]
        prompt_tokens = (
            self.prompt_tokens[self.active_adapter]
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(prompt_encoder.embedding.weight.device)
        )
        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : peft_config.num_virtual_tokens]
            if peft_config.inference_mode:
                past_key_values = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                past_key_values = prompt_encoder(prompt_tokens)
            if self.base_model_torch_dtype is not None:
                past_key_values = past_key_values.to(self.base_model_torch_dtype)
            past_key_values = past_key_values.view(
                batch_size,
                peft_config.num_virtual_tokens,
                peft_config.num_layers * 2,
                peft_config.num_attention_heads,
                peft_config.token_dim // peft_config.num_attention_heads,
            )
            if peft_config.num_transformer_submodules == 2:
                past_key_values = torch.cat([past_key_values, past_key_values], dim=2)

            # Transpose: 2 x [num_layers, batch_size, num_heads, num_virtual_tokens, head_dim]
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(
                peft_config.num_transformer_submodules * 2
            )

            base_model = self.get_base_model()
            model_config = getattr(base_model, "config", None)
            model_type = getattr(model_config, "model_type", "")
            if TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING.get(self.config.model_type, None) is not None:
                post_process_fn = TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING[self.config.model_type]
                past_key_values = post_process_fn(past_key_values)
            elif ("gemma2" in model_type) or ("gemma3_text" in model_type):
                # Gemma2 and Gemma3 only support HybridCache (which does not have the from_legacy_cache method)
                if max_cache_len is None:
                    raise ValueError(
                        "max_cache_len is None but it should have been passed. Something went wrong, please open an "
                        "issue on GitHub with a reproducer: https://github.com/huggingface/peft/issues"
                    )
                base_config = base_model.config
                if hasattr(base_config, "get_text_config"):
                    base_config = base_config.get_text_config()
                new_cache = HybridCache(
                    base_config,
                    max_batch_size=batch_size,
                    max_cache_len=max_cache_len,
                    dtype=past_key_values[0].dtype,
                    device=past_key_values[0].device,
                )
                cache_position = torch.arange(peft_config.num_virtual_tokens, device=past_key_values[0].device)
                for layer_idx in range(peft_config.num_layers):
                    key_states, value_states = past_key_values[0][layer_idx], past_key_values[1][layer_idx]
                    new_cache.update(
                        key_states, value_states, layer_idx, cache_kwargs={"cache_position": cache_position}
                    )
                past_key_values = new_cache
            elif peft_config.num_transformer_submodules == 1:
                # Dont' apply this to encoder-decoder models and not to models requiring special processing.
                # local import in case users use a very old transformers version
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            elif peft_config.num_transformer_submodules == 2 and self.base_model._supports_cache_class:
                # Dont' apply this to encoder-decoder models that don't support new Cachc format yet
                # If we don't apply this, prefix-tuning fails to update cross-attn cache
                past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)
                past_key_values.cross_attention_cache = DynamicCache()
                past_key_values.is_updated = {
                    layer_idx: False for layer_idx in range(len(past_key_values.cross_attention_cache.key_cache))
                }
            map_cache_to_layer_device_map(self.get_base_model(), past_key_values)  # no-op if not a Cache instance
            return past_key_values
        else:
            if peft_config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
                prompts = prompt_encoder(prompt_tokens, task_ids)
            else:
                if peft_config.inference_mode:
                    prompts = prompt_encoder.embedding.weight
                else:
                    # Take only one prompt token sample and expand the output instead of expanding the input, see:
                    # https://github.com/huggingface/peft/issues/2043#issuecomment-2321522577
                    prompt_tokens = prompt_tokens[:1]
                    prompts = prompt_encoder(prompt_tokens)
                prompts = prompts.repeat(batch_size, 1, 1)
            return prompts

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        r"""
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                if hasattr(param, "element_size"):
                    num_bytes = param.element_size()
                elif not hasattr(param, "quant_storage"):
                    num_bytes = 1
                else:
                    num_bytes = param.quant_storage.itemsize
                num_params = num_params * 2 * num_bytes

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def print_trainable_parameters(self) -> None:
        """
        Prints the number of trainable parameters in the model.

        Note: print_trainable_parameters() uses get_nb_trainable_parameters() which is different from
        num_parameters(only_trainable=True) from huggingface/transformers. get_nb_trainable_parameters() returns
        (trainable parameters, all parameters) of the Peft Model which includes modified backbone transformer model.
        For techniques like LoRA, the backbone transformer model is modified in place with LoRA modules. However, for
        prompt tuning, the backbone transformer model is unmodified. num_parameters(only_trainable=True) returns number
        of trainable parameters of the backbone transformer model which can be different.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
        )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "base_model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.base_model, name)

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        # If the base model has a method called _enable_peft_forward_hooks, it is invoked as a context. Otherwise, this
        # runs without any changes
        if hasattr(self.base_model, "_enable_peft_forward_hooks"):
            with self.base_model._enable_peft_forward_hooks(*args, **kwargs):
                yield
            return
        else:
            # nothing to enable
            yield
            return

    def forward(self, *args: Any, **kwargs: Any):
        """
        Forward pass of the model.
        """
        with self._enable_peft_forward_hooks(*args, **kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
            return self.get_base_model()(*args, **kwargs)

    def generate(self, *args, **kwargs):
        with self._enable_peft_forward_hooks(*args, **kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
            return self.get_base_model().generate(*args, **kwargs)

    def _get_base_model_class(self, is_prompt_tuning=False):
        """
        Returns the base model class.
        """
        if not is_prompt_tuning:
            return self.base_model.model.__class__
        return self.base_model.__class__

    @contextmanager
    def disable_adapter(self):
        """
        Context manager that disables the adapter module. Use this to run inference on the base model.

        Example:

        ```py
        >>> with model.disable_adapter():
        ...     model(inputs)
        ```
        """
        if self.peft_config[self.active_adapter].is_prompt_learning:
            try:
                # TODO: consider replacing this patching of methods with a more robust mechanism: setting a flag and
                # letting the underlying methods deal with it, same as how LoRA does it.
                old_forward = self.forward
                self.forward = self.base_model.forward
                old_prepare_inputs_for_generation = self.prepare_inputs_for_generation
                self.prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
                yield
            finally:
                self.forward = old_forward
                self.prepare_inputs_for_generation = old_prepare_inputs_for_generation

        elif self.peft_config[self.active_adapter].is_adaption_prompt:
            try:
                self.base_model.disable_adapter_layers()
                yield
            finally:
                self.base_model.enable_adapter_layers()

        else:  # LoRA, LoHa, etc.
            model_status = self.get_model_status()
            if model_status.enabled == "irregular":
                warnings.warn(
                    "The model contains some adapter layers that are enabled and others that are disabled. "
                    "This is most likely unintentional. After exiting the disable_adapter context, all adapters "
                    "will be enabled"
                )
            try:
                self.base_model.disable_adapter_layers()
                yield
            finally:
                if model_status.enabled is not False:
                    # model_status.enabled is `True` or `"irregular"`
                    self.base_model.enable_adapter_layers()

    def get_base_model(self) -> torch.nn.Module:
        """
        Returns the base model.
        """
        return (
            self.base_model
            if (self.active_peft_config.is_prompt_learning or self.peft_type == PeftType.POLY)
            else self.base_model.model
        )

    def add_adapter(self, adapter_name: str, peft_config: PeftConfig, low_cpu_mem_usage: bool = False) -> None:
        """
        Add an adapter to the model based on the passed configuration.

        This adapter is not trained. To load a trained adapter, check out [`PeftModel.load_adapter`].

        The name for the new adapter should be unique.

        The new adapter is not automatically set as the active adapter. Use [`PeftModel.set_adapter`] to set the active
        adapter.

        Args:
            adapter_name (`str`):
                The name of the adapter to be added.
            peft_config ([`PeftConfig`]):
                The configuration of the adapter to be added.
            low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
                Create empty adapter weights on meta device. Useful to speed up the process when loading saved
                adapters. Don't use this option when creating a new PEFT adapter for training.

        """
        prefix = PEFT_TYPE_TO_PREFIX_MAPPING.get(peft_config.peft_type)
        if prefix and adapter_name in prefix:
            warnings.warn(
                f"Adapter name {adapter_name} should not be contained in the prefix {prefix}."
                "This may lead to reinitialization of the adapter weights during loading."
            )

        if peft_config.peft_type != self.peft_type:
            raise ValueError(
                f"Cannot combine adapters with different peft types. "
                f"Found {self.peft_type} and {peft_config.peft_type}."
            )

        try:
            if peft_config.is_prompt_learning:
                self.peft_config[adapter_name] = peft_config
                if hasattr(self.config, "to_dict"):
                    dict_config = self.config.to_dict()
                else:
                    dict_config = self.config

                peft_config = _prepare_prompt_learning_config(peft_config, dict_config)
                self._setup_prompt_encoder(adapter_name)
                set_additional_trainable_modules(
                    model=self.base_model,
                    peft_config=peft_config,
                    model_config=BaseTuner.get_model_config(self),
                    adapter_name=adapter_name,
                )
            elif peft_config.is_adaption_prompt:
                self.base_model.add_adapter(adapter_name, peft_config)
                set_additional_trainable_modules(
                    model=self.base_model,
                    peft_config=peft_config,
                    model_config=BaseTuner.get_model_config(self),
                    adapter_name=adapter_name,
                )
            else:
                self.peft_config[adapter_name] = peft_config
                self.base_model.inject_adapter(
                    self.base_model.model, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage
                )
        except Exception:  # something went wrong, roll back
            if adapter_name in self.peft_config:
                del self.peft_config[adapter_name]
            raise

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in self.peft_config:
            raise ValueError(f"Adapter {adapter_name} does not exist")

        self.base_model.delete_adapter(adapter_name=adapter_name)
        new_active_adapters = self.active_adapters
        num_adapters = len(new_active_adapters)
        # Note: PeftModel assumes that there is exactly one active adapter, so we should theoretically raise if
        # num_adapters != 1. However, we have allowed this in the past (maybe inadvertently), so we let it slip and
        # don't introduce a backwards incompatibility by raising an error.
        if num_adapters == 1:
            self.active_adapter = new_active_adapters[0]

    @property
    def modules_to_save(self) -> Optional[set[str]]:
        modules: set[str] = set()
        for config in self.peft_config.values():
            if getattr(config, "modules_to_save", None) is not None:
                # modules_to_save can only be a sequence of str, not a str
                modules.update(config.modules_to_save)

        if not modules:
            # for backwards compatibility, as modules_to_save was initialized as None
            return None
        return modules

    def get_layer_status(self) -> list[TunerLayerStatus]:
        """Get the status of each adapter layer in the model.

        This method returns a list of `TunerLayerStatus` dataclass instances, each of which contains the following
        attributes:

        - `name` (`str`):
           The name of the adapter layer, e.g. `model.encoder.block.0.layer.0.SelfAttention.q`.
        - `module_type` (`str`):
           The type of the adapter layer, e.g. `lora.Linear`.
        - `enabled` (`bool`):
           Whether the adapter layer is enabled.
        - `active_adapters` (`list[str]`):
           The names of the active adapters, if any, e.g. `["default"]`.
        - `merged_adapters` (`list[str]`):
           The names of the merged adapters, if any, e.g. `["default"]`.
        - `available_adapters` (`list[str]`):
           The names of the available adapters, e.g. `["default"]`.

        Args:
            model ([`~PeftModel`]):
                The model to get the adapter layer status from.

        Returns:
            list[`peft.peft_model.TunerLayerStatus`]:
                A list of dataclasses, each containing the status of the corresponding adapter layer.

        """
        return get_layer_status(self)

    def get_model_status(self) -> TunerModelStatus:
        """Get the status of tuners of the model.

        This method returns a `TunerModelStatus` dataclass instance, which contains the following attributes:

        - `base_model_type` (`str`):
           The type of the base model, e.g. `T5Model`.
        - `adapter_model_type` (`str`):
           The type of the adapter model, e.g. `LoraModel`.
        - `peft_types` (`dict[str, str]`):
           The mapping of adapter name to adapter type, e.g. `{"default": "LORA"}`.
        - `trainable_params` (`int`):
           The number of trainable parameters in the model.
        - `total_params` (`int`):
           The total number of parameters in the model.
        - `num_adapter_layers` (`int`):
           The number of adapter layers in the model.
        - `enabled` (`bool`, `Literal["irregular"]`):
           Whether all adapter layers are enabled. If some are enabled and some are not, this will be `"irregular"`.
           This means that your model is in an inconsistent state and might not work as expected.
        - `active_adapters` (`list[str]`, `Literal["irregular"]`):
           The names of the active adapters. If the active adapters are not consistent across all layers, this will be
           `"irregular"`, which means that your model is in an inconsistent state and might not work as expected.
        - `merged_adapters` (`list[str]`, `Literal["irregular"]`):
           The names of the merged adapters. If the merged adapters are not consistent across all layers, this will be
           `"irregular"`, which means that your model is in an inconsistent state and might not work as expected.
        - `available_adapters` (`list[str]`):
           The names of the available adapters, e.g. `["default"]`.

        Args:
            model ([`~PeftModel`]):
                The model to get the adapter layer status from.

        Returns:
            `peft.peft_model.TunerModelStatus`:
                A dataclass containing the status of the model.

        """
        return get_model_status(self)

    @classmethod
    def _split_kwargs(cls, kwargs: dict[str, Any]):
        _kwargs_not_in_hf_hub_download_signature = ("use_auth_token",)
        hf_hub_download_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters or key in _kwargs_not_in_hf_hub_download_signature:
                hf_hub_download_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return hf_hub_download_kwargs, other_kwargs

    def _update_offload(self, offload_index: dict[str, dict[str, str]], adapters_weights: dict[str, torch.tensor]):
        """
        Update the offload_index and safetensors files for loading and mergine PeftModels with disk-offloaded modules.

        Args:
            offload_index (Dict[str: str]):
                Dictionary of disk-offloaded modules with their metadata and safetensors filenames
            adapters_weights (Dict[str: torch.tensor]):
                Dictionary of Peft adapter module names and weights
        """

        if not offload_index:
            return offload_index

        prefix = "base_model.model."
        # rename offload index weight and model names
        adapter_names = list(self.peft_config.keys())
        for adapter_name in adapter_names:
            keys = list(offload_index.keys())
            block_id = keys[0].split(".")[0] + "."  # for writing safetensors key,

            # replace original offload index keys with PeftModel keys
            for key in keys:
                suffix_pos = key.rfind(".")
                extended_prefix = prefix + key[:suffix_pos]
                module = dict(self.named_modules())[extended_prefix]
                if isinstance(module, BaseTunerLayer):
                    new_key = prefix + key[:suffix_pos] + ".base_layer" + key[suffix_pos:]
                else:
                    new_key = prefix + key
                offload_index[key]["weight_name"] = new_key
                offload_index[new_key] = offload_index[key]
                del offload_index[key]

            files_seen = set()
            # rename safetensors for dispatch
            for new_key in list(offload_index.keys()):
                fname = offload_index[new_key]["safetensors_file"]

                # make a new file name
                new_fname_list = list(fname.split(os.sep))
                for i, name in enumerate(new_fname_list):
                    if "--" in name:
                        new_fname_list[i] += "-peft"
                        break
                new_fname = os.path.join(*new_fname_list)

                if fname in files_seen:
                    continue
                safe_dict = {}
                with safe_open(fname, framework="pt") as f:
                    for safe_key in f.keys():
                        safe_tensor = f.get_tensor(safe_key)
                        metadata = f.metadata()
                        suffix_pos = safe_key.rfind(".")
                        extended_prefix = prefix + block_id + safe_key[:suffix_pos]
                        safe_module = dict(self.named_modules())[extended_prefix]
                        if isinstance(safe_module, BaseTunerLayer):
                            final_key = extended_prefix + ".base_layer" + safe_key[suffix_pos:]
                            lora_dict = {key: val for key, val in adapters_weights.items() if extended_prefix in key}

                            # add LoRA keys and values to disk offload
                            for lora_key, lora_val in lora_dict.items():
                                divide = lora_key.rfind(".")
                                new_key = lora_key[:divide] + f".{adapter_name}" + lora_key[divide:]
                                safe_dict[new_key] = lora_val
                        else:
                            final_key = prefix + block_id + safe_key
                        safe_dict[final_key] = safe_tensor
                    files_seen.add(new_fname)

                    # avoid overwriting original safetensors
                    for key in safe_dict.keys():
                        offload_index[key] = {"safetensors_file": new_fname, "weight_name": key}

                    base_name = os.path.dirname(new_fname)
                    if not os.path.exists(base_name):
                        os.makedirs(base_name)
                    safe_save_file(safe_dict, new_fname, metadata=metadata)

    def _check_new_adapter_config(self, peft_config: PeftConfig, is_trainable: bool) -> None:
        """Perform checks on newly added PEFT configs to ensure integrity."""
        if peft_config.is_prompt_learning and is_trainable:
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")

        # Since PiSSA/CorDA/OLoRA modifies the base weights, it should not be combined with other adapters.
        all_configs = [peft_config] + list(self.peft_config.values())
        if len(all_configs) > 1:
            if any(getattr(config, "init_lora_weights", None) == "pissa" for config in all_configs):
                msg = (
                    "PiSSA changes the base weights of the model and should thus not be used with other adapters. "
                    "Consider converting the PiSSA adapter into a normal LoRA adapter: "
                    "https://github.com/huggingface/peft/tree/main/examples/pissa_finetuning#convert-pissa-to-lora"
                )
                warnings.warn(msg)
            elif any(getattr(config, "init_lora_weights", None) == "corda" for config in all_configs):
                msg = (
                    "CorDA changes the base weights of the model and should thus not be used with other adapters. "
                    "Consider converting the CorDA adapter into a normal LoRA adapter: "
                    "https://github.com/huggingface/peft/tree/main/examples/corda_finetuning#convert-corda-to-lora"
                )
                warnings.warn(msg)
            elif any(getattr(config, "init_lora_weights", None) == "olora" for config in all_configs):
                msg = (
                    "OLoRA changes the base weights of the model and should thus not be used with other adapters. "
                    "Consider converting the OLoRA adapter into a normal LoRA adapter: "
                    "https://github.com/huggingface/peft/tree/main/examples/olora_finetuning#olora-and-lora"
                )
                warnings.warn(msg)

    def load_adapter(
        self,
        model_id: Union[str, os.PathLike],
        adapter_name: str,
        is_trainable: bool = False,
        torch_device: Optional[str] = None,
        autocast_adapter_dtype: bool = True,
        ephemeral_gpu_offload: bool = False,
        low_cpu_mem_usage: bool = False,
        key_mapping: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ):
        """
        Load a trained adapter into the model.

        The name for the new adapter should be unique.

        The new adapter is not automatically set as the active adapter. Use [`PeftModel.set_adapter`] to set the active
        adapter.

        Args:
            model_id (`str` or `os.PathLike`):
                The name of the PEFT configuration to use. Can be either:
                    - A string, the `model id` of a PEFT configuration hosted inside a model repo on the Hugging Face
                      Hub.
                    - A path to a directory containing a PEFT configuration file saved using the `save_pretrained`
                      method (`./my_peft_config_directory/`).
            adapter_name (`str`):
                The name of the adapter to be added.
            is_trainable (`bool`, *optional*, defaults to `False`):
                Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and can only be
                used for inference.
            torch_device (`str`, *optional*, defaults to None):
                The device to load the adapter on. If `None`, the device will be inferred.
            autocast_adapter_dtype (`bool`, *optional*, defaults to `True`):
                Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter
                weights using float16 and bfloat16 to float32, as this is typically required for stable training, and
                only affect select PEFT tuners.
            ephemeral_gpu_offload (`bool`, *optional*, defaults to `False`):
                Whether to use ephemeral GPU offloading for partially loaded modules. Defaults to `False`.
            low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
                Create empty adapter weights on meta device before loading the saved weights. Useful to speed up the
                process.
            key_mapping (dict, *optional*, defaults to None)
                Extra mapping of PEFT `state_dict` keys applied before loading the `state_dict`. When this mapping is
                applied, the PEFT-specific `"base_model.model"` prefix is removed beforehand and the adapter name (e.g.
                `"default"`) is not inserted yet. Only pass this argument if you know what you're doing.
            kwargs: (`optional`):
                Additional arguments to modify the way the adapter is loaded, e.g. the token for Hugging Face Hub.
        """
        from .mapping import PEFT_TYPE_TO_CONFIG_MAPPING

        hf_hub_download_kwargs, kwargs = self._split_kwargs(kwargs)
        if torch_device is None:
            torch_device = infer_device()

        if adapter_name not in self.peft_config:
            # load the config
            peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(
                    model_id,
                    **hf_hub_download_kwargs,
                )
            ].from_pretrained(
                model_id,
                ephemeral_gpu_offload=ephemeral_gpu_offload,
                **hf_hub_download_kwargs,
            )
            self._check_new_adapter_config(peft_config, is_trainable=is_trainable)
            peft_config.inference_mode = not is_trainable
            self.add_adapter(adapter_name, peft_config, low_cpu_mem_usage=low_cpu_mem_usage)

        adapters_weights = load_peft_weights(
            model_id, device=torch_device, key_mapping=key_mapping, **hf_hub_download_kwargs
        )

        # load the weights into the model
        ignore_mismatched_sizes = kwargs.get("ignore_mismatched_sizes", False)
        load_result = set_peft_model_state_dict(
            self,
            adapters_weights,
            adapter_name=adapter_name,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

        tuner = self.peft_config[adapter_name].peft_type
        tuner_prefix = PEFT_TYPE_TO_PREFIX_MAPPING.get(tuner, "")
        adapter_missing_keys = []

        # Filter missing keys specific to the current adapter and tuner prefix.
        for key in load_result.missing_keys:
            if tuner_prefix in key and adapter_name in key:
                adapter_missing_keys.append(key)

        load_result.missing_keys.clear()
        load_result.missing_keys.extend(adapter_missing_keys)

        if (
            (getattr(self, "hf_device_map", None) is not None)
            and (len(set(self.hf_device_map.values()).intersection({"cpu", "disk"})) > 0)
            and len(self.peft_config) == 1
        ):
            device_map = kwargs.get("device_map", "auto")
            max_memory = kwargs.get("max_memory", None)
            offload_folder = kwargs.get("offload_folder", None)
            offload_dir = kwargs.get("offload_dir", None)
            offload_index = kwargs.get("offload_index", None)

            if offload_dir is not None and offload_folder is not None:
                # see https://github.com/huggingface/peft/issues/2541
                raise ValueError("Cannot use `offload_folder` when `offload_dir` is specified.")
            elif offload_dir is None:
                # to keep backwards compatibility
                offload_dir = offload_folder

            dispatch_model_kwargs = {}
            # Safety checker for previous `accelerate` versions
            # `offload_index` was introduced in https://github.com/huggingface/accelerate/pull/873/
            if "offload_index" in inspect.signature(dispatch_model).parameters:
                dispatch_model_kwargs["offload_index"] = offload_index

            no_split_module_classes = self._no_split_modules

            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    self,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == "balanced_low_0"),
                )

            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    self, max_memory=max_memory, no_split_module_classes=no_split_module_classes
                )

            self._update_offload(offload_index, adapters_weights)
            dispatch_model_kwargs["offload_index"] = offload_index

            dispatch_model(
                self,
                device_map=device_map,
                offload_dir=offload_dir,
                **dispatch_model_kwargs,
            )

            hook = AlignDevicesHook(io_same_device=True)
            if self.peft_config[adapter_name].is_prompt_learning:
                remove_hook_from_submodules(self.prompt_encoder)
            add_hook_to_module(self.get_base_model(), hook)

        if hasattr(self.base_model, "_cast_adapter_dtype"):
            self.base_model._cast_adapter_dtype(
                adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype
            )

        # Set model in evaluation mode to deactivate Dropout modules by default
        if not is_trainable:
            self.eval()
        return load_result

    def set_adapter(self, adapter_name: str) -> None:
        """
        Sets the active adapter.

        Only one adapter can be active at a time.

        Additionally, this function will set the specified adapter to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str`):
                The name of the adapter to be set as active. The adapter must be loaded first.
        """
        if adapter_name not in self.peft_config:
            raise ValueError(f"Adapter {adapter_name} not found.")
        self.active_adapter = adapter_name
        if not self.peft_config[adapter_name].is_prompt_learning:
            self.base_model.set_adapter(adapter_name)
        _set_adapter(self, adapter_name)

    @property
    def base_model_torch_dtype(self):
        return getattr(self.base_model, "dtype", None)

    @property
    def active_peft_config(self):
        return self.peft_config[self.active_adapter]

    def _get_peft_specific_model_tags(self):
        """Derive tags for the model card from the adapter's config. For example, setting the
        base model is important for enabling support for HF inference providers but it also makes models more
        searchable on the HF hub.
        """
        peft_method = self.active_peft_config.peft_type.value

        tags = []

        if hasattr(self.base_model, "model") and isinstance(self.base_model.model, transformers.PreTrainedModel):
            tags.append("transformers")

        if peft_method == "LORA":
            tags.append("lora")

        if hasattr(self.base_model, "name_or_path"):
            tags.append(f"base_model:adapter:{self.base_model.name_or_path}")

        return tags

    def create_or_update_model_card(self, output_dir: str):
        """
        Updates or create model card to include information about peft:
        1. Adds `peft` library tag
        2. Adds peft version
        3. Adds base model info
        4. Adds quantization information if it was used
        """

        filename = os.path.join(output_dir, "README.md")

        card = ModelCard.load(filename) if os.path.exists(filename) else ModelCard.from_template(ModelCardData())

        card.data["library_name"] = "peft"

        tags = set()
        base_model = self.get_base_model()
        if hasattr(base_model, "model_tags"):
            tags = tags.union(base_model.model_tags or [])

        tags = tags.union(self._get_peft_specific_model_tags())
        if tags:
            card.data["tags"] = sorted(tags)

        # One of the rare moments where we can select the pipeline tag with certainty, so let's do that.
        # Makes it easier to deploy an adapter with auto inference since the user doesn't have to add any tags.
        if not card.data.pipeline_tag and isinstance(self, PeftModelForCausalLM):
            card.data.pipeline_tag = "text-generation"

        model_config = BaseTuner.get_model_config(self)
        model_config = None if model_config == DUMMY_MODEL_CONFIG else model_config
        if model_config is not None and "_name_or_path" in model_config:
            card.data["base_model"] = model_config["_name_or_path"]

        lines = card.text.splitlines()

        quantization_config = None
        if hasattr(model_config, "quantization_config"):
            quantization_config = self.config.quantization_config.to_dict()
        training_config_text = ""
        quantization_prefix = "The following `bitsandbytes` quantization config was used during training:"
        # Adds quantization information if it was used
        if quantization_config is not None:
            training_config_text += f"\n{quantization_prefix}\n"
            training_config_text += "\n".join([f"- {name}: {value}" for name, value in quantization_config.items()])
            training_config_text += "\n"

        training_procedure_heading = "## Training procedure"
        if quantization_prefix not in lines and bool(training_config_text):
            if training_procedure_heading in lines:
                lines.insert(lines.index(training_procedure_heading) + 2, training_config_text)
            else:
                lines.append(f"{training_procedure_heading}\n{training_config_text}")

        # Adds peft version
        framework_block_heading = "### Framework versions"
        if f"- PEFT {__version__}" not in lines:
            if framework_block_heading in lines:
                lines.insert(lines.index(framework_block_heading) + 2, f"- PEFT {__version__}")
            else:
                lines.append(f"{framework_block_heading}\n\n- PEFT {__version__}")

        card.text = "\n".join(lines)
        card.save(filename)


class PeftModelForSequenceClassification(PeftModel):
    """
    Peft model for sequence classification tasks.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.
        adapter_name (`str`,  *optional*): The name of the adapter, defaults to `"default"`.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter weights
            using float16 and bfloat16 to float32, as this is typically required for stable training, and only affect
            select PEFT tuners.

    **Attributes**:
        - **config** ([`~transformers.PretrainedConfig`]) -- The configuration object of the base model.
        - **cls_layer_name** (`str`) -- The name of the classification layer.

    Example:

        ```py
        >>> from transformers import AutoModelForSequenceClassification
        >>> from peft import PeftModelForSequenceClassification, get_peft_config

        >>> config = {
        ...     "peft_type": "PREFIX_TUNING",
        ...     "task_type": "SEQ_CLS",
        ...     "inference_mode": False,
        ...     "num_virtual_tokens": 20,
        ...     "token_dim": 768,
        ...     "num_transformer_submodules": 1,
        ...     "num_attention_heads": 12,
        ...     "num_layers": 12,
        ...     "encoder_hidden_size": 768,
        ...     "prefix_projection": False,
        ...     "postprocess_past_key_value_function": None,
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
        >>> peft_model = PeftModelForSequenceClassification(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 370178 || all params: 108680450 || trainable%: 0.3406113979101117
        ```
    """

    def __init__(
        self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default", **kwargs
    ) -> None:
        classifier_module_names = ["classifier", "score"]

        if hasattr(peft_config, "modules_to_save"):
            if peft_config.modules_to_save is None:
                peft_config.modules_to_save = classifier_module_names[:]
            else:
                peft_config.modules_to_save.extend(classifier_module_names)

        # The modification of peft_config must happen before the init call as the `modules_to_save` information
        # will be used to guard the target layer matching against matching `modules_to_save` layers. Only the
        # config is relevant for this, the `modules_to_save` attribute can follow later.
        super().__init__(model, peft_config, adapter_name, **kwargs)

        if hasattr(peft_config, "modules_to_save"):
            for name, _ in self.base_model.named_children():
                if any(module_name in name for module_name in self.modules_to_save):
                    self.cls_layer_name = name
                    break

        # to make sure classifier layer is trainable; this may add a new ModulesToSaveWrapper
        _set_trainable(self, adapter_name, module_names=getattr(peft_config, "modules_to_save", None))

    def add_adapter(self, adapter_name: str, peft_config: PeftConfig, low_cpu_mem_usage: bool = False) -> None:
        """
        Add an adapter to the model based on the passed configuration.

        This adapter is not trained. To load a trained adapter, check out [`PeftModel.load_adapter`].

        The name for the new adapter should be unique.

        The new adapter is not automatically set as the active adapter. Use [`PeftModel.set_adapter`] to set the active
        adapter.

        Args:
            adapter_name (`str`):
                The name of the adapter to be added.
            peft_config ([`PeftConfig`]):
                The configuration of the adapter to be added.
            low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
                Create empty adapter weights on meta device. Useful to speed up the process when loading saved
                adapters. Don't use this option when creating a new PEFT adapter for training.

        """
        # ensure that additional adapters also add the classifier layer to modules_to_save
        if hasattr(peft_config, "modules_to_save"):
            classifier_module_names = ["classifier", "score"]
            if peft_config.modules_to_save is None:
                peft_config.modules_to_save = classifier_module_names[:]
            else:
                peft_config.modules_to_save.extend(classifier_module_names)

        return super().add_adapter(adapter_name, peft_config, low_cpu_mem_usage=low_cpu_mem_usage)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning:
            with self._enable_peft_forward_hooks(**kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                if peft_config.peft_type == PeftType.POLY:
                    kwargs["task_ids"] = task_ids
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

        batch_size = _get_batch_size(input_ids, inputs_embeds)
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            return self._prefix_tuning_forward(input_ids=input_ids, **kwargs)
        else:
            if kwargs.get("token_type_ids", None) is not None:
                kwargs["token_type_ids"] = torch.cat(
                    (
                        torch.zeros(batch_size, peft_config.num_virtual_tokens).to(self.word_embeddings.weight.device),
                        kwargs["token_type_ids"],
                    ),
                    dim=1,
                ).long()
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def _prefix_tuning_forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        batch_size = _get_batch_size(input_ids, inputs_embeds)
        past_key_values = self.get_prompt(batch_size)
        fwd_params = list(inspect.signature(self.base_model.forward).parameters.keys())
        kwargs.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
                "past_key_values": past_key_values,
            }
        )
        if "past_key_values" in fwd_params:
            return self.base_model(labels=labels, **kwargs)
        else:
            transformer_backbone_name = self.base_model.get_submodule(self.transformer_backbone_name)
            fwd_params = list(inspect.signature(transformer_backbone_name.forward).parameters.keys())
            if "past_key_values" not in fwd_params:
                raise ValueError("Model does not support past key values which are required for prefix tuning.")
            outputs = transformer_backbone_name(**kwargs)
            pooled_output = outputs[1] if len(outputs) > 1 else outputs[0]
            if "dropout" in [name for name, _ in list(self.base_model.named_children())]:
                pooled_output = self.base_model.dropout(pooled_output)
            logits = self.base_model.get_submodule(self.cls_layer_name)(pooled_output)

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.base_model.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.base_model.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    if self.base_model.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.base_model.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)
            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


class PeftModelForCausalLM(PeftModel):
    """
    Peft model for causal language modeling.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.
        adapter_name (`str`,  *optional*): The name of the adapter, defaults to `"default"`.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter weights
            using float16 and bfloat16 to float32, as this is typically required for stable training, and only affect
            select PEFT tuners.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModelForCausalLM, get_peft_config

        >>> config = {
        ...     "peft_type": "PREFIX_TUNING",
        ...     "task_type": "CAUSAL_LM",
        ...     "inference_mode": False,
        ...     "num_virtual_tokens": 20,
        ...     "token_dim": 1280,
        ...     "num_transformer_submodules": 1,
        ...     "num_attention_heads": 20,
        ...     "num_layers": 36,
        ...     "encoder_hidden_size": 1280,
        ...     "prefix_projection": False,
        ...     "postprocess_past_key_value_function": None,
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2-large")
        >>> peft_model = PeftModelForCausalLM(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 1843200 || all params: 775873280 || trainable%: 0.23756456724479544
        ```
    """

    def __init__(
        self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default", **kwargs
    ) -> None:
        super().__init__(model, peft_config, adapter_name, **kwargs)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning:
            if self.base_model.config.model_type == "mpt":
                if inputs_embeds is not None:
                    raise AssertionError("forward in MPTForCausalLM does not support inputs_embeds")
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

            if peft_config.peft_type == PeftType.POLY:
                kwargs["task_ids"] = task_ids

            with self._enable_peft_forward_hooks(**kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

        batch_size = _get_batch_size(input_ids, inputs_embeds)
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            # overwrite past_kv in kwargs
            # some archs require max_cache_len to re-initialize the cache
            if input_ids is not None:
                max_cache_len = input_ids.shape[1] + peft_config.num_virtual_tokens
            else:
                max_cache_len = inputs_embeds.shape[1] + peft_config.num_virtual_tokens
            kwargs["past_key_values"] = self.get_prompt(batch_size, max_cache_len=max_cache_len)
            return self.base_model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
        elif peft_config.peft_type == PeftType.CPT:
            return self._cpt_forward(input_ids, inputs_embeds, peft_config, task_ids, batch_size, **kwargs)
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # concat prompt labels
            if labels is not None:
                prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
                kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def _cpt_forward(self, input_ids, inputs_embeds, peft_config, task_ids, batch_size, **kwargs):
        # Extract labels from kwargs
        labels = kwargs.pop("labels")
        device = [i.device for i in [input_ids, inputs_embeds, labels] if i is not None][0]
        # Extract input_type_mask from kwargs and move it to the same device as labels
        if "input_type_mask" in kwargs.keys():
            input_type_mask = kwargs.pop("input_type_mask").to(device)
        else:
            if input_ids is None:
                N_tokens = inputs_embeds.shape[1]
            else:
                N_tokens = input_ids.shape[1]
            input_type_mask = torch.ones((batch_size, N_tokens)).to(device) * 4

        cpt_token_ids = peft_config.cpt_token_ids
        cpt_tokens_type_mask = peft_config.cpt_tokens_type_mask

        # Generate embeddings if not provided
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # Get prompt and concatenate with input embeddings
        prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
        prompts = prompts.to(inputs_embeds.dtype)
        inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
        # If labels are provided, generate prefix labels and type mask
        cpt_labels = None
        if labels is not None:
            # Generate prefix labels and concatenate with the input labels
            prefix_labels = torch.Tensor(cpt_token_ids).long().view(1, -1)
            prefix_labels = prefix_labels.repeat(batch_size, 1).to(labels.device)
            cpt_labels = torch.cat((prefix_labels, labels), dim=1)
            # Generate prefix type mask and shift input type mask values to avoid conflicts
            prefix_type_mask = torch.Tensor(cpt_tokens_type_mask).long().view(1, -1)
            prefix_type_mask = prefix_type_mask.repeat(batch_size, 1).to(labels.device)
            adjusted_input_type_mask = input_type_mask
            adjusted_input_type_mask[adjusted_input_type_mask > 0] += prefix_type_mask.max()
            # Concatenate prefix and shifted input type masks
            cpt_type_mask = torch.cat((prefix_type_mask, adjusted_input_type_mask), dim=1)
            # Identify valid label positions and mask invalid ones with -100
            labels_idx = (cpt_type_mask > 0) & (cpt_type_mask % 4 == 0)
            cpt_labels[~labels_idx] = -100
            # Update kwargs with the modified labels

        kwargs["labels"] = cpt_labels
        # Pass the modified inputs to the base model
        base_model_output = self.base_model(inputs_embeds=inputs_embeds, **kwargs)
        if labels is None:
            return base_model_output
        else:
            # Calculate the loss using the custom CPT loss function
            cpt_embedding = PEFT_TYPE_TO_TUNER_MAPPING[peft_config.peft_type]
            base_model_output = cpt_embedding.calculate_loss(
                base_model_output, cpt_labels, cpt_type_mask, self.peft_config["default"]
            )
            return base_model_output

    def generate(self, *args, **kwargs):
        peft_config = self.active_peft_config
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        if hasattr(self.base_model, "model"):
            self.base_model.model.generation_config = self.generation_config
        else:
            self.base_model.generation_config = self.generation_config
        try:
            if not peft_config.is_prompt_learning:
                with self._enable_peft_forward_hooks(*args, **kwargs):
                    kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                    outputs = self.base_model.generate(*args, **kwargs)
            else:
                outputs = self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs

    def prepare_inputs_for_generation(self, *args, task_ids: Optional[torch.Tensor] = None, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)

        # https://github.com/huggingface/transformers/pull/26681/ introduced new cache format
        # for some architectures which requires a special fix for prompt tuning etc.
        # TODO: starting with transformers 4.38, all architectures should support caching.
        uses_transformers_4_38 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.38.0")
        uses_transformers_4_36 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.36.0")
        transformers_new_cache_archs = ["llama", "mistral", "persimmon", "phi"]
        if packaging.version.parse(transformers.__version__) > packaging.version.parse("4.43.3"):
            # https://github.com/huggingface/transformers/pull/31445
            transformers_new_cache_archs.append("bloom")

        uses_cache = uses_transformers_4_38 or (
            uses_transformers_4_36 and self.base_model.config.model_type in transformers_new_cache_archs
        )

        # heuristic to determine if we're in 'prefill stage' (when the KV cache is filled with the values from the
        # initial input)
        is_prefill = (model_kwargs.get("cache_position") is not None) and (model_kwargs["cache_position"][0] == 0)

        if peft_config.peft_type == PeftType.POLY:
            model_kwargs["task_ids"] = task_ids
        if peft_config.is_prompt_learning:
            if uses_cache and (model_kwargs.get("past_key_values", None) is not None):
                # change in the logic of `prepare_inputs_for_generation` makes the below code necessary
                # In prompt learning methods, past key values are longer when compared to the `input_ids`.
                # As such only consider the last input ids in the autogressive generation phase.
                past_key_values = model_kwargs["past_key_values"]
                if isinstance(past_key_values, (tuple, list)):
                    seq_len = past_key_values[0][0].shape[-2]
                else:  # using transformers kv cache
                    seq_len = past_key_values.get_seq_length()
                if seq_len >= model_kwargs["input_ids"].shape[1]:
                    model_kwargs["input_ids"] = model_kwargs["input_ids"][:, -1:]

            if (attention_mask := model_kwargs.get("attention_mask", None)) is not None:
                if isinstance(attention_mask, dict):
                    # see: https://github.com/huggingface/transformers/pull/37866
                    # For now, just deal with the case of a single attention mask
                    if len(attention_mask) != 1:
                        raise ValueError(
                            f"Expected a single attention mask, got {len(attention_mask)} instead, please open an "
                            "issue (https://github.com/huggingface/peft/issues) and report the error."
                        )
                    attention_mask = list(attention_mask.values())[0]

                size = model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
                prefix_attention_mask = torch.ones(size).to(model_kwargs["input_ids"].device)
                if attention_mask.dim() == 4:
                    # Transform the 4d attention mask to 2d, leave it up to the model to deal with it instead of trying
                    # to create a 4d attention mask here.
                    # from [batch_size, heads, input_ids_length, total_sequence_length]
                    # to   [batch_size, total_sequence_length]
                    bs = attention_mask.shape[0]
                    total_seq_len = prefix_attention_mask.shape[1] + attention_mask.shape[2]
                    attention_mask_2d = torch.ones((bs, total_seq_len), dtype=attention_mask.dtype)

                    if is_prefill and (peft_config.peft_type != PeftType.PREFIX_TUNING):
                        # if in prefill stage, for prompt learning methods that are not prefix tuning, new tokens
                        # (embeddings) are inserted, thus set cache_position to correspond to these tokens
                        cache_position_ = torch.arange(total_seq_len, device=model_kwargs["input_ids"].device)
                    else:
                        # prefix tuning acts directly on the cache, no need to upate cache_position
                        cache_position_ = model_kwargs["cache_position"]

                    attention_mask_new = create_attention_mask(
                        self.get_base_model(),
                        model_input=None,
                        attention_mask=attention_mask_2d,
                        past_key_values=model_kwargs.get("past_key_values"),
                        cache_position=cache_position_,
                        batch_size=bs,
                        sequence_length=total_seq_len,
                    )
                    model_kwargs["attention_mask"] = attention_mask_new
                else:
                    # 2d attention mask
                    model_kwargs["attention_mask"] = torch.cat((prefix_attention_mask, attention_mask), dim=1)

            if model_kwargs.get("position_ids", None) is not None:
                warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
                model_kwargs["position_ids"] = None

            if kwargs.get("token_type_ids", None) is not None:
                warnings.warn(
                    "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                )
                kwargs["token_type_ids"] = None

            # no past_key_values or past_key_values empty cache
            requires_prompt_injection = (model_kwargs.get("past_key_values", None) is None) or (
                isinstance(model_kwargs["past_key_values"], transformers.Cache)
                and not model_kwargs["past_key_values"].get_seq_length()
            )

            if requires_prompt_injection and peft_config.peft_type == PeftType.PREFIX_TUNING:
                # some archs require max_cache_len to re-initialize the cache
                max_cache_len = getattr(model_kwargs.get("past_key_values", None), "max_cache_len", None)
                new_past_key_values = self.get_prompt(
                    batch_size=model_kwargs["input_ids"].shape[0],
                    max_cache_len=max_cache_len,
                )
                model_kwargs["past_key_values"] = new_past_key_values
            elif requires_prompt_injection:
                inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0], task_ids=task_ids)
                prompts = prompts.to(inputs_embeds.dtype)
                model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
                model_kwargs["input_ids"] = None

        # if we're in the prefill stage
        if is_prefill and (peft_config.peft_type == PeftType.PREFIX_TUNING):
            # for prefix tuning, the past_key_values have been prefilled
            model_kwargs["cache_position"] += peft_config.num_virtual_tokens
        elif peft_config.peft_type != PeftType.PREFIX_TUNING:  # prefix tuning needs cache_position
            # For transformers>=4.38.0 - for some architectures such as Llama, `cache_position` is passed in the forward
            # pass to keep track of the position ids of the cache. We have to pop that from `model_kwargs` as
            # `cache_position` is properly created by the model, using the passed `inputs_embeds`:
            # https://github.com/huggingface/transformers/blob/593230f0a1150ea9c0477b9d859f25daf73c8c33/src/transformers/models/llama/modeling_llama.py#L956
            _ = model_kwargs.pop("cache_position", None)

        return model_kwargs


class PeftModelForSeq2SeqLM(PeftModel):
    """
    Peft model for sequence-to-sequence language modeling.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.
        adapter_name (`str`,  *optional*): The name of the adapter, defaults to `"default"`.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter weights
            using float16 and bfloat16 to float32, as this is typically required for stable training, and only affect
            select PEFT tuners.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import PeftModelForSeq2SeqLM, get_peft_config

        >>> config = {
        ...     "peft_type": "LORA",
        ...     "task_type": "SEQ_2_SEQ_LM",
        ...     "inference_mode": False,
        ...     "r": 8,
        ...     "target_modules": ["q", "v"],
        ...     "lora_alpha": 32,
        ...     "lora_dropout": 0.1,
        ...     "fan_in_fan_out": False,
        ...     "enable_lora": None,
        ...     "bias": "none",
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> peft_model = PeftModelForSeq2SeqLM(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 884736 || all params: 223843584 || trainable%: 0.3952474242013566
        ```
    """

    def __init__(
        self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default", **kwargs
    ) -> None:
        super().__init__(model, peft_config, adapter_name, **kwargs)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self.base_model_prepare_encoder_decoder_kwargs_for_generation = (
            self.base_model._prepare_encoder_decoder_kwargs_for_generation
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning:
            if peft_config.peft_type == PeftType.POLY:
                kwargs["task_ids"] = task_ids

            with self._enable_peft_forward_hooks(**kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

        batch_size = _get_batch_size(input_ids, inputs_embeds)
        if decoder_attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(
                decoder_attention_mask.device
            )
            if peft_config.peft_type not in [PeftType.PROMPT_TUNING, PeftType.P_TUNING]:
                decoder_attention_mask = torch.cat((prefix_attention_mask, decoder_attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "decoder_attention_mask": decoder_attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            # overwrite past_kv in kwargs
            kwargs["past_key_values"] = self.get_prompt(batch_size)
            return self.base_model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_inputs_embeds=decoder_inputs_embeds,
                **kwargs,
            )
        elif peft_config.peft_type in [PeftType.PROMPT_TUNING, PeftType.P_TUNING]:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)

            if attention_mask is not None:
                # concat prompt attention mask
                prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(
                    attention_mask.device
                )
                kwargs["attention_mask"] = torch.cat((prefix_attention_mask, attention_mask), dim=1)

            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts[:, : peft_config.num_virtual_tokens], inputs_embeds), dim=1)

            return self.base_model(
                inputs_embeds=inputs_embeds,
                decoder_input_ids=decoder_input_ids,
                decoder_inputs_embeds=decoder_inputs_embeds,
                **kwargs,
            )
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            if decoder_inputs_embeds is None and decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
                decoder_inputs_embeds = self.word_embeddings(decoder_input_ids)

            if attention_mask is not None:
                # concat prompt attention mask
                prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(
                    attention_mask.device
                )
                kwargs["attention_mask"] = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            # concat prompt labels
            if labels is not None:
                if peft_config.num_transformer_submodules == 1:
                    kwargs["labels"] = labels
                elif peft_config.num_transformer_submodules == 2:
                    prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
                    kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts[:, : peft_config.num_virtual_tokens], inputs_embeds), dim=1)
            if peft_config.num_transformer_submodules == 1:
                return self.base_model(inputs_embeds=inputs_embeds, **kwargs)
            elif peft_config.num_transformer_submodules == 2:
                decoder_inputs_embeds = torch.cat(
                    (prompts[:, peft_config.num_virtual_tokens :], decoder_inputs_embeds), dim=1
                )
                return self.base_model(
                    inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, **kwargs
                )

    def generate(self, **kwargs):
        peft_config = self.active_peft_config
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
            self._prepare_encoder_decoder_kwargs_for_generation
        )
        try:
            if not peft_config.is_prompt_learning:
                with self._enable_peft_forward_hooks(**kwargs):
                    kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                    outputs = self.base_model.generate(**kwargs)
            else:
                if "input_ids" not in kwargs:
                    raise ValueError("input_ids must be provided for Peft model generation")
                if kwargs.get("position_ids", None) is not None:
                    warnings.warn(
                        "Position ids are not supported for parameter efficient tuning. Ignoring position ids."
                    )
                    kwargs["position_ids"] = None
                if kwargs.get("token_type_ids", None) is not None:
                    warnings.warn(
                        "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                    )
                    kwargs["token_type_ids"] = None

                if peft_config.peft_type == PeftType.PREFIX_TUNING:
                    outputs = self.base_model.generate(**kwargs)
                elif peft_config.peft_type in [
                    PeftType.PROMPT_TUNING,
                    PeftType.P_TUNING,
                    PeftType.MULTITASK_PROMPT_TUNING,
                ]:
                    kwargs = deepcopy(kwargs)

                    if "encoder_outputs" in kwargs:
                        del kwargs["encoder_outputs"]
                        warnings.warn(
                            "`encoder_outputs` should not be passed to `generate` when using prompt tuning. Ignoring it."
                        )

                    input_ids = kwargs.pop("input_ids")
                    inputs_embeds = self.word_embeddings(input_ids)
                    batch_size = inputs_embeds.shape[0]
                    prompts = self.get_prompt(batch_size=batch_size, task_ids=kwargs.pop("task_ids", None))
                    prompts = prompts.to(inputs_embeds.dtype)

                    inputs_embeds = torch.cat((prompts[:, : peft_config.num_virtual_tokens], inputs_embeds), dim=1)
                    kwargs["inputs_embeds"] = inputs_embeds

                    if "attention_mask" in kwargs:
                        prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(
                            kwargs["attention_mask"].device
                        )
                        kwargs["attention_mask"] = torch.cat((prefix_attention_mask, kwargs["attention_mask"]), dim=1)

                    return self.base_model.generate(**kwargs)
                else:
                    raise NotImplementedError
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
            return outputs

    def prepare_inputs_for_generation(self, *args, task_ids: torch.Tensor = None, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if peft_config.peft_type == PeftType.POLY:
            model_kwargs["task_ids"] = task_ids
        elif peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = model_kwargs.get("past_key_values", None)
            cache_position = model_kwargs.get("cache_position", [None])
            # check prefill stage
            is_prefill_stage = (
                # old cache implementation
                (past_key_values is None)
                # new cache implementation
                or (isinstance(past_key_values, Cache) and (cache_position[0] == 0))
            )
            if is_prefill_stage:
                batch_size = model_kwargs["decoder_input_ids"].shape[0]
                new_past_key_values = self.get_prompt(batch_size)
                model_kwargs["past_key_values"] = new_past_key_values

        return model_kwargs


class PeftModelForTokenClassification(PeftModel):
    """
    Peft model for token classification tasks.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.
        adapter_name (`str`,  *optional*): The name of the adapter, defaults to `"default"`.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter weights
            using float16 and bfloat16 to float32, as this is typically required for stable training, and only affect
            select PEFT tuners.

    **Attributes**:
        - **config** ([`~transformers.PretrainedConfig`]) -- The configuration object of the base model.
        - **cls_layer_name** (`str`) -- The name of the classification layer.

    Example:

        ```py
        >>> from transformers import AutoModelForSequenceClassification
        >>> from peft import PeftModelForTokenClassification, get_peft_config

        >>> config = {
        ...     "peft_type": "PREFIX_TUNING",
        ...     "task_type": "TOKEN_CLS",
        ...     "inference_mode": False,
        ...     "num_virtual_tokens": 20,
        ...     "token_dim": 768,
        ...     "num_transformer_submodules": 1,
        ...     "num_attention_heads": 12,
        ...     "num_layers": 12,
        ...     "encoder_hidden_size": 768,
        ...     "prefix_projection": False,
        ...     "postprocess_past_key_value_function": None,
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForTokenClassification.from_pretrained("bert-base-cased")
        >>> peft_model = PeftModelForTokenClassification(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 370178 || all params: 108680450 || trainable%: 0.3406113979101117
        ```
    """

    def __init__(
        self, model: torch.nn.Module, peft_config: PeftConfig = None, adapter_name: str = "default", **kwargs
    ) -> None:
        super().__init__(model, peft_config, adapter_name, **kwargs)

        classifier_module_names = ["classifier", "score"]
        if hasattr(peft_config, "modules_to_save"):
            if peft_config.modules_to_save is None:
                peft_config.modules_to_save = classifier_module_names[:]
            else:
                peft_config.modules_to_save.extend(classifier_module_names)

        for name, _ in self.base_model.named_children():
            if any(module_name in name for module_name in self.modules_to_save):
                self.cls_layer_name = name
                break

        # to make sure classifier layer is trainable; this may add a new ModulesToSaveWrapper
        _set_trainable(self, adapter_name, module_names=getattr(peft_config, "modules_to_save", None))

    def add_adapter(self, adapter_name: str, peft_config: PeftConfig, low_cpu_mem_usage: bool = False) -> None:
        """
        Add an adapter to the model based on the passed configuration.

        This adapter is not trained. To load a trained adapter, check out [`PeftModel.load_adapter`].

        The name for the new adapter should be unique.

        The new adapter is not automatically set as the active adapter. Use [`PeftModel.set_adapter`] to set the active
        adapter.

        Args:
            adapter_name (`str`):
                The name of the adapter to be added.
            peft_config ([`PeftConfig`]):
                The configuration of the adapter to be added.
            low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
                Create empty adapter weights on meta device. Useful to speed up the process when loading saved
                adapters. Don't use this option when creating a new PEFT adapter for training.

        """
        # ensure that additional adapters also add the classifier layer to modules_to_save
        if hasattr(peft_config, "modules_to_save"):
            classifier_module_names = ["classifier", "score"]
            if peft_config.modules_to_save is None:
                peft_config.modules_to_save = classifier_module_names[:]
            else:
                peft_config.modules_to_save.extend(classifier_module_names)

        return super().add_adapter(adapter_name, peft_config, low_cpu_mem_usage=low_cpu_mem_usage)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not peft_config.is_prompt_learning:
            with self._enable_peft_forward_hooks(**kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                if peft_config.peft_type == PeftType.POLY:
                    kwargs["task_ids"] = task_ids
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

        batch_size = _get_batch_size(input_ids, inputs_embeds)
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            return self._prefix_tuning_forward(input_ids=input_ids, **kwargs)
        else:
            if kwargs.get("token_type_ids", None) is not None:
                kwargs["token_type_ids"] = torch.cat(
                    (
                        torch.zeros(batch_size, peft_config.num_virtual_tokens).to(self.word_embeddings.weight.device),
                        kwargs["token_type_ids"],
                    ),
                    dim=1,
                ).long()
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def _prefix_tuning_forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        batch_size = _get_batch_size(input_ids, inputs_embeds)
        past_key_values = self.get_prompt(batch_size)
        fwd_params = list(inspect.signature(self.base_model.forward).parameters.keys())
        kwargs.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
                "past_key_values": past_key_values,
            }
        )
        if "past_key_values" in fwd_params:
            return self.base_model(labels=labels, **kwargs)
        else:
            transformer_backbone_name = self.base_model.get_submodule(self.transformer_backbone_name)
            fwd_params = list(inspect.signature(transformer_backbone_name.forward).parameters.keys())
            if "past_key_values" not in fwd_params:
                raise ValueError("Model does not support past key values which are required for prefix tuning.")
            outputs = transformer_backbone_name(**kwargs)
            sequence_output = outputs[0]
            if "dropout" in [name for name, _ in list(self.base_model.named_children())]:
                sequence_output = self.base_model.dropout(sequence_output)
            logits = self.base_model.get_submodule(self.cls_layer_name)(sequence_output)

            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


class PeftModelForQuestionAnswering(PeftModel):
    """
    Peft model for extractive question answering.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.
        adapter_name (`str`,  *optional*): The name of the adapter, defaults to `"default"`.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter weights
            using float16 and bfloat16 to float32, as this is typically required for stable training, and only affect
            select PEFT tuners.

    **Attributes**:
        - **config** ([`~transformers.PretrainedConfig`]) -- The configuration object of the base model.
        - **cls_layer_name** (`str`) -- The name of the classification layer.

    Example:

        ```py
        >>> from transformers import AutoModelForQuestionAnswering
        >>> from peft import PeftModelForQuestionAnswering, get_peft_config

        >>> config = {
        ...     "peft_type": "LORA",
        ...     "task_type": "QUESTION_ANS",
        ...     "inference_mode": False,
        ...     "r": 16,
        ...     "target_modules": ["query", "value"],
        ...     "lora_alpha": 32,
        ...     "lora_dropout": 0.05,
        ...     "fan_in_fan_out": False,
        ...     "bias": "none",
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForQuestionAnswering.from_pretrained("bert-base-cased")
        >>> peft_model = PeftModelForQuestionAnswering(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 592900 || all params: 108312580 || trainable%: 0.5473971721475013
        ```
    """

    def __init__(
        self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default", **kwargs
    ) -> None:
        super().__init__(model, peft_config, adapter_name, **kwargs)

        qa_module_names = ["qa_outputs"]
        if hasattr(peft_config, "modules_to_save"):
            if peft_config.modules_to_save is None:
                peft_config.modules_to_save = qa_module_names[:]
            else:
                peft_config.modules_to_save.extend(qa_module_names)

        for name, _ in self.base_model.named_children():
            if any(module_name in name for module_name in self.modules_to_save):
                self.cls_layer_name = name
                break

        # to make sure classifier layer is trainable; this may add a new ModulesToSaveWrapper
        _set_trainable(self, adapter_name, module_names=getattr(peft_config, "modules_to_save", None))

    def add_adapter(self, adapter_name: str, peft_config: PeftConfig, low_cpu_mem_usage: bool = False) -> None:
        """
        Add an adapter to the model based on the passed configuration.

        This adapter is not trained. To load a trained adapter, check out [`PeftModel.load_adapter`].

        The name for the new adapter should be unique.

        The new adapter is not automatically set as the active adapter. Use [`PeftModel.set_adapter`] to set the active
        adapter.

        Args:
            adapter_name (`str`):
                The name of the adapter to be added.
            peft_config ([`PeftConfig`]):
                The configuration of the adapter to be added.
            low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
                Create empty adapter weights on meta device. Useful to speed up the process when loading saved
                adapters. Don't use this option when creating a new PEFT adapter for training.

        """
        # ensure that additional adapters also add the classifier layer to modules_to_save
        if hasattr(peft_config, "modules_to_save"):
            qa_module_names = ["qa_outputs"]
            if peft_config.modules_to_save is None:
                peft_config.modules_to_save = qa_module_names[:]
            else:
                peft_config.modules_to_save.extend(qa_module_names)

        return super().add_adapter(adapter_name, peft_config, low_cpu_mem_usage=low_cpu_mem_usage)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not peft_config.is_prompt_learning:
            if peft_config.peft_type == PeftType.POLY:
                kwargs["task_ids"] = task_ids

            with self._enable_peft_forward_hooks(**kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    start_positions=start_positions,
                    end_positions=end_positions,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

        batch_size = _get_batch_size(input_ids, inputs_embeds)
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "start_positions": start_positions,
                "end_positions": end_positions,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            return self._prefix_tuning_forward(input_ids=input_ids, **kwargs)
        else:
            if kwargs.get("token_type_ids", None) is not None:
                kwargs["token_type_ids"] = torch.cat(
                    (
                        torch.zeros(batch_size, peft_config.num_virtual_tokens).to(self.word_embeddings.weight.device),
                        kwargs["token_type_ids"],
                    ),
                    dim=1,
                ).long()
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def _prefix_tuning_forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        batch_size = _get_batch_size(input_ids, inputs_embeds)
        past_key_values = self.get_prompt(batch_size)
        fwd_params = list(inspect.signature(self.base_model.forward).parameters.keys())
        kwargs.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
                "past_key_values": past_key_values,
            }
        )
        if "past_key_values" in fwd_params:
            return self.base_model(start_positions=start_positions, end_positions=end_positions, **kwargs)
        else:
            transformer_backbone_name = self.base_model.get_submodule(self.transformer_backbone_name)
            fwd_params = list(inspect.signature(transformer_backbone_name.forward).parameters.keys())
            if "past_key_values" not in fwd_params:
                raise ValueError("Model does not support past key values which are required for prefix tuning.")
            outputs = transformer_backbone_name(**kwargs)
            sequence_output = outputs[0]
            if "dropout" in [name for name, _ in list(self.base_model.named_children())]:
                sequence_output = self.base_model.dropout(sequence_output)
            logits = self.base_model.get_submodule(self.cls_layer_name)(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()

            total_loss = None
            if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions = start_positions.clamp(0, ignored_index)
                end_positions = end_positions.clamp(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2

            if not return_dict:
                output = (start_logits, end_logits) + outputs[2:]
                return ((total_loss,) + output) if total_loss is not None else output

            return QuestionAnsweringModelOutput(
                loss=total_loss,
                start_logits=start_logits,
                end_logits=end_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


class PeftModelForFeatureExtraction(PeftModel):
    """
    Peft model for extracting features/embeddings from transformer models

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.
        adapter_name (`str`,  *optional*): The name of the adapter, defaults to `"default"`.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter weights
            using float16 and bfloat16 to float32, as this is typically required for stable training, and only affect
            select PEFT tuners.

    **Attributes**:
        - **config** ([`~transformers.PretrainedConfig`]) -- The configuration object of the base model.

    Example:

        ```py
        >>> from transformers import AutoModel
        >>> from peft import PeftModelForFeatureExtraction, get_peft_config

        >>> config = {
        ...     "peft_type": "LORA",
        ...     "task_type": "FEATURE_EXTRACTION",
        ...     "inference_mode": False,
        ...     "r": 16,
        ...     "target_modules": ["query", "value"],
        ...     "lora_alpha": 32,
        ...     "lora_dropout": 0.05,
        ...     "fan_in_fan_out": False,
        ...     "bias": "none",
        ... }
        >>> peft_config = get_peft_config(config)
        >>> model = AutoModel.from_pretrained("bert-base-cased")
        >>> peft_model = PeftModelForFeatureExtraction(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        ```
    """

    def __init__(self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default", **kwargs):
        super().__init__(model, peft_config, adapter_name, **kwargs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning:
            if peft_config.peft_type == PeftType.POLY:
                kwargs["task_ids"] = task_ids

            with self._enable_peft_forward_hooks(**kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

        batch_size = _get_batch_size(input_ids, inputs_embeds)
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            # overwrite past_kv in kwargs
            kwargs["past_key_values"] = self.get_prompt(batch_size)
            return self.base_model(input_ids=input_ids, **kwargs)
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)


@dataclass
class TunerLayerStatus:
    name: str
    module_type: str
    enabled: bool
    active_adapters: list[str]
    merged_adapters: list[str]
    requires_grad: dict[str, bool | Literal["irregular"]]
    available_adapters: list[str]
    devices: dict[str, list[str]]


def get_layer_status(model: torch.nn.Module) -> list[TunerLayerStatus]:
    """Get the status of each adapter layer in the model.

    This function returns a list of `TunerLayerStatus` dataclass instances, each of which contains the following
    attributes:

    - `name` (`str`):
       The name of the adapter layer, e.g. `model.encoder.block.0.layer.0.SelfAttention.q`.
    - `module_type` (`str`):
       The type of the adapter layer, e.g. `lora.Linear`.
    - `enabled` (`bool`):
       Whether the adapter layer is enabled.
    - `active_adapters` (`list[str]`):
       The names of the active adapters, if any, e.g. `["default"]`.
    - `merged_adapters` (`list[str]`):
       The names of the merged adapters, if any, e.g. `["default"]`.
    - requires_grad : dict[str, bool | Literal["irregular"]]
       The requires_grad status of the parameters for each adapter module. Ideally, it should be either `True` or
       `False`. If the requires_grad status is not consistent across all parameters, the value will be set to
       `"irregular"`.
    - `available_adapters` (`list[str]`):
       The names of the available adapters, e.g. `["default"]`.
    - `devices` (`dict[str, list[str]]`):
       The devices where the parameters of the given adapter are stored, e.g. `["cuda"]`.

    Args:
        model ([Union[`~PeftModel`, `~transformers.PreTrainedModel`, `nn.Module`]]):
            The model to get the adapter layer status from.

    Returns:
        list[`peft.peft_model.TunerLayerStatus`]:
            A list of dataclasses, each containing the status of the corresponding adapter layer.

    """
    if isinstance(model, PeftModel):
        base_model = model.base_model
        if not isinstance(base_model, BaseTuner):
            raise TypeError(
                "get_layer_status() got an invalid PeftModel instance; prefix tuning and adaption prompt are not "
                "supported."
            )
    else:
        base_model = model

    layer_status: list[TunerLayerStatus] = []
    for name, module in base_model.named_modules():
        if not isinstance(module, BaseTunerLayer):
            continue

        # determine if all submodules/parameters if this module require grad or not
        mapping_requires_grad_list: dict[str, list[bool]] = collections.defaultdict(list)
        for adapter_module_name in module.adapter_layer_names:
            adapter_module = getattr(module, adapter_module_name)
            if isinstance(adapter_module, torch.nn.ModuleDict):
                for key, submodule in adapter_module.items():
                    for param in submodule.parameters():
                        mapping_requires_grad_list[key].append(param.requires_grad)
            elif isinstance(adapter_module, torch.nn.ParameterDict):
                for key, param in adapter_module.items():
                    mapping_requires_grad_list[key].append(param.requires_grad)
            else:
                # strange, we don't know how to handle this, ignore for now
                pass

        def check_irrgular(vals: list[bool]) -> bool | Literal["irregular"]:
            if all(vals):
                return True
            if not any(vals):
                return False
            return "irregular"

        requires_grad = {key: check_irrgular(vals) for key, vals in mapping_requires_grad_list.items()}

        devices_dd = collections.defaultdict(list)
        for adapter_module_name in module.adapter_layer_names + module.other_param_names:
            adapter_module = getattr(module, adapter_module_name)
            if isinstance(adapter_module, torch.nn.ModuleDict):
                for key, submodule in adapter_module.items():
                    devices_dd[key].extend([param.device.type for param in submodule.parameters()])
            elif isinstance(adapter_module, torch.nn.ParameterDict) or (
                adapter_module.__class__.__name__ == "BufferDict"
            ):  # VeRA
                for key, param in adapter_module.items():
                    devices_dd[key].append(param.device.type)
        devices = {key: sorted(set(val)) for key, val in devices_dd.items()}

        status = TunerLayerStatus(
            name=name,
            module_type=repr(module).partition("(")[0],
            enabled=not module.disable_adapters,
            active_adapters=module.active_adapters,
            merged_adapters=module.merged_adapters,
            requires_grad=requires_grad,
            available_adapters=sorted(module._get_available_adapters()),
            devices=devices,
        )
        layer_status.append(status)

    if not layer_status:
        raise ValueError(
            "No adapter layers found in the model, please ensure that it's a PEFT model or that you have PEFT adapters "
            "injected in the model."
        )

    return layer_status


@dataclass
class TunerModelStatus:
    base_model_type: str
    adapter_model_type: str
    peft_types: dict[str, str]
    trainable_params: int
    total_params: int
    num_adapter_layers: int
    enabled: bool | Literal["irregular"]
    active_adapters: list[str] | Literal["irregular"]
    merged_adapters: list[str] | Literal["irregular"]
    requires_grad: dict[str, bool | Literal["irregular"]]
    available_adapters: list[str]
    devices: dict[str, list[str]]


def get_model_status(model: torch.nn.Module) -> TunerModelStatus:
    """Get the status of tuners of the model.

    This function returns a `TunerModelStatus` dataclass instance, which contains the following attributes:

    - `base_model_type` (`str`):
       The type of the base model, e.g. `T5Model`.
    - `adapter_model_type` (`str`):
       The type of the adapter model, e.g. `LoraModel`.
    - `peft_types` (`dict[str, str]`):
       The mapping of adapter name to adapter type, e.g. `{"default": "LORA"}`.
    - `trainable_params` (`int`):
       The number of trainable parameters in the model.
    - `total_params` (`int`):
       The total number of parameters in the model.
    - `num_adapter_layers` (`int`):
       The number of adapter layers in the model.
    - `enabled` (`bool`, `Literal["irregular"]`):
       Whether all adapter layers are enabled. If some are enabled and some are not, this will be `"irregular"`. This
       means that your model is in an inconsistent state and might not work as expected.
    - `active_adapters` (`list[str]`, `Literal["irregular"]`):
       The names of the active adapters. If the active adapters are not consistent across all layers, this will be
       `"irregular"`, which means that your model is in an inconsistent state and might not work as expected.
    - `merged_adapters` (`list[str]`, `Literal["irregular"]`):
       The names of the merged adapters. If the merged adapters are not consistent across all layers, this will be
       `"irregular"`, which means that your model is in an inconsistent state and might not work as expected.
    - `requires_grad` (`dict[str, bool | Literal["irregular"]]`):
       Whether for the given adapter, all adapter layers have `requires_grad` set to `True` or `False`. If there is a
       mix, this will be set to `"irregular"`, which means that your model is in an inconsistent state and might not
       work as expected.
    - `available_adapters` (`list[str]`):
       The names of the available adapters, e.g. `["default"]`.
    - `devices` (`dict[str, list[str]]`):
       The devices where the parameters of the given adapter are stored, e.g. `["cuda"]`.

    Args:
        model ([Union[`~PeftModel`, `~transformers.PreTrainedModel`, `nn.Module`]]):
            The model to get the adapter layer status from.

    Returns:
        `peft.peft_model.TunerModelStatus`:
            A dataclass containing the status of the model.

    """
    if isinstance(model, PeftModel):
        if not isinstance(model.base_model, BaseTuner):
            raise TypeError(
                "get_model_status() got an invalid PeftModel instance; prefix tuning and adaption prompt are not "
                "supported."
            )
        base_model_type = model.get_base_model().__class__.__name__
        trainable_params, total_params = model.get_nb_trainable_parameters()
        base_model = model.base_model
        peft_types = {key: str(config.peft_type).partition(".")[-1] for key, config in base_model.peft_config.items()}
        adapter_model_type = base_model.__class__.__name__
    elif isinstance(model, PreTrainedModel):
        base_model_type = model.__class__.__name__
        trainable_params, total_params = PeftModel.get_nb_trainable_parameters(model)
        base_model = model
        peft_types = {}
        adapter_model_type = "None"
    else:
        base_model_type = "other"
        trainable_params, total_params = PeftModel.get_nb_trainable_parameters(model)
        base_model = model
        peft_types = {}
        adapter_model_type = "None"

    layer_status = get_layer_status(model)
    num_adapter_layers = len(layer_status)

    enabled_set: set[bool] = {status.enabled for status in layer_status}  # must be {True}, {False}, or {True, False}
    enabled: bool | Literal["irregular"]
    if len(enabled_set) == 1:
        enabled = enabled_set.pop()
    else:
        enabled = "irregular"

    available_adapters: list[str] = sorted(set().union(*(status.available_adapters for status in layer_status)))

    # ideally, active adapters should be consistent across all layers of the model, but we cannot guarantee it
    all_active_adapters: set[tuple[str, ...]] = {tuple(status.active_adapters) for status in layer_status}
    active_adapters: list[str] | Literal["irregular"]
    if not all_active_adapters:
        active_adapters = []
    elif len(all_active_adapters) == 1:
        active_adapters = list(all_active_adapters.pop())
    else:
        active_adapters = "irregular"

    # Here we determine what adapters are merged. This is not trivial because multiple adapters can be merged or not at
    # the same time. Some layers may only have adapter A, some only adapter B, so it's not as easy as just checking
    # which adapters are merged on each layer.

    # First, determine all adapters that are merged on at least on module.
    merged_all: set[str] = set()
    for status in layer_status:
        merged_all.update(status.merged_adapters)

    # Next, check if on any layer, on of these adapters is not merged.
    merged_adapters: list[str] | Literal["irregular"] = sorted(merged_all)
    for status in layer_status:
        unmerged = set(status.available_adapters) - set(status.merged_adapters)
        if unmerged & merged_all:
            # there is overlap between unmerged adapters and adapters that should be merged
            merged_adapters = "irregular"
            break

    # check status of requires_grad
    # first, merge the values for all layers
    requires_grad_all: dict[str, list[bool | Literal["irregular"]]] = collections.defaultdict(list)
    for status in layer_status:
        for key, val in status.requires_grad.items():
            requires_grad_all[key].append(val)

    # then, check if the values are consistent
    def check_irrgular(vals: list[bool | Literal["irregular"]]) -> bool | Literal["irregular"]:
        if all(val is True for val in vals):
            return True
        if all(val is False for val in vals):
            return False
        return "irregular"

    requires_grad = {key: check_irrgular(vals) for key, vals in requires_grad_all.items()}

    devices_dd = collections.defaultdict(list)
    for status in layer_status:
        for key, val in status.devices.items():
            devices_dd[key].extend(val)
    devices = {key: sorted(set(val)) for key, val in devices_dd.items()}

    adapter_model_status = TunerModelStatus(
        base_model_type=base_model_type,
        adapter_model_type=adapter_model_type,
        peft_types=peft_types,
        trainable_params=trainable_params,
        total_params=total_params,
        num_adapter_layers=num_adapter_layers,
        enabled=enabled,
        active_adapters=active_adapters,
        merged_adapters=merged_adapters,
        requires_grad=requires_grad,
        available_adapters=available_adapters,
        devices=devices,
    )
    return adapter_model_status


def __getattr__(name):
    if name == "PEFT_TYPE_TO_MODEL_MAPPING":
        # This is for backwards compatibility: In #2282, PEFT_TYPE_TO_MODEL_MAPPING was removed as it was redundant with
        # PEFT_TYPE_TO_TUNER_MAPPING. However, third party code could still use this mapping, e.g.:
        # https://github.com/AutoGPTQ/AutoGPTQ/blob/6689349625de973b9ee3016c28c11f32acf7f02c/auto_gptq/utils/peft_utils.py#L8
        # TODO: Remove after 2026-01
        msg = (
            "PEFT_TYPE_TO_MODEL_MAPPING is deprecated, please use `from peft import PEFT_TYPE_TO_TUNER_MAPPING` instead. "
            "The deprecated variable will be removed in 2026."
        )
        warnings.warn(msg, category=DeprecationWarning)
        return PEFT_TYPE_TO_TUNER_MAPPING

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
