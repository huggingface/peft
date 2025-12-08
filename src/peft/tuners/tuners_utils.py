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
import dataclasses
import os
import re
import textwrap
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from typing import Any, Optional, Union, overload

import torch
from accelerate.hooks import AlignDevicesHook
from accelerate.utils import named_module_tensors, offload_state_dict
from packaging import version
from torch import nn
from tqdm import tqdm
from transformers import PreTrainedModel
from transformers.pytorch_utils import Conv1D

from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING
from peft.utils import INCLUDE_LINEAR_LAYERS_SHORTHAND
from peft.utils.constants import (
    DUMMY_MODEL_CONFIG,
    DUMMY_TARGET_MODULES,
    EMBEDDING_LAYER_NAMES,
    MIN_TARGET_MODULES_FOR_OPTIMIZATION,
    SEQ_CLS_HEAD_NAMES,
)
from peft.utils.integrations import init_empty_weights
from peft.utils.other import (
    AuxiliaryTrainingWrapper,
    _get_module_names_tied_with_embedding,
    _set_adapter,
    match_target_against_key,
    set_additional_trainable_modules,
)
from peft.utils.peft_types import PeftType, TaskType
from peft.utils.warning import PeftWarning

from ..config import PeftConfig
from ..utils import _get_submodules
from ._buffer_dict import BufferDict


@contextmanager
def onload_layer(layer):
    r"""
    A utility for modifying a module containing one or more tuners and a base layer, any of which are offloaded to the
    CPU or disk. Moves a module's sub-modules to the execution device before some action is performed, after that the
    base layer state dictionary is re-assigned (if that layer was offloaded to the disk) and finally the parameters are
    offloaded.

    If the module has no offloaded sub-modules, this function does nothing.

    Args:
        layer ('torch.nn.Module'):
            layer with tuners to be merged
    """

    offloaded_modules = []
    for name, module in layer.named_modules():
        if name in ["", "base_layer"]:
            continue
        if hasattr(module, "_hf_hook") and isinstance(module._hf_hook, AlignDevicesHook) and module._hf_hook.offload:
            module._hf_hook.pre_forward(module)
            offloaded_modules.append(module)

    base_layer_offload = False
    if hasattr(layer, "base_layer") and (
        hasattr(layer.base_layer, "_hf_hook")
        and isinstance(layer.base_layer._hf_hook, AlignDevicesHook)
        and layer.base_layer._hf_hook.offload
    ):
        # check if the base layer is disk-offloaded (must contain a 'dataset' and an offload index)
        if torch.device("meta") in layer.base_layer._hf_hook.original_devices.values() and hasattr(
            layer.base_layer._hf_hook.weights_map, "dataset"
        ):
            # find the disk-offload index (maps modules to safetensors) from the `dataset` (OffloadedWeightsLoader object)
            index = layer.base_layer._hf_hook.weights_map.dataset.index
            module_name = list(dict(layer.base_layer._hf_hook.weights_map.dataset).keys())[0]  # any module will do
            file_name = index[module_name]["safetensors_file"]
            base_name_arr = []
            # get effective dir name
            for i in os.path.split(file_name):
                if "--" in i:
                    base_name_arr.append(i)
                    break
                base_name_arr.append(i)
            base_name = os.path.join(*base_name_arr)
            safetensors_filename = base_name + "-merged"
        layer.base_layer._hf_hook.pre_forward(layer.base_layer)
        base_layer_offload = True

    yield

    for module in offloaded_modules:
        module._hf_hook.post_forward(module, torch.tensor([]))

    if base_layer_offload:
        # re-make weights map (must be on cpu to send params to the disk via memmap if disk offload)
        layer.base_layer._hf_hook.weights_map = {
            name: param.to("cpu") for name, param in named_module_tensors(layer.base_layer)
        }
        # offload weights map to disk if original device is the disk
        if torch.device("meta") in layer.base_layer._hf_hook.original_devices.values() and hasattr(
            layer.base_layer._hf_hook.weights_map, "dataset"
        ):
            # rewrite directory with merged weights
            offload_state_dict(safetensors_filename, layer.base_layer._hf_hook.weights_map)
        layer.base_layer._hf_hook.post_forward(layer.base_layer, torch.tensor([]))


def _check_lora_target_modules_mamba(peft_config: PeftConfig, model: nn.Module, target_name: str):
    """
    Prevent applying LoRA to incompatible modules in specific architectures (e.g., Mamba).
    """

    lora_like_types = {"LORA", "ADALORA", "XLORA", "RANDLORA"}
    incompatible_modules = {"out_proj", "conv1d"}
    mamba_model_types = {"falcon_h1", "mamba", "mamba2", "falcon_mamba"}

    if (
        peft_config.peft_type in lora_like_types
        and hasattr(model, "config")
        and getattr(model.config, "model_type", None) in mamba_model_types
    ):
        if target_name in incompatible_modules:
            raise ValueError(
                f"[PEFT:{peft_config.peft_type}] Module '{target_name}' is incompatible with Mamba-based models "
                f"(model_type='{model.config.model_type}'). Incompatible modules: {incompatible_modules}. "
                "Please remove it from `target_modules` to avoid compatibility issues."
            )


def _get_in_out_features(module: nn.Module) -> tuple[int, int] | tuple[None, None]:
    """
    Get the in_features and out_features of the layer.

    Returns in_features and out_features as a tuple. If they cannot be determined, return a tuple of None and None.
    This function covers a broad range of layers, some of which the caller might not support. Therefore, just because
    this function returns a valid result does not imply that the layer type is supported.
    """
    if isinstance(module, nn.Linear):
        torch_supports_dtensor = version.parse(torch.__version__) >= version.parse("2.5.0")
        if torch_supports_dtensor and isinstance(module.weight, torch.distributed.tensor.DTensor):
            # If Tensor Parallel is used, the weight is sharded, so we need to get the local shape
            out_features, in_features = module.weight.to_local().shape
        else:
            in_features, out_features = module.in_features, module.out_features
    elif isinstance(module, nn.Conv1d):
        in_features, out_features = module.in_channels, module.out_channels
    elif isinstance(module, nn.Conv2d):
        in_features, out_features = module.in_channels, module.out_channels
    elif isinstance(module, nn.Conv3d):
        in_features, out_features = module.in_channels, module.out_channels
    elif isinstance(module, nn.Embedding):
        in_features, out_features = module.num_embeddings, module.embedding_dim
    elif isinstance(module, Conv1D):
        in_features, out_features = (
            module.weight.ds_shape if hasattr(module.weight, "ds_shape") else module.weight.shape
        )
    elif isinstance(module, nn.MultiheadAttention):
        if not module._qkv_same_embed_dim:
            raise ValueError("Only same dim for query/key/value is supported as of now for MultiheadAttention.")
        in_features, out_features = module.embed_dim, 3 * module.embed_dim
    elif hasattr(module, "infeatures") and hasattr(module, "outfeatures"):
        # QuantLinear
        in_features, out_features = module.infeatures, module.outfeatures
    elif hasattr(module, "input_size") and hasattr(module, "output_size"):
        # Megatron ColumnParallelLinear,RowParallelLinear
        in_features, out_features = module.input_size, module.output_size
    elif hasattr(module, "codebooks") and module.__class__.__name__ == "QuantizedLinear":
        # AQLM QuantLinear
        in_features, out_features = module.in_features, module.out_features
    elif hasattr(module, "w_bit") and module.__class__.__name__ == "WQLinear_GEMM":
        # Awq layers
        in_features, out_features = module.in_features, module.out_features
    elif module.__class__.__name__ == "EetqLinear":
        # Eetq layers
        in_features, out_features = module.in_features, module.out_features
    elif hasattr(module, "W_q") and module.__class__.__name__ == "HQQLinear":
        # HQQ layers
        in_features, out_features = module.in_features, module.out_features
    elif module.__class__.__name__ == "PatchedLinear":
        # INC layers
        in_features, out_features = module.in_features, module.out_features
    else:
        # possibly support user provided custom layer types using dynamic dispatch
        if hasattr(module, "in_features") and hasattr(module, "out_features"):
            in_features, out_features = module.in_features, module.out_features
        else:
            in_features, out_features = None, None
        warnings.warn(f"Unsupported layer type '{type(module)}' encountered, proceed at your own risk.", UserWarning)
    return in_features, out_features


class BaseTuner(nn.Module, ABC):
    r"""
    A base tuner model that provides the common methods and attributes for all tuners that are injectable into a
    torch.nn.Module

    For adding a new Tuner class, one needs to overwrite the following methods:

    - **_prepare_adapter_config**:
        A private method to eventually prepare the adapter config, for example in case the field `target_modules` is
        missing.
    - **_create_and_replace**:
        A private method to create and replace the target module with the adapter module.
    - **_check_target_module_exists**:
        A private helper method to check if the passed module's key name matches any of the target modules in the
        adapter_config.

    The easiest is to check what is done in the `peft.tuners.lora.LoraModel` class.

    Attributes:
        model (`torch.nn.Module`):
            The model to which the adapter tuner layers will be attached.
        forward (`Callable`):
            The forward method of the model.
        peft_config (`Union[`PeftConfig`, dict[str, PeftConfig]]`):
            The adapter configuration object, it should be a dictionary of `str` to `PeftConfig` objects. One can also
            pass a PeftConfig object and a new adapter will be created with the default name `adapter` or create a new
            dictionary with a key `adapter_name` and a value of that peft config.
        config (`dict[str, Any]`):
            The model configuration object, it should be a dictionary of `str` to `Any` objects.
        targeted_module_names (`list[str]`):
            The list of module names that were actually adapted. Can be useful to inspect if you want to quickly
            double-check that the `config.target_modules` were specified correctly.
        targeted_parameter_names (`list[str]`):
            The list of parameter names that were actually adapted. Can be useful to inspect if you want to quickly
            double-check that the `config.target_parameters` were specified correctly.
        prefix (`str`)
            The PEFT-method specific unique prefix. E.g. `"lora_"` for LoRA.
    """

    # Required attributes for child classes:

    # The unique prefix for this PEFT method, e.g. 'lora_' for LoRA.
    prefix: str
    # The class of the tuner layer, e.g. `LoraLayer` for LoRA.
    tuner_layer_cls: type[BaseTunerLayer]
    # The default target modules for various transformers model architectures, like Llama. This is useful to allow users
    # to skip specifying the `target_modules` in the config of the PEFT method. The default is often something like
    # `{'llama': ['q_proj', 'v_proj'], ...}`.
    target_module_mapping: dict[str, list[str]]

    def __init__(
        self,
        model,
        peft_config: Union[PeftConfig, dict[str, PeftConfig]],
        adapter_name: str,
        low_cpu_mem_usage: bool = False,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        super().__init__()

        self.model = model
        self.targeted_module_names: list[str] = []
        self.targeted_parameter_names: list[str] = []

        # For advanced developers, if you want to attach multiple adapters to your
        # model, just add a `peft_config` dict attribute to your model.
        if not hasattr(self, "peft_config"):
            self.peft_config = {adapter_name: peft_config} if isinstance(peft_config, PeftConfig) else peft_config
        else:
            warnings.warn(
                "Already found a `peft_config` attribute in the model. This will lead to having multiple adapters"
                " in the model. Make sure to know what you are doing!"
            )
            if isinstance(peft_config, PeftConfig):
                self.peft_config[adapter_name] = peft_config
            else:
                # user is adding a dict of PeftConfigs
                self.peft_config.update(peft_config)

        self.active_adapter: str | list[str] = adapter_name
        self._pre_injection_hook(self.model, self.peft_config[adapter_name], adapter_name)
        if peft_config != PeftType.XLORA or peft_config[adapter_name] != PeftType.XLORA:
            self.inject_adapter(self.model, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage, state_dict=state_dict)

        # Copy the peft_config in the injected model.
        self.model.peft_config = self.peft_config

    @property
    def active_adapters(self) -> list[str]:
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        # is already a list of str
        return self.active_adapter

    def forward(self, *args: Any, **kwargs: Any):
        return self.model.forward(*args, **kwargs)

    def _pre_injection_hook(self, model: nn.Module, config: PeftConfig, adapter_name: str) -> None:
        r"""
        A hook to be called before the adapter is injected into the model. This method can be overridden by child
        classes to perform any pre-injection operations.

        Args:
            model (`nn.Module`):
                The model to be adapted.
            config (`PeftConfig`):
                The adapter config.
            adapter_name (`str`):
                The adapter name.
        """
        pass

    def _prepare_adapter_config(self, peft_config: PeftConfig, model_config: dict) -> PeftConfig:
        r"""
        A private method to prepare the adapter config.

        For transformers based models, if `peft_config.target_modules` is None, for some model architectures, we can
        automatically infer the target modules from the `TRANSFORMERS_MODELS_TO_XXX_TARGET_MODULES_MAPPING`.

        Args:
            peft_config (`PeftConfig`):
                The adapter config.
            model_config (`dict`):
                The transformers model config, that config should contain the `model_type` key.

        Returns:
            peft_config (`PeftConfig`):
                The PEFT config with updated `target_modules`.

        Raises:
            ValueError:
                Raises an error if the model type was not recognized.
        """
        if peft_config.target_modules is None:
            target_modules = self.target_module_mapping.get(model_config["model_type"])
            if target_modules is None:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(target_modules)
        return peft_config

    def _prepare_model(self, peft_config: PeftConfig, model: nn.Module):
        r"""
        A private method to modify the model structure before adapter is applied.

        See `peft.tuner.lora.LoraModel._prepare_model` for an example.

        Args:
            peft_config (`PeftConfig`):
                The prepared adapter config.
            model (`nn.Module`):
                The model that is going to be adapted.
        """
        pass

    @staticmethod
    def _check_target_module_exists(peft_config: PeftConfig, key: str) -> bool | re.Match[str] | None:
        """
        A helper method to check if the passed module's key name matches any of the target modules in the
        adapter_config.

        Args:
            config (`PeftConfig`):
                A config to match target modules from.
            key (`str`):
                A key to search any matches in config.

        Returns:
            `bool` | `re.Match[str]` | `None`:
                True or re.Match object if key matches any target modules from config, False or None if no match found.
        """
        return check_target_module_exists(peft_config, key)

    @abstractmethod
    def _create_and_replace(
        self,
        peft_config: PeftConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        parameter_name: Optional[str] = None,
    ) -> None:
        r"""
        Inplace replacement of the target module with the adapter layer. This method needs to be overridden by all the
        tuner classes.

        Check `peft.tuners.lora.LoraModel._create_and_replace` for an example.

        Args:
            peft_config (`PeftConfig`):
                The adapter config.
            adapter_name (`str`):
                The adapter name.
            target (`nn.Module`):
                The target module.
            target_name (`str`):
                The target module's name.
            parent (`nn.Module`):
                The parent module.
            current_key (`str`):
                The key of the current target being adapted.
            parameter_name (`str`, *optional*)
                If, and only if, an `nn.Parameter` is being targeted, this is the name of the parameter.
        """
        ...

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        """
        A helper method to mark only the adapter layers as trainable (i.e. module.requires_grad = False).
        """
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = getattr(self.peft_config[active_adapter], "bias", "none")
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias.endswith("_only"):  # e.g. "lora_only" or "boft_only"
                for m in model.modules():
                    if isinstance(m, self.tuner_layer_cls) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    def _enable_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, AuxiliaryTrainingWrapper)):
                module.enable_adapters(enabled)

    def disable_adapter_layers(self) -> None:
        """
        Disable all adapters in-place.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
        # TODO: deprecate in favor of enable_adapters
        for active_adapter in self.active_adapters:
            bias_val = getattr(self.peft_config[active_adapter], "bias", "none")
            if bias_val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{bias_val}' does not produce the "
                    "same output as the base model would without adaption."
                )
                warnings.warn(msg)
        self._enable_adapter_layers(enabled=False)

    def enable_adapter_layers(self) -> None:
        """
        Enable all adapters in-place
        """
        # TODO: deprecate in favor of enable_adapters
        self._enable_adapter_layers(enabled=True)

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        new_adapter = delete_adapter(
            model=self.model, adapter_name=adapter_name, prefix=self.prefix, layer_cls=self.tuner_layer_cls
        )
        self.active_adapter = new_adapter or []

    def set_requires_grad(self, adapter_names: str | Sequence[str], requires_grad: bool = True) -> None:
        """
        Enable or disable gradients on the given adapter(s).

        Args:
            adapter_name (`str` or `Sequence[str]`):
                The name of the adapter(s) whose gradients should be enabled/disabled.
            requires_grad (`bool`, *optional*)
                Whether to enable (`True`, default) or disable (`False`).
        """
        set_requires_grad(self.model, adapter_names=adapter_names, requires_grad=requires_grad)

    def _check_new_adapter_config(self, config: PeftConfig) -> None:
        """
        A helper method to check the config of a new adapter being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        if len(self.peft_config) <= 1:
            return

        # It is assumed that the config was added to self.peft_config *before* calling this check. We should thus never
        # encounter the error below. Still, it is better to verify this, or else subsequent checks could be incorrect.
        if not any(conf is config for conf in self.peft_config.values()):
            raise ValueError(
                "_check_new_peft_config was called incorrectly, this should not happen. Please open an issue and "
                "report the error: https://github.com/huggingface/peft/issues"
            )

        bias_values = [getattr(conf, "bias", "none") for conf in self.peft_config.values()]
        if sum(bias_value != "none" for bias_value in bias_values) > 1:
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

    def _cast_adapter_dtype(self, adapter_name: str, autocast_adapter_dtype: bool = True) -> None:
        """
        A helper method to cast the adapter weights to the correct dtype.

        Currently, this only upcasts float16 and bfloat16 to float32.

        Args:
            adapter_name (`str`):
                The adapter name.
            autocast_adapter_dtype (`bool`, *optional*):
                Whether to autocast the adapter dtype. Defaults to `True`.

        """
        cast_adapter_dtype(self.model, adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype)

    def _check_merge_allowed(self):
        """Helper method to check whether the adapter can be merged.

        Raise a ValueError if it is not possible to merge the adapter with the given configuration.
        """
        example_code = textwrap.dedent(
            """
            ```python
            from transformers import AutoModelForCausalLM

            # Load original tied model
            model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", tie_word_embeddings=False)

            # Set the randomly initialized lm_head to the previously tied embeddings
            model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

            # Save the untied model
            untied_model_dir = "dir/for/untied/model"
            model.save_pretrained(untied_model_dir)
            model.config.save_pretrained(untied_model_dir)

            # Now use the original model but in untied format
            model = AutoModelForCausalLM.from_pretrained(untied_model_dir)
            ```
            """
        )
        tied_target_modules = self._get_tied_target_modules(self.model)
        if tied_target_modules:
            warnings.warn(
                f"Model with `tie_word_embeddings=True` and the {tied_target_modules=} are part of the adapter. "
                "This can lead to complications. "
                "You can opt to merge the adapter after cloning the weights (to untie the embeddings). "
                "You can untie the embeddings by loading the model with `tie_word_embeddings=False`. For example:"
                + example_code
            )

    def _unload_and_optionally_merge(
        self,
        merge: bool = True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ) -> None:
        if merge:
            self._check_merge_allowed()

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "unload_and_optionally_merge_module"):
                    # if layers have special unloading method, like MultiheadAttention, use that
                    unloaded_module = target.unload_and_optionally_merge_module(
                        merge=merge, safe_merge=safe_merge, adapter_names=adapter_names
                    )
                    self._replace_module(parent, target_name, unloaded_module, target)
                elif hasattr(target, "base_layer"):
                    if merge:
                        target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    self._replace_module(parent, target_name, target.get_base_layer(), target)

        return self.model

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        r"""
        This method merges the adapter layers into the base model.

        This is needed if someone wants to use the base model as a standalone model. The returned model has the same
        architecture as the original base model.

        It is important to assign the returned model to a variable and use it, this is not an in-place operation!

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process (default: False).
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.

        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> model_id = ...
        >>> base_model = AutoModelForCausalLM.from_pretrained(model_id)
        >>> peft_model_id = ...
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self) -> torch.nn.Module:
        """
        Return the base model by removing all the PEFT modules.

        It is important to assign the returned model to a variable and use it, this is not an in-place operation!
        """
        return self._unload_and_optionally_merge(merge=False)

    def _check_target_module_compatiblity(self, peft_config: PeftConfig, model: nn.Module, target_name: str):
        """
        Prevent applying LoRA to incompatible modules in specific architectures (e.g., Mamba).
        """
        _check_lora_target_modules_mamba(peft_config, model, target_name)

    def _create_and_replace_parameter(
        self, peft_config, adapter_name, target, target_name, parent, current_key
    ) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not support targeting nn.Parameter.")

    def inject_adapter(
        self,
        model: nn.Module,
        adapter_name: str,
        autocast_adapter_dtype: bool = True,
        low_cpu_mem_usage: bool = False,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        r"""
        Creates adapter layers and replaces the target modules with the adapter layers. This method is called under the
        hood by `peft.mapping.get_peft_model` if a non-prompt tuning adapter class is passed.

        The corresponding PEFT config is directly retrieved from the `peft_config` attribute of the BaseTuner class.

        Args:
            model (`nn.Module`):
                The model to be tuned.
            adapter_name (`str`):
                The adapter name.
            autocast_adapter_dtype (`bool`, *optional*):
                Whether to autocast the adapter dtype. Defaults to `True`.
            low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
                Create empty adapter weights on meta device. Useful to speed up the loading process.
            state_dict (`dict`, *optional*, defaults to `None`)
                If a state_dict is passed here, the adapters will be injected based on the entries of the state_dict.
                This can be useful when the exact `target_modules` of the PEFT method is unknown, for instance because
                the checkpoint was created without meta data. Note that the values from the state_dict are not used,
                only the keys are used to determine the correct layers that should be adapted.

        """
        ###################################
        # PREPARATION OF MODEL AND CONFIG #
        ###################################

        peft_config = self.peft_config[adapter_name]
        excluded_modules = []
        unmatched_modules = []
        targeted_modules_from_peft_config: list[str] = []  # only relevant if state_dict is passed
        # Note: If possible, all checks should be performed *at the start of this method*.
        # This way, we can raise early if something goes wrong, without leaving the model
        # in a bad (half-initialized) state.
        self._check_new_adapter_config(peft_config)

        self._check_tied_modules(model, peft_config)

        model_config = self.get_model_config(model)

        peft_config = self._prepare_adapter_config(peft_config, model_config)

        self._prepare_model(peft_config, model)

        if getattr(peft_config, "target_parameters", []) and state_dict:
            raise ValueError(
                "Trying to inject a PEFT adapter from a state_dict but the PEFT config uses `target_parameters`. This "
                "is not supported -- when using `target_parameters`, please inject the adapter without the state_dict."
            )

        named_modules = list(model.named_modules())
        key_list = [key for key, _ in named_modules]

        uses_dummy_target_modules = getattr(peft_config, "target_modules", None) == DUMMY_TARGET_MODULES
        if uses_dummy_target_modules:
            # dummy adapter, we allow not matching any module
            named_modules = []
            key_list = []

        # update peft_config.target_modules if required
        peft_config = _maybe_include_all_linear_layers(peft_config, model)

        # This is an optimization to reduce the number of entries in the target_modules list. The reason is that in some
        # circumstances, target_modules can contain hundreds of entries. Since each target module is checked against
        # each module of the net (which can be thousands), this can become quite expensive when many adapters are being
        # added. Often, the target_modules can be condensed in such a case, which speeds up the process.
        # A context in which this can happen is when diffusers loads non-PEFT LoRAs. As there is no meta info on
        # target_modules in that case, they are just inferred by listing all keys from the state_dict, which can be
        # quite a lot. See: https://github.com/huggingface/diffusers/issues/9297
        # As there is a small chance for undiscovered bugs, we apply this optimization only if the list of
        # target_modules is sufficiently big.
        # We also exclude IA³ from this optimization. This is because IA³ has both target_modules and
        # feedforward_modules, which are coupled (the latter must be a subset). It would be possible to change the logic
        # to keep both in sync, but it's not quite trivial and probably not worth the effort. See #2429.
        if (
            isinstance(peft_config.target_modules, (list, set))
            and (len(peft_config.target_modules) >= MIN_TARGET_MODULES_FOR_OPTIMIZATION)
            and (peft_config.peft_type != PeftType.IA3)
        ):
            suffixes = tuple("." + suffix for suffix in peft_config.target_modules)
            names_no_target = [
                name for name in key_list if (name not in peft_config.target_modules) and not name.endswith(suffixes)
            ]
            new_target_modules = _find_minimal_target_modules(peft_config.target_modules, names_no_target)
            if len(new_target_modules) < len(peft_config.target_modules):
                peft_config.target_modules = new_target_modules

        ###############################
        # MATCHING & CREATING MODULES #
        ###############################

        existing_adapter_prefixes = []
        for key, module in named_modules:
            if isinstance(module, BaseTunerLayer):
                existing_adapter_prefixes.append(key + ".")

        # TODO: check if this the most robust way
        module_names: set[str] = set()
        if state_dict is not None:
            prefix = PEFT_TYPE_TO_PREFIX_MAPPING[peft_config.peft_type]
            module_names = {k.rsplit("." + prefix, 1)[0] for k in state_dict}

        for key, module in named_modules:
            if not key:
                continue

            # It is possible that we're adding an additional adapter, so if we encounter a key that clearly belongs to a
            # previous adapter we can skip here since we don't want to interfere with adapter internals.
            for adapter_key in existing_adapter_prefixes:
                if key.startswith(adapter_key):
                    excluded_modules.append(key)
                    break

            if excluded_modules and excluded_modules[-1] == key:
                continue

            if state_dict is None:
                # normal mechanism: match the modules using the peft_config
                result = self._check_target_module_exists(peft_config, key)
                if isinstance(result, _ExcludedModule):
                    excluded_modules.append(key)
                elif not result:
                    unmatched_modules.append(key)
                else:
                    self.targeted_module_names.append(key)
                    parent, target, target_name = _get_submodules(model, key)
                    self._check_target_module_compatiblity(peft_config, model, target_name)
                    ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
                    with ctx():
                        self._create_and_replace(
                            peft_config, adapter_name, target, target_name, parent, current_key=key
                        )
            else:
                # use the state_dict to match modules instead
                if key not in module_names:
                    unmatched_modules.append(key)
                else:
                    self.targeted_module_names.append(key)
                    parent, target, target_name = _get_submodules(model, key)
                    self._check_target_module_compatiblity(peft_config, model, target_name)
                    ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
                    with ctx():
                        self._create_and_replace(
                            peft_config, adapter_name, target, target_name, parent, current_key=key
                        )

                # still record what would have been matched via the config so that the two results can be compared
                if self._check_target_module_exists(peft_config, key):
                    targeted_modules_from_peft_config.append(key)

        if getattr(peft_config, "target_parameters", []):
            # Note: We don't need to check for no state_dict being passed, since we already checked this earlier.
            self._inject_parameters(
                peft_config=peft_config, model=model, adapter_name=adapter_name, low_cpu_mem_usage=low_cpu_mem_usage
            )

        ####################
        # CHECK FOR ERRORS #
        ####################

        if state_dict is not None:
            # in case that the state_dict was used as source of truth and it resulted in different outcomes than what
            # would have been matched with the PEFT config, warn the user about that.
            targeted_set_from_peft_config = set(targeted_modules_from_peft_config)
            targeted_set_from_state_dict = set(self.targeted_module_names)
            diff_peft_config = targeted_set_from_peft_config - targeted_set_from_state_dict
            diff_state_dict = targeted_set_from_state_dict - targeted_set_from_peft_config
            warning_msg = ""
            if diff_peft_config or diff_state_dict:
                warning_msg = (
                    "While injecting the PEFT adapters, an inconsistency was discovered between the PEFT config and "
                    "the provided state_dict. This is not necessarily an issue and can be ignored if this was the "
                    "intent. "
                )
            if diff_peft_config:
                warning_msg += (
                    f"The PEFT config contained these additional target modules: {sorted(diff_peft_config)}. "
                )
            if diff_state_dict:
                warning_msg += f"The state_dict contained these additional target modules: {sorted(diff_state_dict)}. "
            if warning_msg:
                warnings.warn(warning_msg, RuntimeWarning)

        if not self.targeted_module_names and not self.targeted_parameter_names and not uses_dummy_target_modules:
            if excluded_modules and not unmatched_modules:
                # All targeted modules were excluded
                raise ValueError(
                    "All modules were excluded. This is likely unintended. "
                    "Check your `target_modules`, `exclude_modules` and `modules_to_save` configuration."
                )
            elif not excluded_modules and unmatched_modules and not peft_config.target_modules:
                raise ValueError(
                    "No `target_modules` passed but also no `target_parameters` found. Please check the values for "
                    "these arguments."
                )
            elif not excluded_modules and unmatched_modules:
                # None of the targeted modules matched
                error_msg = (
                    f"Target modules {peft_config.target_modules} not found in the base model. "
                    f"Please check the target modules and try again."
                )
                if getattr(peft_config, "layers_to_transform", None) is not None:
                    error_msg += f" Note: You specified 'layers_to_transform': {peft_config.layers_to_transform}."
                if getattr(peft_config, "layers_pattern", None) is not None:
                    error_msg += f" You also specified 'layers_pattern': {peft_config.layers_pattern}."
                raise ValueError(error_msg)
            else:
                # Some modules did not match and some matched but were excluded
                error_msg = (
                    "No modules were targeted for adaptation. "
                    "This might be caused by a combination of mismatched target modules and excluded modules. "
                    "Please check your `target_modules` and `exclude_modules` configuration. You may also have "
                    "only targeted modules that are marked to be saved (`modules_to_save`)."
                )
                if getattr(peft_config, "layers_to_transform", None) is not None:
                    error_msg += f" Note: You specified 'layers_to_transform': {peft_config.layers_to_transform}."
                if getattr(peft_config, "layers_pattern", None) is not None:
                    error_msg += f" You also specified 'layers_pattern': {peft_config.layers_pattern}."
                raise ValueError(error_msg)

        elif hasattr(peft_config, "exclude_modules") and peft_config.exclude_modules and not excluded_modules:
            # exclude_modules was passed but was not used
            warnings.warn(
                f"You have passed exclude_modules={peft_config.exclude_modules} but no modules were excluded. "
                "Please check that exclude_modules was set correctly."
            )

        elif not uses_dummy_target_modules:
            # If we landed here, it means that at least one module or parameter was adapted, so let's not raise an
            # error. However, let's warn the user if it seems like
            # - they wanted to match a module but there was no match
            # - they wanted to match a parameter but there was no match
            if peft_config.target_modules and not self.targeted_module_names:
                warnings.warn(
                    f"target_modules={peft_config.target_modules} were set but no module was matched.", RuntimeWarning
                )
            elif getattr(peft_config, "target_parameters", []) and not self.targeted_parameter_names:
                warnings.warn(
                    f"target_parameters={peft_config.target_parameters} were set but no parameter was matched.",
                    RuntimeWarning,
                )

        tied_target_modules = self._get_tied_target_modules(model=model)
        if tied_target_modules:
            warnings.warn(
                f"Model with `tie_word_embeddings=True` and the {tied_target_modules=} are part of the adapter. "
                "This can lead to complications, for example when merging the adapter "
                "or converting your model to formats other than safetensors. "
                "See for example https://github.com/huggingface/peft/issues/2018."
            )

        ################
        # HOUSEKEEPING #
        ################

        # It's important to set the adapter here (again), because otherwise it can happen that if a 2nd adapter is
        # added, and it targets different layer(s) than the first adapter (which is active), then those different
        # layers will be activated, which we don't want.
        self.set_adapter(self.active_adapters, inference_mode=peft_config.inference_mode)
        self._mark_only_adapters_as_trainable(model)

        if self.peft_config[adapter_name].inference_mode:
            for n, p in model.named_parameters():
                if adapter_name in n:
                    p.requires_grad = False

        set_additional_trainable_modules(
            model=model,
            peft_config=peft_config,
            model_config=BaseTuner.get_model_config(self),
            adapter_name=adapter_name,
            activate_adapter=adapter_name in self.active_adapters,
        )

    def _inject_parameters(
        self, peft_config: PeftConfig, model: nn.Module, adapter_name: str, low_cpu_mem_usage: bool
    ) -> None:
        """Inject layers based on peft_config.target_modules"""

        def strip_base_layer_from_name(module_name):
            # It is possible that the layer is already a PEFT layer and needs updating with a new adapter. In this case,
            # the name of parameter would be something like `model.layers.0.experts.base_layer.weight`, i.e. there is a
            # "base_layer" inserted in the name. We need to remove that, otherwise we won't be able to match correctly
            # (in this case, "experts.weight" would not match).
            name = ".base_layer"
            while name in module_name:
                prefix, _, suffix = module_name.rpartition(name)
                module_name = prefix + suffix
            return module_name

        def create_and_replace_param(module_name, key, param_name):
            # helper function to avoid duplication
            parent, target, target_name = _get_submodules(model, module_name)
            unwrapped_module_name = strip_base_layer_from_name(module_name)
            unwrapped_module = model.get_submodule(unwrapped_module_name)
            # use the class name for checking to avoid circular import
            if isinstance(unwrapped_module, BaseTunerLayer) and unwrapped_module.__class__.__name__ != "ParamWrapper":
                raise ValueError(
                    f"Trying to wrap an `nn.Parameter` of layer '{unwrapped_module_name}' of type "
                    f"{type(target).__name__}, which is not a valid target. Make sure that this layer is not "
                    "also targeted with `target_modules`. For some models, PEFT will do this automatically, "
                    "try setting `target_modules=[]` to prevent it."
                )

            self._check_target_module_compatiblity(peft_config, model, target_name)
            ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
            with ctx():
                self._create_and_replace(
                    peft_config,
                    adapter_name,
                    target,
                    target_name,
                    parent,
                    current_key=key,
                    parameter_name=param_name.rpartition(".")[-1],
                )

        # TODO very simple matching, might not cover all use cases
        unsorted_target_names = set(peft_config.target_parameters)
        # As the order of matching can influence the nesting of multiple params on the same module, ensure determinism
        # by sorting.
        target_names = sorted(unsorted_target_names)
        for module_name, module in model.named_modules():
            if hasattr(module, "parametrizations"):
                # Deal with the case that the parameter is already parametrized. The issue is that we would not be able
                # to match `f"{module_name}.{param_name}"`, as the parameter is now something like
                # `module.parametrization.weight`.
                for key in target_names:
                    target_module_name, _, param_name = key.rpartition(".")
                    if target_module_name != module_name:
                        continue
                    if getattr(module, param_name, None) is None:
                        continue
                    create_and_replace_param(module_name, key, param_name)
                    self.targeted_parameter_names.append(key)
            else:
                # Standard case: the parameter is not already parametrized. Note, however, that the model could already
                # be nested with lora.ParamWrapper, as this is how we allow targeting multiple Parameters on the same
                # module.
                unwrapped_module_name = strip_base_layer_from_name(module_name)
                # we're interested in finding the "lowest" module that contains the parameter, hence recurse=False
                for param_name, param in module.named_parameters(recurse=False):
                    key = f"{unwrapped_module_name}.{param_name}"
                    if (key in target_names) or any(key.endswith(f".{target_key}") for target_key in target_names):
                        # Note: We use the unwrapped_module_name to check if the key matches, but we use the module_name for
                        # replacement, since we want to replace the wrapped module.
                        create_and_replace_param(module_name, key, param_name)
                        self.targeted_parameter_names.append(key)

    def _replace_module(self, parent, child_name, new_module, child) -> None:
        """
        Replace the sub-module of a given moduel with a new PEFT module.

        This also deals with device placement of the new module to be in line with the child module.

        Args:
            parent (`nn.Module`):
                The parent module on which the replacement should take place.
            child_name (`str`):
                The name of the child module to be replaced.
            new_module (`nn.Module`):
                The new PEFT module.
            child (`nn.Module`):
                The original child module that is being replaced.

        """
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        meta = torch.device("meta")
        # dispatch to correct device
        for name, module in new_module.named_modules():
            if self.prefix in name:
                if hasattr(child, "qweight"):
                    weight = child.qweight
                elif hasattr(child, "W_q"):
                    weight = child.W_q
                elif hasattr(child, "weight"):
                    weight = child.weight
                elif getattr(child, "in_proj_weight", None) is not None:  # MHA
                    weight = child.in_proj_weight
                else:
                    weight = next(child.parameters())

                if not any(p.device == meta for p in module.parameters()):
                    module.to(weight.device)

    def merge_adapter(self, adapter_names: Optional[list[str]] = None, safe_merge: bool = False) -> None:
        """
        This method merges the adapter layers into the base model.

        Merging adapters can lead to a speed up of the forward pass. A copy of the adapter weights is still kept in
        memory, which is required to unmerge the adapters. In order to merge the adapter weights without keeping them
        in memory, please call `merge_and_unload`.

        Args:
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If `None`, all active adapters will be merged.
                Defaults to `None`.
            safe_merge (`bool`, *optional*):
                If `True`, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
        """
        # Note: The order of arguments here is:
        #   adapter_names, safe_merge
        # For layer.merge, the order is:
        #   safe_merge, adapter_names
        # This is not so nice but this method here started with only adapter_names, thus putting safe_merge first would
        # be a backwards incompatible change.
        self._check_merge_allowed()
        for module in self.model.modules():
            if isinstance(module, BaseTunerLayer):
                with onload_layer(module):
                    module.merge(adapter_names=adapter_names, safe_merge=safe_merge)

    def unmerge_adapter(self):
        """
        This method unmerges all merged adapter layers from the base model.
        """
        for module in self.model.modules():
            if isinstance(module, BaseTunerLayer):
                with onload_layer(module):
                    module.unmerge()

    def set_adapter(self, adapter_name: str | list[str], inference_mode: bool = False) -> None:
        """Set the active adapter(s).

        Args:
            adapter_name (str, list[str]):
                The name(s) of the adapter(s) to set as active
            inference_mode (bool, optional):
                 Whether the activated adapter should be frozen (i.e. `requires_grad=False`). Default is False.
        """
        set_adapter(
            self.model, adapter_name=adapter_name, inference_mode=inference_mode, layer_cls=self.tuner_layer_cls
        )
        self.active_adapter = adapter_name

    @staticmethod
    def get_model_config(model: nn.Module) -> dict:
        """
        This method gets the config from a model in dictionary form. If model has not attribute config, then this
        method returns a default config.

        Args:
            model (`nn.Module`):
                Model to get the config from.
            default (`dict|None`, *optional*)::
                What to return if model does not have a config attribute.
        """
        model_config = getattr(model, "config", DUMMY_MODEL_CONFIG)
        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()
        elif dataclasses.is_dataclass(model_config):
            model_config = dataclasses.asdict(model_config)
        return model_config

    def _get_tied_target_modules(self, model: nn.Module) -> list[str]:
        tied_target_modules = []
        model_config = self.get_model_config(model)
        if model_config.get("tie_word_embeddings"):
            for target_module in self.targeted_module_names:
                # This potentially yields false positives since we're just looking at the layer names. So if we use a
                # model that uses weight-tying of lm_head and embed_tokens, a third, unrelated, layer which is
                # unfortunately named so that it is in EMBEDDING_LAYER_NAMES will be falsely reported here as well.
                if target_module.split(".")[-1] in EMBEDDING_LAYER_NAMES:
                    tied_target_modules.append(target_module)
        return tied_target_modules

    def _get_module_names_tied_with_embedding(self) -> list[str]:
        return _get_module_names_tied_with_embedding(self)

    def _add_modules_to_tie(self, peft_config, tied_weight_keys):
        """
        This method adds modules to tie to `peft_config` so that those modules can be tied downstream. By default this
        method raises a warning, and each tuner class extending `BaseTuner` can choose to implement this.
        """
        msg = (
            "Model has `tie_word_embeddings=True` and a tied layer is part of the adapter, "
            "but no implementation exists to tie the adapters. "
            "This can lead to complications, for example when merging the adapter "
            "or converting your model to formats other than safetensors. "
            "Check the discussion here: https://github.com/huggingface/peft/issues/2777"
        )
        warnings.warn(msg)

    def _check_tied_modules(self, model: nn.Module, peft_config):
        """
        Checks if any of the tied layers are targetted via `modules_to_save`. Updates the `peft_config.modules_to_tie`
        with any layers that needs to be tied
        """
        modules_to_save = set(getattr(peft_config, "modules_to_save", []) or [])
        is_embedding_to_save = any(m in EMBEDDING_LAYER_NAMES for m in modules_to_save)

        tied_weight_keys = self._get_module_names_tied_with_embedding()

        if getattr(peft_config, "ensure_weight_tying", False):
            if is_embedding_to_save and tied_weight_keys:
                self._add_modules_to_tie(peft_config, tied_weight_keys)

            elif not is_embedding_to_save and tied_weight_keys:
                warnings.warn(
                    "You have requested `ensure_weight_tying`, but no tied modules are added in `modules_to_save`"
                )

            elif not tied_weight_keys:
                warnings.warn("You have requested `ensure_weight_tying`, but no tied modules were found in the model")

        elif is_embedding_to_save and tied_weight_keys:
            if hasattr(peft_config, "ensure_weight_tying"):
                msg = (
                    "Model has `tie_word_embeddings=True` and a tied layer is part of the adapter, "
                    "but `ensure_weight_tying` is not set to True. "
                    "This can lead to complications, for example when merging the adapter "
                    "or converting your model to formats other than safetensors. "
                    "Check the discussion here: https://github.com/huggingface/peft/issues/2777"
                )
                warnings.warn(msg)
            else:
                msg = (
                    "Model has `tie_word_embeddings=True` and a tied layer is part of the adapter, "
                    "but no implementation exists to tie the adapters. "
                    "This can lead to complications, for example when merging the adapter "
                    "or converting your model to formats other than safetensors. "
                    "Check the discussion here: https://github.com/huggingface/peft/issues/2777"
                )
                warnings.warn(msg)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.model, name)


class BaseTunerLayer(ABC):
    r"""
    A tuner layer mixin that provides the common methods and attributes for all tuners.

    Args:
        is_pluggable (`bool`, *optional*):
            Whether the adapter layer can be plugged to any pytorch module
        active_adapters (Union[List[`str`], `str`], *optional*):
            The name of the active adapter.
    """

    # All names of layers that may contain adapter (trainable) weights
    adapter_layer_names: tuple[str, ...] = ()
    # All names of other parameters that may contain adapter-related parameters
    other_param_names: tuple[str, ...] = ()

    # indicates whether all adapters should be disabled
    _disable_adapters: bool = False

    # the currently active adapter(s)
    _active_adapter: str | list[str] = "default"

    # List all merged adapters
    merged_adapters: list[str] = []

    def get_base_layer(self) -> nn.Module:
        """
        (Recursively) get the base_layer.

        This is necessary for the case that the tuner layer wraps another tuner layer.

        """
        base_layer = self
        while hasattr(base_layer, "base_layer"):
            base_layer = base_layer.base_layer
        return base_layer

    def _get_embed_scale(self):
        """
        Extract embed_scale from base layer if present and valid.

        Some embedding layers (e.g., Gemma3TextScaledWordEmbedding) apply scaling to embeddings in their forward
        method. This method checks for the presence of an `embed_scale` attribute. If it exists, it is assumed to be a
        scalar. Its shape is validated accordingly.

        Returns:
            torch.Tensor or None: The embed_scale tensor if found and valid, None otherwise.
        """
        base_layer = self.get_base_layer()
        if not hasattr(base_layer, "embed_scale"):
            return None

        embed_scale = base_layer.embed_scale

        # Convert scalar values to tensors
        if isinstance(embed_scale, (int, float)):
            return torch.tensor(embed_scale, device=base_layer.weight.device, dtype=base_layer.weight.dtype)

        # Validate tensor shape - must be scalar (0-d) or 1-element tensor for proper broadcasting
        if isinstance(embed_scale, torch.Tensor):
            if embed_scale.numel() == 1:
                return embed_scale
            else:
                # Log warning but don't fail - this maintains backward compatibility
                warnings.warn(
                    f"Found embed_scale attribute with shape {embed_scale.shape}, expected scalar. "
                    "Embedding scaling will not be applied. If this is unexpected, please open an issue at "
                    "https://github.com/huggingface/peft/issues",
                    PeftWarning,
                )
                return None

        return None

    @property
    def weight(self) -> torch.Tensor:
        # This is required for some transformers code, e.g. for T5, weight is accessed as:
        #     self.wo.weight
        # where "wo" is the adapter layer.
        # https://github.com/huggingface/transformers/blob/78f6ed6c70b29c1560780e3869a7ad4c6b3d2710/src/transformers
        # /models/t5/modeling_t5.py#L292
        base_layer = self.get_base_layer()
        if hasattr(base_layer, "qweight"):
            # QuantLinear
            weight = base_layer.qweight
        else:
            # Other layers
            weight = base_layer.weight
        return weight

    @property
    def bias(self) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return base_layer.bias

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError

    def unmerge(self) -> None:
        raise NotImplementedError

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    @property
    def disable_adapters(self) -> bool:
        # use a property to ensure that disable_adapters is not set directly, instead use the enable_adapters method
        return self._disable_adapters

    @property
    def active_adapter(self) -> str | list[str]:
        # use a property to ensure that active_adapter is not set directly, instead use the set_adapter method
        return self._active_adapter

    def _get_available_adapters(self) -> set[str]:
        """Return all adapter names that can be found on this module."""
        adapters = set()
        for layer_name in self.adapter_layer_names:
            module = getattr(self, layer_name)
            if not isinstance(module, (nn.ModuleDict, nn.ParameterDict)):
                continue
            adapters.update(set(module.keys()))
        return adapters

    @property
    def active_adapters(self):
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        # is already a list of str
        return self.active_adapter

    def enable_adapters(self, enabled: bool) -> None:
        """Toggle the enabling and disabling of adapters

        Takes care of setting the requires_grad flag for the adapter weights.

        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
        if enabled:
            self.set_adapter(self.active_adapters)
            self._disable_adapters = False
        else:
            # disable grads on all adapter layers
            for layer_name in self.adapter_layer_names:
                layer = getattr(self, layer_name)
                layer.requires_grad_(False)
            self._disable_adapters = True

    def set_adapter(self, adapter_names: str | list[str], inference_mode: bool = False) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapter to trainable (i.e., requires_grad=True) unless
        inference_mode is True.

        Args:
            adapter_name (`str` or `list[str]`):
                 The name(s) of the adapter(s) to set as active.
            inference_mode (bool, optional):
                 Whether the activated adapter should be frozen (i.e. `requires_grad=False`). Default is False.
        """
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Deactivate grads on the inactive adapter and activate grads on the active adapter (if not in inference mode)
        for layer_name in self.adapter_layer_names:
            module_dict = getattr(self, layer_name)
            for key, layer in module_dict.items():
                if (key in adapter_names) and (not inference_mode):
                    # Note: It is possible that not a single layer is called with requires_grad_(True) here. This may
                    # happen if a completely different adapter layer is being activated.
                    layer.requires_grad_(True)
                else:
                    layer.requires_grad_(False)

        self._active_adapter = adapter_names

    def _all_available_adapter_names(self) -> list[str]:
        """Return a sorted list of all available adapter names"""
        adapter_names = set()
        for name in self.adapter_layer_names + self.other_param_names:
            # we check each possible attribute and if it's a dict or ModuleDict, we assume that the keys are the adapter
            # names
            attr = getattr(self, name)
            if hasattr(attr, "keys"):
                adapter_names.update(attr.keys())
        return sorted(adapter_names)

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Delete an adapter from the layer

        This should be called on all adapter layers, or else we will get an inconsistent state.

        This method will also set a new active adapter if the deleted adapter was an active adapter. It is important
        that the new adapter is chosen in a deterministic way, so that the same adapter is chosen on all layers.

        Args:
            adapter_name (`str`): The name of the adapter to delete

        """
        for attr in self.adapter_layer_names + self.other_param_names:
            if adapter_name in getattr(self, attr):
                del getattr(self, attr)[adapter_name]

        if adapter_name in self.active_adapters:
            # choose a new active adapter
            active_adapters = self.active_adapters[:]
            active_adapters.remove(adapter_name)
            if active_adapters:
                self.set_adapter(active_adapters)
            else:
                # no active adapters left, set a new default adapter
                # here we get the list of all adapters existing adapter names and choose the first one
                remaining_adapters = self._all_available_adapter_names()
                if not remaining_adapters:
                    self.set_adapter([])
                else:
                    new_active_adapter = remaining_adapters[0]
                    warnings.warn(
                        f"Adapter {adapter_name} was active which is now deleted. Setting active adapter to "
                        f"{new_active_adapter}."
                    )
                    self.set_adapter(remaining_adapters[0])

    def set_requires_grad(self, adapter_names: str | Sequence[str], requires_grad: bool = True) -> None:
        """
        Enable or disable gradients on the given adapter(s).

        Args:
            adapter_name (`str` or `Sequence[str]`):
                The name of the adapter(s) whose gradients should be enabled/disabled.
            requires_grad (`bool`, *optional*)
                Whether to enable (`True`, default) or disable (`False`).
        """
        if isinstance(adapter_names, str):
            adapter_names_set = {adapter_names}
        else:
            adapter_names_set = set(adapter_names)

        for layer_name in self.adapter_layer_names:
            module_dict = getattr(self, layer_name)
            for key, layer in module_dict.items():
                if key in adapter_names_set:
                    layer.requires_grad_(requires_grad)

    def _move_adapter_to_device_of_base_layer(self, adapter_name: str, device: Optional[torch.device] = None) -> None:
        """
        Move the adapter of the given name to the device of the base layer.
        """
        if device is None:
            base_layer = self.get_base_layer()
            if isinstance(base_layer, nn.MultiheadAttention):
                base_layer = base_layer.out_proj
            # check weight and qweight (for GPTQ)
            for weight_name in ("weight", "qweight"):
                weight = getattr(base_layer, weight_name, None)
                if weight is not None:
                    device = weight.device
                    dtype = weight.dtype
                    break
            else:
                # no break encountered: could not determine the device
                return

        meta = torch.device("meta")

        # loop through all potential adapter layers and move them to the device of the base layer; be careful to only
        # move this specific adapter to the device, as the other adapters could be on different devices
        # see #1639
        for adapter_layer_name in self.adapter_layer_names + self.other_param_names:
            adapter_layer = getattr(self, adapter_layer_name, None)
            if not isinstance(adapter_layer, (nn.ModuleDict, nn.ParameterDict, BufferDict)):
                continue
            if adapter_name not in adapter_layer:
                continue
            if any(p.device == meta for p in adapter_layer.parameters()):
                continue

            # TODO: weight is not necessarily defined here, leading to a NameError, fix that
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device, dtype=dtype)
            else:
                adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device)

    @overload
    def _cast_input_dtype(self, x: None, dtype: torch.dtype) -> None: ...

    @overload
    def _cast_input_dtype(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor: ...

    def _cast_input_dtype(self, x, dtype: torch.dtype):
        """
        Whether to cast the dtype of the input of the forward method.

        Usually, we want to enable this to align the input dtype with the dtype of the weight, but by setting
        layer.cast_input_dtype=False, this can be disabled if necessary.

        Enabling or disabling can be managed via the peft.helpers.disable_lora_input_dtype_casting context manager.
        """
        if x is None:  # useful e.g. if x is the bias, which can be None
            return None

        cast_input_dtype_enabled = getattr(self, "cast_input_dtype_enabled", True)
        if (not cast_input_dtype_enabled) or (x.dtype == dtype):
            return x
        return x.to(dtype=dtype)


def _find_minimal_target_modules(
    target_modules: list[str] | set[str], other_module_names: list[str] | set[str]
) -> set[str]:
    """Find the minimal set of target modules that is sufficient to separate them from the other modules.

    Sometimes, a very large list of target_modules could be passed, which can slow down loading of adapters (e.g. when
    loaded from diffusers). It may be possible to condense this list from hundreds of items to just a handful of
    suffixes that are sufficient to distinguish the target modules from the other modules.

    Example:
        ```py
        >>> from peft.tuners.tuners_utils import _find_minimal_target_modules

        >>> target_modules = [f"model.decoder.layers.{i}.self_attn.q_proj" for i in range(100)]
        >>> target_modules += [f"model.decoder.layers.{i}.self_attn.v_proj" for i in range(100)]
        >>> other_module_names = [f"model.encoder.layers.{i}.self_attn.k_proj" for i in range(100)]
        >>> _find_minimal_target_modules(target_modules, other_module_names)
        {"q_proj", "v_proj"}
        ```

    Args:
        target_modules (`list[str]` | `set[str]`):
            The list of target modules.
        other_module_names (`list[str]` | `set[str]`):
            The list of other module names. They must not overlap with the target modules.

    Returns:
        `set[str]`:
            The minimal set of target modules that is sufficient to separate them from the other modules.

    Raises:
        ValueError:
            If `target_modules` is not a list or set of strings or if it contains an empty string. Also raises an error
            if `target_modules` and `other_module_names` contain common elements.
    """
    if isinstance(target_modules, str) or not target_modules:
        raise ValueError("target_modules should be a list or set of strings.")

    target_modules = set(target_modules)
    if "" in target_modules:
        raise ValueError("target_modules should not contain an empty string.")

    other_module_names = set(other_module_names)
    if not target_modules.isdisjoint(other_module_names):
        msg = (
            "target_modules and other_module_names contain common elements, this should not happen, please "
            "open a GitHub issue at https://github.com/huggingface/peft/issues with the code to reproduce this issue"
        )
        raise ValueError(msg)

    # it is assumed that module name parts are separated by a "."
    def generate_suffixes(s):
        parts = s.split(".")
        return [".".join(parts[i:]) for i in range(len(parts))][::-1]

    # Create a reverse lookup for other_module_names to quickly check suffix matches
    other_module_suffixes = {suffix for item in other_module_names for suffix in generate_suffixes(item)}

    # Find all potential suffixes from target_modules
    target_modules_suffix_map = {item: generate_suffixes(item) for item in target_modules}

    # Initialize a set for required suffixes
    required_suffixes = set()

    # We sort the target_modules_suffix_map simply to get deterministic behavior, since sets have no order. In theory
    # the order should not matter but in case there is a bug, it's better for the bug to be deterministic.
    for item, suffixes in sorted(target_modules_suffix_map.items(), key=lambda tup: tup[1]):
        # Go through target_modules items, shortest suffixes first
        for suffix in suffixes:
            # If the suffix is already in required_suffixes or matches other_module_names, skip it
            if suffix in required_suffixes or suffix in other_module_suffixes:
                continue
            # Check if adding this suffix covers the item
            if not any(item.endswith("." + req_suffix) for req_suffix in required_suffixes):
                required_suffixes.add(suffix)
                break

    if not required_suffixes:
        return set(target_modules)
    return required_suffixes


class _ExcludedModule:
    """
    A private helper method used to represent excluded modules in the check_target_module_exists function.
    """

    def __bool__(self):
        return False


def check_target_module_exists(config, key: str) -> bool | re.Match[str] | None:
    """A helper method to check if the passed module's key name matches any of the target modules in the adapter_config.

    Args:
        config (`PeftConfig`):
            A config to match target modules from.
        key (`str`):
            A key to search any matches in config

    Returns:
        `bool` | `re.Match[str]` | `None`:
            True or re.Match object if key matches any target modules from config, False or None if no match found.
    """
    if hasattr(config, "exclude_modules") and config.exclude_modules:
        if isinstance(config.exclude_modules, str):
            if re.fullmatch(config.exclude_modules, key):
                return _ExcludedModule()
        elif key in config.exclude_modules:
            return _ExcludedModule()
        elif any(key.endswith(f".{exclude_key}") for exclude_key in config.exclude_modules):
            return _ExcludedModule()

    # Adapters should never match on modules to save modules as it is a guarantee for conflicts of behavior
    # between `ModulesToSaveWrapper` internals and the potential adapter.
    modules_to_save = getattr(config, "modules_to_save", None)
    if modules_to_save:
        if any(re.match(rf"(^|.*\.){m}($|\..*)", key) for m in modules_to_save):
            return _ExcludedModule()

    if (config.target_modules is None) and (config.target_parameters is not None):
        # this is allowed if config.target_parameters are specified
        return False

    if isinstance(config.target_modules, str):
        target_module_found = match_target_against_key(config.target_modules, key)
    elif key in config.target_modules:
        # this module is specified directly in target_modules
        target_module_found = True
    else:
        target_module_found = any(key.endswith(f".{target_key}") for target_key in config.target_modules)

        layer_indexes = getattr(config, "layers_to_transform", None)
        layers_pattern = getattr(config, "layers_pattern", None)

        is_using_layer_indexes = layer_indexes is not None and (
            len(layer_indexes) != 0 if isinstance(layer_indexes, list) else True
        )
        if is_using_layer_indexes and target_module_found:
            layer_index = None
            # TODO: It's still unclear how empty layers_pattern (None, [], or "") should behave
            # For now, empty layers_pattern means any layer pattern is ok
            if layers_pattern is None or len(layers_pattern) == 0:
                layer_index = re.match(r".*\.[^.]*\.(\d+)\.", key)
            else:
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern
                for pattern in layers_pattern:
                    layer_index = re.match(rf".*\.{pattern}\.(\d+)\.", key)
                    if layer_index is not None:
                        break

            if layer_index is None:
                target_module_found = False
            else:
                layer_index = int(layer_index.group(1))
                if isinstance(layer_indexes, int):
                    target_module_found = layer_index == layer_indexes
                else:
                    target_module_found = layer_index in layer_indexes

    return target_module_found


def inspect_matched_modules(tuner: BaseTuner, adapter_name: str = "default") -> dict:
    """
    A helper function to inspect the set of matched and unmatched modules for a PEFT model and the given adapter.
    """
    config = tuner.peft_config[adapter_name]
    key_list = [key for key, _ in tuner.model.named_modules()]
    module_dict = {"matched": [], "unmatched": []}
    for key in key_list:
        if tuner._check_target_module_exists(config, key):
            module_dict["matched"].append(key)
        else:
            module_dict["unmatched"].append(key)
    return module_dict


def _maybe_include_all_linear_layers(peft_config: PeftConfig, model: nn.Module) -> PeftConfig:
    """
    Helper function to update `target_modules` to all linear/Conv1D layers if provided as 'all-linear'. Adapted from
    the QLoRA repository: https://github.com/artidoro/qlora/blob/main/qlora.py
    """
    if not hasattr(peft_config, "target_modules"):
        return peft_config

    # if `target_modules` is a string, convert to lower case and check if it matches "all-linear"
    if not (
        isinstance(peft_config.target_modules, str)
        and peft_config.target_modules.lower() == INCLUDE_LINEAR_LAYERS_SHORTHAND
    ):
        return peft_config

    linear_classes = (torch.nn.Linear, Conv1D)
    linear_names = ("Linear",)
    linear_module_names = set()
    for name, module in model.named_modules():
        # match with all linear classes.
        if isinstance(module, linear_classes):
            linear_module_names.add(name)
        elif isinstance(module, BaseTunerLayer) and any(n in type(module).__name__ for n in linear_names):
            # If the model already has adapter layers applied, then the "linear" layer is actually an adapter layer,
            # e.g. lora.Linear, and not nn.Linear. To target this layer, we don't want to check the layer type, as there
            # are many possible layer types (one for each PEFT method) and the list would quickly get out of date. Thus
            # we rely on the name of the layer class, which by convention is something like "Linear", "Linear4bit",
            # "HqqLoraLinear", ... in PEFT. It's not pretty but should generally work.
            # See 2390
            linear_module_names.add(name)

    # Try to remove linear layers that should not be targeted as best as possible. We have to rely on convention as
    # there are no hard rules to detect these modules.
    module_names_to_exclude = set()
    if isinstance(model, PreTrainedModel):
        output_emb = model.get_output_embeddings()
        if output_emb is not None:
            # ignore the last classification head for text generation models
            last_module_name = [name for name, module in model.named_modules() if module is output_emb][0]
            module_names_to_exclude.add(last_module_name)
        elif peft_config.task_type == TaskType.SEQ_CLS:
            # ignore classifier head for classification models (issue 2027)
            # there is no fix name for the classifier head, so check the common ones
            for name in SEQ_CLS_HEAD_NAMES:
                cls_head = getattr(model, name, None)
                if cls_head is not None:
                    last_module_name = [name for name, module in model.named_modules() if module is cls_head][0]
                    module_names_to_exclude.add(last_module_name)
                    break

    # we don't want nested LoRA layers, i.e. LoRA being applied to possibly existing lora_A, lora_B, etc.
    # see 2390
    for prefix, module in model.named_modules():
        if isinstance(module, BaseTunerLayer):
            for suffix, child in module.named_modules():
                if suffix:
                    module_names_to_exclude.add(f"{prefix}.{suffix}")

    linear_module_names -= module_names_to_exclude
    peft_config.target_modules = linear_module_names
    return peft_config


def check_adapters_to_merge(module: BaseTunerLayer, adapter_names: Optional[list[str]] = None) -> list[str]:
    """
    Helper function to check which adapters should be merged.

    Only return those adapters that are not already merged. Give a warning if some or all of the adapters are already
    merged.

    """
    if adapter_names is None:
        adapter_names = module.active_adapters
    if isinstance(adapter_names, str):
        raise ValueError(f"adapter_names should be a list of strings, got {adapter_names!r}.")

    if module.merged:
        merged_adapters = set(module.merged_adapters)
        adapter_names = [name for name in adapter_names if name not in merged_adapters]

        if adapter_names:
            warnings.warn(
                f"Already following adapters were merged {','.join(module.merged_adapters)}. "
                f"You are now additionally merging {','.join(adapter_names)}."
            )
        else:
            warnings.warn("All adapters are already merged, nothing to do.")

    return adapter_names


def clone_module(module: nn.Module, share_weights=False):
    """Clone a module in a pytorch model.

    Clones a module of a model, optionally sharing all the parameters between the original and the clone. Simplifies
    reusing a module when manipulating the architecture of a model.
    """
    clone = copy.deepcopy(module)

    def _share_weights(src: nn.Module, dst: nn.Module):
        for name, param in src.named_parameters(recurse=False):
            dst.register_parameter(name, param)

    if share_weights:
        for name, submodule in module.named_modules():
            _share_weights(submodule, clone.get_submodule(name))

    return clone


def replicate_layers(model: nn.Module, layer_map: list[tuple[int, int]]):
    """Replicate layers in a transfomer model with weight sharing.

    This function looks for a module list attribute at model[(.model)*].layers and replicates the layers in the module
    list according to the layer map. For example the map `[[0, 4], [2, 5]]` will take the set of layers `[0, 1, 2, 3,
    4]` and replace them with a module list containing `[0, 1, 2, 3, 2, 3, 4]`.
    """
    while hasattr(model, "model"):
        model = model.model
    # Some variants of the bert model nest the main model under the bert attribute.
    if hasattr(model, "bert"):
        model = model.bert

    model_type = None
    layers: nn.ModuleList = None
    if hasattr(model, "layers"):
        model_type = "llama"
        layers = model.layers
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        model_type = "bert"
        layers = model.encoder.layer
    elif hasattr(model, "h"):
        model_type = "falcon"
        layers = model.h
    if not model_type or not isinstance(layers, nn.ModuleList):
        raise ValueError(
            "Could not locate the layers attribute in the model. "
            "Expected Llama, Bert or Falcon compatible architectures."
        )

    new_layers = []
    for start, end in layer_map:
        for i in range(start, end):
            current_idx = len(new_layers)
            new_layers.append(clone_module(layers[i], share_weights=True))
            # This is a hack needed to work around the layer_idx introduced in HF transformers.
            for submodule in new_layers[-1].modules():
                if hasattr(submodule, "layer_idx"):
                    submodule.layer_idx = current_idx
    layers = nn.ModuleList(new_layers)
    if model_type == "llama":
        model.layers = layers
    elif model_type == "bert":
        model.encoder.layer = layers
    elif model_type == "falcon":
        model.h = layers
    else:
        raise ValueError("Unexpected model type, need to handle post-processing of layers.")
    if hasattr(model.config, "num_hidden_layers"):  # Common to Llama, Bert, Falcon.
        model.config.num_hidden_layers = len(new_layers)


###############################
# FUNCTIONS FOR functional.py #
###############################


def set_adapter(
    model,
    adapter_name: str | list[str],
    inference_mode: bool = False,
    layer_cls: type[BaseTunerLayer] = BaseTunerLayer,
) -> None:
    """Set the active PEFT adapter(s) of the model.

    Active adapters are those adapters that participate in the forward pass. Use this function if you want to switch
    between multiple PEFT adapters.

    Args:
        model (`nn.Module`):
            The model on which the adapter(s) should be set.
        adapter_name (str, list[str]):
            The name(s) of the adapter(s) to set as active
        inference_mode (bool, optional):
             Whether the activated adapter should be frozen (i.e. `requires_grad=False`). Default is False.
        layer_cls (type, optional):
            The class of the adapter layer. Defaults to `BaseTunerLayer`.
    """
    _set_adapter(model, adapter_name, inference_mode=inference_mode)  # auxiliary modules
    for module in model.modules():
        if isinstance(module, layer_cls):
            if module.merged:
                warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                module.unmerge()
            module.set_adapter(adapter_name, inference_mode=inference_mode)


def _delete_auxiliary_adapter(model, adapter_name: str, new_active_adapters: Optional[list[str]]) -> None:
    for module in model.modules():
        if isinstance(module, AuxiliaryTrainingWrapper):
            module.delete_adapter(adapter_name, new_active_adapters=new_active_adapters)


def delete_adapter(
    model: nn.Module, adapter_name: str, prefix: str, layer_cls: type[BaseTunerLayer] = BaseTunerLayer
) -> list[str] | None:
    """
    Delete an existing PEFT adapter.

    Note: This function does not delete the PEFT config on the model, if there is one. It will also not completely
    purge the PEFT layers if the last PEFT adapter is deleted. For this, consider using `model.unload()` if using a
    PEFT model instance, or just reloading the base model.

    Args:
        model (`nn.Module`):
            The model from which the adapter should be deleted.
        adapter_name (str):
            The name of the adapter to be deleted.
        prefix (str):
            The prefix of the PEFT method, e.g. "lora_" for LoRA.
        layer_cls (type, optional):
            The class of the adapter layer. Defaults to `BaseTunerLayer`.

    Returns:
        new_adapter (list[str] | None):
            The name of remaining adapter(s) after deletion, or `None` if there are no active adapters left. Use this
            to set the new active adapter of the model if necessary.
    """
    key_list = [key for key, _ in model.named_modules() if prefix not in key]
    new_adapter = None

    for key in key_list:
        _, target, _ = _get_submodules(model, key)
        if isinstance(target, layer_cls):
            target.delete_adapter(adapter_name)
            if new_adapter is None:
                new_adapter = target.active_adapters[:]

    _delete_auxiliary_adapter(model, adapter_name=adapter_name, new_active_adapters=new_adapter)
    return new_adapter


def cast_adapter_dtype(model: nn.Module, adapter_name: str, autocast_adapter_dtype: bool = True) -> None:
    """
    A helper method to cast the adapter weights to the correct dtype.

    Currently, this only upcasts float16 and bfloat16 to float32.

    Args:
        adapter_name (`str`):
            The adapter name.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`.
    """
    if not autocast_adapter_dtype:
        return

    dtypes_to_convert_to_fp32 = {torch.float16, torch.bfloat16}

    for module in model.modules():
        if not isinstance(module, BaseTunerLayer):
            continue

        for submodule in module.modules():
            if not isinstance(submodule, (nn.ModuleDict, nn.ParameterDict, BufferDict)):
                continue

            if adapter_name not in submodule:
                continue

            if isinstance(submodule[adapter_name], nn.Parameter):
                if submodule[adapter_name].dtype in dtypes_to_convert_to_fp32:
                    submodule[adapter_name].data = submodule[adapter_name].data.to(torch.float32)
                continue

            if isinstance(submodule[adapter_name], torch.Tensor):  # e.g. from a BufferDict
                if submodule[adapter_name].dtype in dtypes_to_convert_to_fp32:
                    submodule[adapter_name] = submodule[adapter_name].to(torch.float32)
                continue

            for param in submodule[adapter_name].parameters():
                if param.dtype in dtypes_to_convert_to_fp32:
                    param.data = param.data.to(torch.float32)


def set_requires_grad(model, adapter_names: str | Sequence[str], requires_grad: bool = True) -> None:
    """
    Enable or disable gradients on the given adapter(s).

    Args:
        model (`nn.Module`):
            The model from which the adapter should be deleted.
        adapter_name (`str` or `Sequence[str]`):
            The name of the adapter(s) whose gradients should be enabled/disabled.
        requires_grad (`bool`, *optional*)
            Whether to enable (`True`, default) or disable (`False`).
    """
    for module in model.modules():
        if isinstance(module, (BaseTunerLayer, AuxiliaryTrainingWrapper)):
            module.set_requires_grad(adapter_names=adapter_names, requires_grad=requires_grad)
