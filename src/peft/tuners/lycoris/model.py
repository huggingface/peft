# coding=utf-8
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

import warnings
from dataclasses import asdict
from enum import Enum
from typing import Any, Union

from torch import nn
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners import adalora, loha, lokr, lora
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftType,
    _get_submodules,
    get_auto_gptq_quant_linear,
)


if is_bnb_available():
    import bitsandbytes as bnb


# Collection of constants used for all tuners
COMPATIBLE_TUNER_TYPES = (PeftType.LORA, PeftType.LOHA, PeftType.LOKR, PeftType.ADALORA)
PREFIXES = [lora.LoraModel.prefix, lokr.LoKrModel.prefix, loha.LoHaModel.prefix]
Configs = Union[lora.LoraConfig, loha.LoHaConfig, lokr.LoKrConfig, adalora.AdaLoraConfig]
Layers = (lora.layer.LoraLayer, loha.layer.LoHaLayer, lokr.layer.LoKrLayer, adalora.layer.AdaLoraLayer)


class LycorisModel(BaseTuner):
    """TODO"""

    def __init__(self, model: nn.Module, config: Configs, adapter_name: str) -> None:
        super().__init__(model, config, adapter_name)

    def _check_new_adapter_config(self, config: Configs) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        if not isinstance(config, Configs.__args__):
            raise ValueError(
                f"{self.__class__.__name__} only supports {COMPATIBLE_TUNER_TYPES} configs, but got {type(config)}."
            )

        biases = (getattr(config, "bias", None) for config in self.peft_config)
        biases = [bias for bias in biases if bias not in (None, "none")]
        if len(biases) > 1:
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_module_exists(config: Configs, key: str):
        return check_target_module_exists(config, key)

    def _create_and_replace(
        self,
        config: Configs,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if isinstance(config, adalora.AdaLoraConfig):
            adalora.AdaLoraModel._create_and_replace(self, config, *args, **kwargs)
        elif isinstance(config, lora.LoraConfig):
            lora.LoraModel._create_and_replace(self, config, *args, **kwargs)
        elif isinstance(config, loha.LoHaConfig):
            loha.LoHaModel._create_and_replace(self, config, *args, **kwargs)
        elif isinstance(config, lokr.LoKrConfig):
            lokr.LoKrModel._create_and_replace(self, config, *args, **kwargs)
        else:
            raise ValueError(f"Unsupporte config type {type(config)}, should be one of {COMPATIBLE_TUNER_TYPES}.")

    def _replace_module(self, parent, child_name, new_module, child) -> None:
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.get_base_layer()
        elif hasattr(child, "quant_linear_module"):
            # TODO maybe not necessary to have special treatment?
            child = child.quant_linear_module

        # TODO: layers with base_layer don't need the weight to be copied, as they have a reference already
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

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if any(prefix in name for prefix in PREFIXES):
                module.to(child.weight.device)
            if "ranknum" in name:
                module.to(child.weight.device)

    def _mark_only_adapters_as_trainable(self) -> None:
        for n, p in self.model.named_parameters():
            if not any(prefix in n for prefix in PREFIXES):
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = getattr(self.peft_config[active_adapter], "bias", "none")
            if bias == "none":
                continue

            if bias == "all":
                for n, p in self.model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "lora_only":
                # TODO: check if this is needed for other supported types
                for m in self.model.modules():
                    if isinstance(m, Layers) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise ValueError(f"Requested bias: {bias}, is not implemented.")

    @staticmethod
    def _create_new_module(config, adapter_name, target, **kwargs):
        gptq_quantization_config = kwargs.get("gptq_quantization_config", None)
        AutoGPTQQuantLinear = get_auto_gptq_quant_linear(gptq_quantization_config)
        if (gptq_quantization_config is not None) or (AutoGPTQQuantLinear is not None):
            raise ValueError("GPTQ quantization not supported for LycorisModel yet")

        loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)
        if loaded_in_8bit or loaded_in_4bit:
            raise ValueError("8bit and 4bit quantization not supported for LycorisModel yet")

        if isinstance(config, adalora.AdaLoraConfig):
            new_module = adalora.AdaLoraModel._create_new_module(config, adapter_name, target, **kwargs)
        elif isinstance(config, lora.LoraConfig):
            new_module = lora.LoraModel._create_new_module(config, adapter_name, target, **kwargs)
        elif isinstance(config, loha.LoHaConfig):
            new_module = loha.LoHaModel._create_new_module(config, adapter_name, target, **kwargs)
        elif isinstance(config, lokr.LoKrConfig):
            new_module = lokr.LoKrModel._create_new_module(config, adapter_name, target, **kwargs)
        else:
            raise ValueError(f"Unknown config type {type(config)}, should be one of {COMPATIBLE_TUNER_TYPES}.")
        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        for active_adapter in self.active_adapters:
            val = getattr(self.peft_config[active_adapter], "bias", "none")
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name: Union[str, list[str]]) -> None:
        for module in self.model.modules():
            if isinstance(module, Layers):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _unload_and_optionally_merge(self, merge=True, progressbar: bool = False, safe_merge: bool = False):
        if merge:
            if getattr(self.model, "quantization_method", None) == "gptq":
                raise ValueError("Cannot merge layers when the model is gptq quantized")

        key_list = [key for key, _ in self.model.named_modules() if not any(prefix in key for prefix in PREFIXES)]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue

            if isinstance(target, Layers):
                target_base_layer = target.get_base_layer()
                # Careful: Below, we need to check *all* possible layers to be merged, i.e. the union of all supported
                # layers for all tuner methods.

                # check bnb layers first because they can be instances of both bnb layers and normal PyTorch layers and
                # we want to match the former, not the latter
                if is_bnb_available() and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
                    bias = target_base_layer.bias is not None
                    new_module = bnb.nn.Linear8bitLt(
                        target.in_features,
                        target.out_features,
                        bias=bias,
                        has_fp16_weights=target_base_layer.state.has_fp16_weights,
                        memory_efficient_backward=target_base_layer.state.memory_efficient_backward,
                        threshold=target_base_layer.state.threshold,
                        index=target_base_layer.index,
                        device=target_base_layer.weight.device,
                    )
                elif is_bnb_4bit_available() and isinstance(target_base_layer, bnb.nn.Linear4bit):
                    bias = target_base_layer.bias is not None
                    new_module = bnb.nn.Linear4bit(
                        target.in_features,
                        target.out_features,
                        bias=bias,
                        compute_dtype=target_base_layer.compute_dtype,
                        compress_statistics=target_base_layer.weight.compress_statistics,
                        quant_type=target_base_layer.weight.quant_type,
                        device=target_base_layer.weight.device,
                    )
                elif isinstance(target_base_layer, nn.Embedding):
                    new_module = nn.Embedding(
                        target_base_layer.num_embeddings,
                        target_base_layer.embedding_dim,
                        padding_idx=target_base_layer.padding_idx,
                        max_norm=target_base_layer.max_norm,
                        norm_type=target_base_layer.norm_type,
                        scale_grad_by_freq=target_base_layer.scale_grad_by_freq,
                        sparse=target_base_layer.sparse,
                    )
                elif isinstance(target_base_layer, nn.Conv2d):
                    new_module = nn.Conv2d(
                        target_base_layer.in_channels,
                        target_base_layer.out_channels,
                        kernel_size=target_base_layer.kernel_size,
                        stride=target_base_layer.stride,
                        padding=target_base_layer.padding,
                        dilation=target_base_layer.dilation,
                    )
                elif isinstance(target_base_layer, (nn.Linear, Conv1D)):
                    # TODO: checking Linear and Conv1D separately, could we get rid of is_target_conv_1d_layer?
                    bias = target_base_layer.bias is not None
                    if getattr(target, "is_target_conv_1d_layer", False):
                        in_features, out_features = (
                            target_base_layer.weight.ds_shape
                            if hasattr(target_base_layer.weight, "ds_shape")
                            else target_base_layer.weight.shape
                        )
                        new_module = Conv1D(out_features, in_features)
                    else:
                        new_module = nn.Linear(
                            target_base_layer.in_features, target_base_layer.out_features, bias=bias
                        )
                else:
                    raise ValueError(f"Unsupported layer type: {type(target_base_layer)}")

                if merge:
                    # buckle in: here we recursively merge the base_layer of the target, starting from the lowest level
                    path = []
                    layer = target
                    while hasattr(layer, "base_layer"):
                        path.append(layer)
                        layer = layer.base_layer
                    for layer_before, layer_after in zip(path[:-1], path[1:]):
                        layer_after.merge(safe_merge=safe_merge)
                        layer_before.base_layer = layer_after.base_layer

                    target.merge(safe_merge=safe_merge)
                    self._replace_module(parent, target_name, new_module, target)
                else:
                    self._replace_module(parent, target_name, new_module, target.get_base_layer())
            # save any additional trainable modules part of `modules_to_save`
            elif isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def add_weighted_adapter(self, *args, **kwargs):
        raise NotImplementedError("Weighted adapters are not supported yet for LycorisModel yet")

    def delete_adapter(self, adapter_name: str):
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in self.peft_config:
            raise KeyError(f"Adapter {adapter_name} does not exist")

        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if not any(prefix in key for prefix in PREFIXES)]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, lora.layer.LoraLayer):
                for attr in [
                    "r",
                    "lora_alpha",
                    "scaling",
                    "lora_A",
                    "lora_B",
                    "lora_embedding_A",
                    "lora_embedding_B",
                    "lora_dropout",
                ]:
                    if adapter_name in getattr(target, attr):
                        getattr(target, attr).pop(adapter_name)
                if adapter_name in target.active_adapters:
                    resetting_active_adapter = (
                        list(self.peft_config.keys())[0] if len(self.peft_config) > 0 else "default"
                    )
                    warnings.warn(
                        f"Adapter {adapter_name} was active which is now deleted. Setting active adapter to {resetting_active_adapter}. "
                    )
                    target.set_adapter(resetting_active_adapter)
            elif isinstance(target, loha.layer.LoHaLayer):
                raise ValueError(f"Deleting LoHa layers are not supported yet for {self.__class__.__name__} yet")

    def merge_and_unload(self, progressbar: bool = False, safe_merge: bool = False):
        r"""
        This method merges the layers into the base model. This is needed if someone wants to use the base model as a
        standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights

        Example:

        TODO adjust example

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(progressbar=progressbar, safe_merge=safe_merge)

    def unload(self):
        """
        Gets back the base model by removing all the lora modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)
