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
import math
import operator
import re
import warnings
from dataclasses import asdict, replace
from enum import Enum
from functools import partial, reduce
from itertools import chain
from typing import Union

import torch
from torch import nn
from torch.nn.init import _calculate_correct_fan
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
    TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
)

from .config import VeraConfig
from .layer import Conv2d, Embedding, Linear, VeraLayer


def _vera_forward_hook(module, args, kwargs, vera_A, vera_B):
    kwargs["vera_A"] = vera_A
    kwargs["vera_B"] = vera_B
    return args, kwargs


def _kaiming_init(
    tensor_or_shape: Union[torch.Tensor, tuple],
    generator: torch.Generator,
):
    if isinstance(tensor_or_shape, tuple):
        tensor = torch.empty(tensor_or_shape)
    else:
        tensor = tensor_or_shape
    fan = _calculate_correct_fan(tensor, "fan_in")
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std

    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=generator)


class VeraModel(BaseTuner):
    """
    Creates Vector-based Random Matrix Adaptation (Vera) model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`VeraConfig`]): The configuration of the Vera model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The Vera model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import VeraModel, VeraConfig

        >>> config = VeraConfig(
        ...     task_type="SEQ_2_SEQ_LM",
        ...     r=8,
        ...     target_modules=["q", "v"],
        ...     vera_dropout=0.01,
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> vera_model = VeraModel(model, config, "default")
        ```

        ```py
        >>> import transformers
        >>> from peft import VeraConfig, PeftModel, get_peft_model

        >>> target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        >>> config = VeraConfig(
        ...     r=4, target_modules=target_modules, vera_dropout=0.1, bias="none", task_type="CAUSAL_LM"
        ... )

        >>> model = transformers.GPTJForCausalLM.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     pad_token_id=tokenizer.eos_token_id,
        ...     use_cache=False,
        ...     device_map={"": rank},
        ...     torch_dtype=torch.float16,
        ...     load_in_8bit=True,
        ... )
        >>> model = prepare_model_for_int8_training(model)
        >>> vera_model = get_peft_model(model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`VeraConfig`]): The configuration of the Vera model.
    """

    def _find_first_dim(self) -> int:
        first_linear, first_embedding = None, None
        for module in self.model.modules():
            if isinstance(module, Linear):
                module_shape = tuple(module.weight.shape)
                if module.fan_in_fan_out:
                    module_shape = module_shape[::-1]

                if first_linear is not None and module_shape != first_linear:
                    raise ValueError(
                        "Multiple target linear layers with different dimensions were specified! Vera only supports a single dimension size."
                    )
                first_linear = module_shape


            elif isinstance(module, Embedding):
                if first_embedding is not None and tuple(module.weight.shape) != first_embedding:
                    raise ValueError(
                        "Multiple target embedding layers with different dimensions or vocabulary sizes were specified! Vera only supports a single size."
                    )
                first_embedding = tuple(module.weight.shape)

            if first_linear is not None and first_embedding is not None:
                return first_linear, first_embedding

        return first_linear, first_embedding

    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)
        config = config[adapter_name]
        generator = torch.Generator(device="cpu").manual_seed(config.projection_prng_key)

        first_linear, first_embedding = self._find_first_dim()

        if first_embedding is not None:
            first_embedding_vocab_size, first_embedding_dim = first_embedding
        if first_linear is not None:
            first_linear_out_dim, first_linear_in_dim = first_linear

        if first_linear is not None:
            vera_A = _kaiming_init((config.r, first_linear_in_dim), generator=generator)
            vera_B = _kaiming_init((first_linear_out_dim, config.r), generator=generator)

            self.register_buffer("vera_A", vera_A, persistent=config.save_projection)
            self.register_buffer("vera_B", vera_B, persistent=config.save_projection)
        else:
            self.vera_A = None
            self.vera_B = None

        if first_embedding is not None:
            vera_embedding_A = torch.randn((config.r, first_embedding_vocab_size), generator=generator)
            vera_embedding_B = torch.randn((first_embedding_dim, config.r), generator=generator)
            self.register_buffer("vera_embedding_A", vera_embedding_A, persistent=config.save_projection)
            self.register_buffer("vera_embedding_B", vera_embedding_B, persistent=config.save_projection)
        else:
            self.vera_embedding_A = None
            self.vera_embedding_B = None

        if not config.save_projection:
            warnings.warn(
                "Specified to not save vera_A and vera_B within the state dictionary, instead they will be restored using the PRNG key store in `config.projection_prng_key`. Consider setting `config.save_projection` to `True` to guarantee restoring the checkpoint correctly on all system configurations."
            )

    def _check_new_adapter_config(self, config: VeraConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        # the below todo is copied from LoRA
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not fully correspond to the error message.
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

        # TODO: check this in conjunction with save_projection
        if config.projection_prng_key is None:
            raise ValueError("Vera PRNG initialisation key cannot be `None`. Set `VeraConfig.projection_prng_key`.")

    @staticmethod
    def _check_target_module_exists(vera_config, key):
        return check_target_module_exists(vera_config, key)

    def _create_and_replace(
        self,
        vera_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(vera_config.rank_pattern.keys(), vera_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(f".*\.{key}$", current_key), pattern_keys), current_key)

        r = vera_config.rank_pattern.get(target_name_key, vera_config.r)
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": r,
            "vera_dropout": vera_config.vera_dropout,
            "fan_in_fan_out": vera_config.fan_in_fan_out,
            "init_vera_weights": vera_config.init_vera_weights,
        }

        # TODO: add back once we have quant support
        # kwargs["loaded_in_8bit"] = False
        # kwargs["loaded_in_4bit"] = False
        kwargs["bias"] = bias

        # TODO: add in quant?
        # quantization_config = get_quantization_config(self.model, method="gptq")
        # if quantization_config is not None:
        # kwargs["gptq_quantization_config"] = quantization_config

        # the below todo is copied from LoRA
        # TODO: better deal with that
        if isinstance(target, VeraLayer) and isinstance(target, torch.nn.Conv2d):
            target.update_layer_conv2d(
                adapter_name,
                r,
                vera_config.vera_dropout,
                vera_config.init_vera_weights,
                d_initial=vera_config.d_initial,
            )
        elif isinstance(target, VeraLayer) and isinstance(target, (torch.nn.Embedding, torch.nn.sparse.Embedding)):
            target.update_layer_embedding(
                adapter_name,
                r,
                vera_config.vera_dropout,
                vera_config.init_vera_weights,
                d_initial=vera_config.d_initial,
            )

        elif isinstance(target, VeraLayer):
            target.update_layer(
                adapter_name,
                r,
                vera_config.vera_dropout,
                vera_config.init_vera_weights,
                d_initial=vera_config.d_initial,
            )
        else:
            new_module = self._create_new_module(vera_config, adapter_name, target, **kwargs)
            if adapter_name != self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        # the below todo is copied from LoRA
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
            if "vera_" in name:
                module.to(child.weight.device)
            if "ranknum" in name:
                module.to(child.weight.device)

    def _mark_only_adapters_as_trainable(self) -> None:
        for n, p in self.model.named_parameters():
            if "vera_lambda_" not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in self.model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "vera_only":
                for m in self.model.modules():
                    if isinstance(m, VeraLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    @staticmethod
    def _create_new_module(vera_config, adapter_name, target, **kwargs):
        bias = kwargs.pop("bias", False)

        if isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = Embedding(
                adapter_name,
                in_features,
                out_features,
                d_initial=vera_config.d_initial,
                **embedding_kwargs,
            )
        elif isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(
                adapter_name,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                d_initial=vera_config.d_initial,
                **kwargs,
            )
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = vera_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = vera_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. Currently, only the following modules are supported: "
                    "`torch.nn.Linear`, `torch.nn.Embedding`, `transformers.pytorch_utils.Conv1D`."
                )
            new_module = Linear(
                adapter_name,
                in_features,
                out_features,
                bias=bias,
                d_initial=vera_config.d_initial,
                **kwargs,
            )

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
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, VeraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    if isinstance(module, Linear):
                        module.unmerge(self.vera_A, self.vera_B)
                    elif isinstance(module, Embedding):
                        module.unmerge(self.vera_embedding_A, self.vera_embedding_B)
                module.set_adapter(adapter_name)

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _unload_and_optionally_merge(self, merge=True, progressbar: bool = False, safe_merge: bool = False):
        key_list = [key for key, _ in self.model.named_modules() if "vera" not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, VeraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                elif isinstance(target, nn.Conv2d):
                    new_module = torch.nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                else:
                    bias = target.bias is not None
                    if getattr(target, "is_target_conv_1d_layer", False):
                        new_module = Conv1D(target.out_features, target.in_features)
                    else:
                        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                if merge:
                    if isinstance(target, Linear):
                        target.merge(self.vera_A, self.vera_B, safe_merge=safe_merge)
                    elif isinstance(target, Embedding):
                        target.merge(self.vera_embedding_A, self.vera_embedding_B, safe_merge=safe_merge)
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def add_weighted_adapter(
        self,
        adapters,
        weights,
        adapter_name,
        combination_type="svd",
        svd_rank=None,
        svd_clamp=None,
        svd_full_matrices=True,
        svd_driver=None,
    ):

        # TODO: implement this for Vera
        raise NotImplementedError("Adapter merging is not supported yet for VeRA.")
        """
        This method adds a new adapter by merging the given adapters with the given weights.

        When using the `cat` combination_type you should be aware that rank of the resulting adapter will be equal to
        the sum of all adapters ranks. So it's possible that the mixed adapter may become too big and result in OOM
        errors.

        Args:
            adapters (`list`):
                List of adapter names to be merged.
            weights (`list`):
                List of weights for each adapter.
            adapter_name (`str`):
                Name of the new adapter.
            combination_type (`str`):
                Type of merging. Can be one of [`svd`, `linear`, `cat`]. When using the `cat` combination_type you
                should be aware that rank of the resulting adapter will be equal to the sum of all adapters ranks. So
                it's possible that the mixed adapter may become too big and result in OOM errors.
            svd_rank (`int`, *optional*):
                Rank of output adapter for svd. If None provided, will use max rank of merging adapters.
            svd_clamp (`float`, *optional*):
                A quantile threshold for clamping SVD decomposition output. If None is provided, do not perform
                clamping. Defaults to None.
            svd_full_matrices (`bool`, *optional*):
                Controls whether to compute the full or reduced SVD, and consequently, the shape of the returned
                tensors U and Vh. Defaults to True.
            svd_driver (`str`, *optional*):
                Name of the cuSOLVER method to be used. This keyword argument only works when merging on CUDA. Can be
                one of [None, `gesvd`, `gesvdj`, `gesvda`]. For more info please refer to `torch.linalg.svd`
                documentation. Defaults to None.
        """

        if adapter_name in list(self.peft_config.keys()):
            return
        for adapter in adapters:
            if adapter not in list(self.peft_config.keys()):
                raise ValueError(f"Adapter {adapter} does not exist")

        # if there is only one adapter, we can only use linear merging
        combination_type = "linear" if len(adapters) == 1 else combination_type

        adapters_ranks = [self.peft_config[adapter].r for adapter in adapters]
        if combination_type == "linear":
            # all adapters ranks should be same, new rank is just this value
            if len(set(adapters_ranks)) != 1:
                raise ValueError("All adapters must have the same r value when using `linear` combination_type")
            new_rank = adapters_ranks[0]
        elif combination_type == "cat":
            # adapters ranks may be different, new rank is sum of all ranks
            # be careful, because output adapter rank may be really big if mixing a lot of adapters
            new_rank = sum(adapters_ranks)
        elif combination_type == "svd":
            # new rank is the max of all ranks of the adapters if not provided
            new_rank = svd_rank or max(adapters_ranks)
        else:
            raise ValueError(f"Invalid combination_type: {combination_type}")

        target_module_types = [type(self.peft_config[adapter].target_modules) for adapter in adapters]
        if not target_module_types:
            raise ValueError(f"Found no adapter matching the names in {adapters}")
        if len(set(target_module_types)) > 1:
            raise ValueError(
                "all adapter configs should follow the same target modules type. "
                "Combining adapters with `target_modules` type being a mix of list/set and string is not supported."
            )

        if target_module_types[0] == str:
            new_target_modules = "|".join(f"({self.peft_config[adapter].target_modules})" for adapter in adapters)
        elif target_module_types[0] == set:
            new_target_modules = reduce(
                operator.or_, (self.peft_config[adapter].target_modules for adapter in adapters)
            )
        else:
            raise TypeError(f"Invalid type {target_module_types[0]} found in target_modules")

        self.peft_config[adapter_name] = replace(
            self.peft_config[adapters[0]],
            r=new_rank,
            target_modules=new_target_modules,
        )
        self.inject_adapter(self.model, adapter_name)

        # Do we really need that?
        _freeze_adapter(self.model, adapter_name)

        key_list = [key for key, _ in self.model.named_modules() if "vera" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, VeraLayer):
                if adapter_name in target.vera_A:
                    target_lora_A = target.lora_A[adapter_name].weight
                    target_lora_B = target.lora_B[adapter_name].weight
                elif adapter_name in target.lora_embedding_A:
                    target_lora_A = target.lora_embedding_A[adapter_name]
                    target_lora_B = target.lora_embedding_B[adapter_name]
                else:
                    continue

                target_lora_A.data = target_lora_A.data * 0.0
                target_lora_B.data = target_lora_B.data * 0.0
                if combination_type == "linear":
                    for adapter, weight in zip(adapters, weights):
                        if adapter in target.lora_A:
                            current_adapter_lora_A = target.lora_A[adapter].weight
                            current_adapter_lora_B = target.lora_B[adapter].weight
                        elif adapter in target.lora_embedding_A:
                            current_adapter_lora_A = target.lora_embedding_A[adapter]
                            current_adapter_lora_B = target.lora_embedding_B[adapter]
                        else:
                            continue
                        target_lora_A.data += current_adapter_lora_A.data * weight
                        target_lora_B.data += current_adapter_lora_B.data
                elif combination_type == "cat":
                    loras_A, loras_B = [], []
                    for adapter, weight in zip(adapters, weights):
                        if adapter in target.lora_A:
                            current_adapter_lora_A = target.lora_A[adapter].weight
                            current_adapter_lora_B = target.lora_B[adapter].weight
                        elif adapter in target.lora_embedding_A:
                            current_adapter_lora_A = target.lora_embedding_A[adapter]
                            current_adapter_lora_B = target.lora_embedding_B[adapter]
                        else:
                            continue
                        loras_A.append(current_adapter_lora_A.data * weight)
                        loras_B.append(current_adapter_lora_B.data)

                    if len(loras_A) == 0:
                        raise ValueError("No matching LoRAs found. Please raise an issue on Github.")
                    loras_A = torch.cat(loras_A, dim=0)
                    loras_B = torch.cat(loras_B, dim=1)
                    target_lora_A.data[: loras_A.shape[0], :] = loras_A
                    target_lora_B.data[:, : loras_B.shape[1]] = loras_B
                elif combination_type == "svd":
                    target_lora_A.data, target_lora_B.data = self._svd_weighted_adapter(
                        adapters,
                        weights,
                        new_rank,
                        target,
                        target_lora_A,
                        target_lora_B,
                        svd_clamp,
                        full_matrices=svd_full_matrices,
                        driver=svd_driver,
                    )

    def _svd_weighted_adapter(
        self,
        adapters,
        weights,
        new_rank,
        target,
        target_lora_A,
        target_lora_B,
        clamp=None,
        full_matrices=True,
        driver=None,
    ):
        raise NotImplementedError("SVD adapter merging is not supported yet for VeRA.")
        valid_adapters = []
        valid_weights = []
        for adapter, weight in zip(adapters, weights):
            if adapter in target.lora_A or adapter in target.lora_embedding_A:
                valid_adapters.append(adapter)
                valid_weights.append(weight)

        # if no valid adapter, nothing to do
        if len(valid_adapters) == 0:
            raise ValueError("No matching LoRAs found. Please raise an issue on Github.")

        delta_weight = valid_weights[0] * target.get_delta_weight(valid_adapters[0])
        for adapter, weight in zip(valid_adapters[1:], valid_weights[1:]):
            delta_weight += weight * target.get_delta_weight(adapter)
        conv2d = isinstance(target, Conv2d)
        if conv2d:
            conv2d_1x1 = target.weight.size()[2:4] == (1, 1)
            if not conv2d_1x1:
                delta_weight = delta_weight.flatten(start_dim=1)
            else:
                delta_weight = delta_weight.squeeze()
        if hasattr(target, "fan_in_fan_out") and target.fan_in_fan_out:
            delta_weight = delta_weight.T

        # based on https://github.com/kohya-ss/sd-scripts/blob/main/networks/svd_merge_lora.py#L114-L131
        U, S, Vh = torch.linalg.svd(delta_weight, full_matrices=full_matrices, driver=driver)
        U = U[:, :new_rank]
        S = S[:new_rank]
        U = U @ torch.diag(S)
        Vh = Vh[:new_rank, :]
        if clamp is not None:
            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, clamp)
            low_val = -hi_val
            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)
        if conv2d:
            U = U.reshape(target_lora_B.data.shape)
            Vh = Vh.reshape(target_lora_A.data.shape)
        return Vh, U

    def delete_adapter(self, adapter_name: str):
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if "vera" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, VeraLayer):
                for attr in [
                    "r",
                    "vera_A",
                    "vera_B",
                    "vera_lambda_b",
                    "vera_lambda_d",
                    "vera_embedding_A",
                    "vera_embedding_B",
                    "vera_dropout",
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

    def merge_and_unload(self, progressbar: bool = False, safe_merge: bool = False):
        r"""
        This method merges the Vera layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights

        Example:

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
        Gets back the base model by removing all the Vera modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)

    def _add_forward_hooks(self):
        hook_handles = []
        for module in self.modules():
            if isinstance(module, Linear):
                pre_forward = partial(_vera_forward_hook, vera_A=self.vera_A, vera_B=self.vera_B)
                handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                hook_handles.append(handle)

            elif isinstance(module, Embedding):
                pre_forward = partial(_vera_forward_hook, vera_A=self.vera_embedding_A, vera_B=self.vera_embedding_B)
                handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                hook_handles.append(handle)

        return hook_handles

    def forward(self, *args, **kwargs):
        hook_handles = self._add_forward_hooks()
        try:
            outputs = super().forward(*args, **kwargs)
        finally:
            for handle in hook_handles:
                handle.remove()

        return outputs