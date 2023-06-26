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
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..import_utils import is_bnb_available
from ..utils import (
    TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftConfig,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)


if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class IA3Config(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`IA3Model`].

    Args:
        target_modules (`Union[List[str],str]`): The names of the modules to apply IA3 to.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.:
        modules_to_save (`List[str]`):List of modules apart from IA3 layers to be set as trainable
            and saved in the final checkpoint.
    """

    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with ia3."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    feedforward_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or a regex expression of module names which are feedforward"
            "For example, ['output.dense']"
        },
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    # bias: str = field(default="none", metadata={"help": "Bias type for ia3. Can be 'none', 'all' or 'ia3_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from ia3 layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_ia3_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the vectors in the IA3 layers."},
    )

    def __post_init__(self):
        self.peft_type = PeftType.IA3


class IA3Model(torch.nn.Module):
    """
    Creates IA3 model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`IA3Config`]): The configuration of the ia3 model.

    Returns:
        `torch.nn.Module`: The ia3 model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM, ia3Config
        >>> from peft import IA3Model, IA3Config

        >>> config = IA3Config(
        ...     peft_type="IA3",
        ...     task_type="SEQ_2_SEQ_LM",
        ...     target_modules=["k", "v"],
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> ia3_model = IA3Model(config, model)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`ia3Config`]): The configuration of the ia3 model.
    """

    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_ia3_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        # if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
        #     raise ValueError(
        #         "IA3 supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
        #     )
        mark_only_ia3_as_trainable(self.model, bias="none")
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name):
        ia3_config = self.peft_config[adapter_name]
        if not ia3_config.feedforward_modules:
            ia3_config.feedforward_modules = []
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use ia3 with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False

        kwargs = {
            "fan_in_fan_out": ia3_config.fan_in_fan_out,
            "init_ia3_weights": ia3_config.init_ia3_weights,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(ia3_config.target_modules, str):
                target_module_found = re.fullmatch(ia3_config.target_modules, key)
            else:
                target_module_found = any(
                    self._is_valid_match(key, target_key) for target_key in ia3_config.target_modules
                )
            if target_module_found:
                if isinstance(ia3_config.feedforward_modules, str):
                    is_feedforward = re.fullmatch(ia3_config.feedforward_modules, key)
                else:
                    is_feedforward = any(key.endswith(target_key) for target_key in ia3_config.feedforward_modules)
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                # import pdb; pdb.set_trace()
                parent, target, target_name = _get_submodules(self.model, key)
                bias = target.bias is not None
                if isinstance(target, IA3Layer):
                    target.update_layer(
                        adapter_name,
                        ia3_config.init_ia3_weights,
                    )
                else:
                    if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                        eightbit_kwargs = kwargs.copy()
                        eightbit_kwargs.update(
                            {
                                "has_fp16_weights": target.state.has_fp16_weights,
                                "memory_efficient_backward": target.state.memory_efficient_backward,
                                "threshold": target.state.threshold,
                                "index": target.index,
                            }
                        )
                        new_module = Linear8bitLt(
                            adapter_name,
                            target.in_features,
                            target.out_features,
                            is_feedforward,
                            bias=bias,
                            **eightbit_kwargs,
                        )
                    else:
                        if isinstance(target, torch.nn.Linear):
                            in_features, out_features = target.in_features, target.out_features
                            if kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                                    "Setting fan_in_fan_out to False."
                                )
                                kwargs["fan_in_fan_out"] = ia3_config.fan_in_fan_out = False
                        elif isinstance(target, Conv1D):
                            in_features, out_features = (
                                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                            )
                            if not kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                                    "Setting fan_in_fan_out to True."
                                )
                                kwargs["fan_in_fan_out"] = ia3_config.fan_in_fan_out = True
                        else:
                            raise ValueError(
                                f"Target module {target} is not supported. "
                                f"Currently, only `torch.nn.Linear` are supported."
                            )
                        new_module = Linear(
                            adapter_name, in_features, out_features, is_feedforward=is_feedforward, bias=bias, **kwargs
                        )

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {ia3_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    @staticmethod
    def _is_valid_match(key: str, target_key: str):
        """
        Helper function to match module names target_key and key. Makes sure that either the key is exactly the
        target_key or the target_key is a submodule of key
        """
        if key.endswith(target_key):
            if len(key) > len(target_key):
                return key.endswith("." + target_key)  # must be a sub module
            return True
        return False

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "ia3_" in name:
                module.to(old_module.weight.device)

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
            if isinstance(module, IA3Layer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, IA3Layer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, IA3Layer):
                module.merge()

    def unmerge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, IA3Layer):
                module.unmerge()

    @staticmethod
    def _prepare_ia3_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING[model_config["model_type"]]
        if peft_config.feedforward_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING:
                raise ValueError("Please specify `feedforward_modules` in `peft_config`")
            peft_config.feedforward_modules = TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING[
                model_config["model_type"]
            ]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def merge_and_unload(self):
        r"""
        This method merges the ia3 layers into the base model. This is needed if someone wants to use the base model as
        a standalone model.
        """
        if getattr(self.config, "model_type", None) == "gpt2":
            raise ValueError("GPT2 models are not supported for merging ia3 layers")

        if getattr(self.model, "is_loaded_in_8bit", False):
            raise ValueError("Cannot merge ia3 layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "ia3" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, IA3Layer):
                bias = target.bias is not None
                new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model


# Below code is based on https://github.com/microsoft/lora/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


def mark_only_ia3_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "ia3_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    else:
        raise NotImplementedError


class IA3Layer:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_feedforward: bool,
    ):
        self.scaling = {}
        self.ia3_l = nn.ParameterDict({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.is_feedforward = is_feedforward

    def update_layer(self, adapter_name, init_ia3_weights):
        # Actual trainable parameters
        if self.is_feedforward:
            self.ia3_l.update(nn.ParameterDict({adapter_name: nn.Parameter(torch.ones(1, self.in_features))}))
        else:
            self.ia3_l.update(nn.ParameterDict({adapter_name: nn.Parameter(torch.ones(self.out_features, 1))}))
        if init_ia3_weights:
            self.reset_ia3_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_ia3_parameters(self, adapter_name):
        if adapter_name in self.ia3_l.keys():
            # initialize learned vector with torch.ones
            nn.init.constant_(self.ia3_l[adapter_name], 1.0)


class Linear(nn.Linear, IA3Layer):
    # IA3 implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_feedforward: bool = False,  # Set to True if the layer is a feedforward layer
        **kwargs,
    ):
        init_ia3_weights = kwargs.pop("init_ia3_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        IA3Layer.__init__(self, in_features=in_features, out_features=out_features, is_feedforward=is_feedforward)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, init_ia3_weights)
        self.active_adapter = adapter_name

        self.is_feedforward = is_feedforward

    def merge(self):

        if self.active_adapter not in self.ia3_l.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return

        self.weight = transpose(self.weight, self.fan_in_fan_out)
        self.weight.data = torch.mul(self.weight.data, self.ia3_l[self.active_adapter].data)
        self.weight = transpose(self.weight, self.fan_in_fan_out)

        self.merged = True

    def unmerge(self):

        if self.active_adapter not in self.ia3_l.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        warnings.warn("Unmerge result can be inaccurate for IA3.")
        self.weight = transpose(self.weight, self.fan_in_fan_out)
        self.weight.data = torch.div(self.weight.data, self.ia3_l[self.active_adapter].data)
        self.weight = transpose(self.weight, self.fan_in_fan_out)

        self.merged = False

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        if self.active_adapter not in self.ia3_l.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if not self.merged:
            if self.is_feedforward:
                result = F.linear(
                    x * self.ia3_l[self.active_adapter].flatten(),
                    transpose(self.weight, self.fan_in_fan_out),
                    bias=self.bias,
                )
            else:
                result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
                result = result * self.ia3_l[self.active_adapter].flatten()

        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result


if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, IA3Layer):
        # ia3 implemented in a dense layer
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            is_feedforward,
            **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            IA3Layer.__init__(self, in_features=in_features, out_features=out_features, is_feedforward=is_feedforward)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_ia3_weights = kwargs.pop("init_ia3_weights", True)
            self.update_layer(adapter_name, init_ia3_weights)
            self.active_adapter = adapter_name
            self.is_feedforward = is_feedforward

        def forward(self, x: torch.Tensor):
            if self.disable_adapters or self.active_adapter not in self.ia3_l.keys():
                return super().forward(x)
            else:
                if not torch.is_autocast_enabled():

                    if x.dtype != torch.float32:
                        x = x.float()
                    if self.is_feedforward:
                        result = super().forward(x * self.ia3_l[self.active_adapter].flatten())
                    else:
                        result = super().forward(x)
                        expected_dtype = result.dtype
                        result = (result * self.ia3_l[self.active_adapter].flatten()).to(expected_dtype)
                else:
                    if self.is_feedforward:
                        result = super().forward(x * self.ia3_l[self.active_adapter].flatten())
                    else:
                        result = result * self.ia3_l[self.active_adapter].flatten()
            return result
