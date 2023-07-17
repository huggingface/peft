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
import re
import warnings
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import List, Optional, Tuple, Union
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PeftConfig
from ..import_utils import is_bnb_4bit_available, is_bnb_available
from ..utils import (
    TRANSFORMERS_MODELS_TO_ADAMIX_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftType,
    _get_submodules,
    transpose,
)

if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class AdaMixConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`AdaMixModel`].

    Args:
        target_modules (`Union[List[str],str]`): The names of the modules to apply AdaMix to.
        adapter_dim (`int`): The hidden dim of the adapter (r in the paper). The downsampling adapter has shape dxr and the upsampling adapter has shape rxd where d is the hidden_dim of the model
        num_expert (`int`): The number of exprts per adapter module
        sharing_down (`bool`): If the weights of the downsampling adapters are shared in each layer
        sharing_up (`bool`): If the weights of the upsampling adapters are shared in each layer
        return_two_views (`bool`): If two stochastic forward passes have to be made, and both outputs returned
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.:
        modules_to_save (`List[str]`):List of modules apart from (IA)^3 layers to be set as trainable
            and saved in the final checkpoint.
    """

    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with ia3."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    adapter_dim: Optional[int] = field(
        default=16,
        metadata={
            "help": "The hidden dim of the adapter (r in the paper). The downsampling adapter has shape dxr and the upsampling adapter has shape rxd where d is the hidden_dim of the model "
        },
    )
    num_expert: Optional[int] = field(
        default=4,
        metadata={
            "help": "The number of exprts per adapter module"
        },
    )
    sharing_down: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If the weights of the downsampling adapters are shared in each layer"
        },
    )
    sharing_up: Optional[bool] = field(
        default=True,
        metadata={
            "help": "If the weights of the upsampling adapters are shared in each layer"
        },
    )
    return_two_views: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If two stochastic forward passes have to be made"
        },
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from (IA)^3 layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.ADAMIX


class AdaMixModel(torch.nn.Module):
    # TODO: Modify description
    """
    Creates a Infused Adapter by Inhibiting and Amplifying Inner Activations ((IA)^3) model from a pretrained
    transformers model. The method is described in detail in https://arxiv.org/abs/2205.05638

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`IA3Config`]): The configuration of the (IA)^3 model.

    Returns:
        `torch.nn.Module`: The (IA)^3 model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM, ia3Config
        >>> from peft import IA3Model, IA3Config

        >>> config = IA3Config(
        ...     peft_type="IA3",
        ...     task_type="SEQ_2_SEQ_LM",
        ...     target_modules=["k", "v", "w0"],
        ...     feedforward_modules=["w0"],
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> ia3_model = IA3Model(config, model)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`ia3Config`]): The configuration of the (IA)^3 model.
    """

    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_adamix_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)

        mark_only_adamix_as_trainable(self.model)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _check_quantization_dependency(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        if loaded_in_4bit:
            raise NotImplementedError(
                "4-bit quantization is not supported for AdaMix yet, 8-bit quantization can be used instead."
            )
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use AdaMix with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )

    def _create_new_module(self, adamix_config, adapter_name, target):
        # loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        # if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
        #     eightbit_kwargs = kwargs.copy()
        #     eightbit_kwargs.update(
        #         {
        #             "has_fp16_weights": target.state.has_fp16_weights,
        #             "memory_efficient_backward": target.state.memory_efficient_backward,
        #             "threshold": target.state.threshold,
        #             "index": target.index,
        #         }
        #     )
        #     new_module = Linear8bitLt(
        #         adapter_name,
        #         target.in_features,
        #         target.out_features,
        #         is_feedforward,
        #         bias=bias,
        #         **eightbit_kwargs,
        #     )
        # else:
        #  Create the Mixture of Adapter Module
        if not hasattr(self.model.config, 'hidden_size'):
            raise KeyError ("Hidden size of the model must be known")

        new_module = ExpertSoup(self.model.config.hidden_size, **adamix_config.to_dict())
        return new_module

    def _check_target_module_exists(self, adamix_config, key):
        if isinstance(adamix_config.target_modules, str):
            target_module_found = True if adamix_config.target_modules == key else False
        else:
            for target in adamix_config.target_modules:
                target_module_found = True if target == key else False
        return target_module_found

    def _find_and_replace(self, adapter_name):
        adamix_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False

        module_dict = dict(self.model.named_modules())
        for key in module_dict:
            module_dict[key] = type(module_dict[key]).__name__
            if not self._check_target_module_exists(adamix_config, module_dict[key]):
                continue

            if not is_target_modules_in_base_model:
                is_target_modules_in_base_model = True
            _, target, _ = _get_submodules(self.model, key)

            new_module = self._create_new_module(adamix_config, adapter_name, target)
            self._add_adapter_module(target, new_module)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {adamix_config.target_modules} not found in the base model. "
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

    def _add_adapter_module(self, target, new_module):
        target.add_module('adamix', new_module)

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
            if isinstance(module, ExpertSoup):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, ExpertSoup):
                module.active_adapter = adapter_name

    @staticmethod
    def _prepare_adamix_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_ADAMIX_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_ADAMIX_TARGET_MODULES_MAPPING[model_config["model_type"]]
        return peft_config

    def zero_adapters(self):
        r"""
        This method removes the added adapters. This is needed if someone wants to use the base model
        as a standalone model.
        """

        # if getattr(self.model, "is_loaded_in_8bit", False):
        #     raise ValueError("Cannot merge adamix layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "adamix" in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, ExpertSoup):
                delattr(parent, target_name)
        return self.model

def mark_only_adamix_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if "adamix" not in n:
            p.requires_grad = False
        else:
            p.requires_grad = True

def get_two_view_from_model(model: nn.Module) -> Tuple[Tuple]:
    # NOTE: To perform backward pass on the two views, you must set retain_graph=True in the optimizer

    two_views = []
    for name, module in model.named_modules():
        if 'adamix' in name:
            if isinstance(module.two_views, list):
                raise ValueError ("One of the returned view is empty")
            two_views.append(module.two_views)
    
    if len(two_views) == 0:
        raise ValueError ("Returned views are empty")

    return torch.stack(two_views, dim=0)


# # Below code is based on https://github.com/microsoft/lora/blob/main/loralib/layers.py
# # and modified to work with PyTorch FSDP


# #  ------------------------------------------------------------------------------------------
# #  Copyright (c) Microsoft Corporation. All rights reserved.
# #  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# #  ------------------------------------------------------------------------------------------

class MixtureSoup(torch.nn.Module):

    def __init__(self, expert, num_local_experts=1):
        super(MixtureSoup, self).__init__()

        self.expert_mixture = torch.nn.ModuleList(
            [copy.deepcopy(expert) for i in range(num_local_experts)])
        self.num_local_experts = num_local_experts

    def get_expert_by_idx(self, idx):
        return self.expert_mixture[idx]

    def expert_soup_forward(self, hidden_state):
        output = F.linear(hidden_state,
                          self.parameter_dict["weight"],
                          self.parameter_dict["bias"])
        return output

    def expert_soup(self):
        self.parameter_dict = {"weight": 0, "bias": 0}
        for idx in range(self.num_local_experts):
            single_expert = self.expert_mixture[idx]
            for s_name, s_param in single_expert.named_parameters():
                if "weight" in s_name:
                    p_name = "weight"
                else:
                    p_name = "bias"
                self.parameter_dict[p_name] = self.parameter_dict[p_name] + s_param/self.num_local_experts
        
    def forward(self, hidden_state: torch.Tensor):
        expert_output = None

        if self.expert_mixture[0].training:
            expert_idx = torch.randint(low=0, high=self.num_local_experts, size=(1,)).item()  # selected expert
            expert_output = self.get_expert_by_idx(expert_idx)(hidden_state)

        else:
            self.expert_soup()
            expert_output = self.expert_soup_forward(input[0])

        return expert_output

class ExpertSoup(nn.Module):
    def __init__(self, hidden_dim, adapter_dim, num_expert=4, sharing_down=False, sharing_up=True, return_two_views=False, **kwargs):
        super().__init__()

        if sharing_down:
            self.MoA_down = MixtureSoup(nn.Linear(hidden_dim, adapter_dim), 1)
        else:
            self.MoA_down = MixtureSoup(nn.Linear(hidden_dim, adapter_dim), num_expert)

        if sharing_up:
            self.MoA_up = MixtureSoup(nn.Linear(adapter_dim, hidden_dim), 1)
        else:
            self.MoA_up = MixtureSoup(nn.Linear(adapter_dim, hidden_dim), num_expert)

        self.two_views = []

    # NOTE: During training, you must forward pass the input twice to get two outputs and apply the KL div minimization
    # Use the first ouput as input to next layer
    # During inference, only one forward pass is required
    def forward(self, x, residual):
        result1 = self.MoA_down(x)
        result1 = nn.GeLU(result1)
        result1 = self.MoA_up(result1)
        result1 = result1 + residual

        if self.training and self.return_two_views:
            result2 = self.MoA_down(x)
            result2 = nn.GeLU(result2)
            result2 = self.MoA_up(result2)
            result2 = result2 + residual
            self.two_views = torch.stack([result1, result2], dim=0)
        return result1
        


# if is_bnb_available():

#     class Linear8bitLt(bnb.nn.Linear8bitLt, IA3Layer):
#         # (IA)^3 implemented in a dense layer
#         def __init__(
#             self,
#             adapter_name,
#             in_features,
#             out_features,
#             is_feedforward,
#             **kwargs,
#         ):
#             bnb.nn.Linear8bitLt.__init__(
#                 self,
#                 in_features,
#                 out_features,
#                 bias=kwargs.get("bias", True),
#                 has_fp16_weights=kwargs.get("has_fp16_weights", True),
#                 memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
#                 threshold=kwargs.get("threshold", 0.0),
#                 index=kwargs.get("index", None),
#             )
#             IA3Layer.__init__(self, in_features=in_features, out_features=out_features, is_feedforward=is_feedforward)

#             # Freezing the pre-trained weight matrix
#             self.weight.requires_grad = False

#             init_adamix_weights = kwargs.pop("init_adamix_weights", True)
#             self.update_layer(adapter_name, init_adamix_weights)
#             self.active_adapter = adapter_name
#             self.is_feedforward = is_feedforward

# #         def forward(self, x: torch.Tensor):
# #             if self.disable_adapters or self.active_adapter not in self.ia3_l.keys():
# #                 return super().forward(x)
# #             else:
# #                 if not torch.is_autocast_enabled():
# #                     if x.dtype != torch.float32:
# #                         x = x.float()
# #                     if self.is_feedforward:
# #                         result = super().forward(x * self.ia3_l[self.active_adapter].flatten())
# #                     else:
# #                         result = super().forward(x)
# #                         expected_dtype = result.dtype
# #                         result = (result * self.ia3_l[self.active_adapter].flatten()).to(expected_dtype)
# #                 else:
# #                     if self.is_feedforward:
# #                         result = super().forward(x * self.ia3_l[self.active_adapter].flatten())
# #                     else:
# #                         result = result * self.ia3_l[self.active_adapter].flatten()
# #             return result
