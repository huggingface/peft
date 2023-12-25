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

import importlib
import math
import operator
import re
import warnings
from dataclasses import asdict, replace
from enum import Enum
from functools import reduce
from itertools import chain
from typing import Any, List, Optional

import torch
from torch import nn
from torch.nn.modules import Module
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D
from peft.config import PeftConfig
from peft.utils import TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists


class LNTuningModel(BaseTuner):
    """
    Creates 
    """
    def __init__(self, model, config, adapter_name) -> None:
        self.adapter_name = adapter_name
        super().__init__(model, config, adapter_name)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)
        
    @staticmethod
    def _prepare_adapter_config(peft_config: PeftConfig, model_config: dict) -> PeftConfig:
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _create_and_replace(self, 
                            peft_config: PeftConfig, 
                            adapter_name: str, 
                            target: Module, 
                            target_name: str, 
                            parent: Module, 
                            **optional_kwargs: Any) -> None:
        # Don't need to do anything here: just mark the layernorms as trainable is enough
        pass
    
    def _mark_only_adapters_as_trainable(self, model: Module):
        # Need to mark all layernorms as trainable
        for n, p in model.named_parameters():
            # check to see if the parameter is a layernorm
            flag = False
            for module_name in self.peft_config[self.adapter_name].target_modules:
                if module_name in n:
                    flag = True
                    break
            p.requires_grad = flag

    
    def _check_target_module_exists(self, peft_config: PeftConfig, key: str) -> bool:
        return check_target_module_exists(peft_config, key)