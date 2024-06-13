import math
import random
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.utils.other import transpose


random.seed(56)
from typing import Any, Optional, Union

import torch.nn.functional as F
from torch import svd_lowrank
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose


# from .dora import DoraConv2dLayer, DoraLinearLayer

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_glora_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if "glora_" not in n:
            p.requires_grad = False


class GLoraLayer(BaseTunerLayer):
    def __init__(self, in_features: int, out_features: int, r: int, adapter_name: str, **kwargs):
        self.r = {}
        self.r[adapter_name] = r
        self.glora_Ad, self.glora_Au = self.make_param((out_features, in_features), f"LoRA_{r}")
        self.glora_Bd, self.glora_Bu = self.make_param((out_features, in_features), f"LoRA_{r}")
        self.glora_Cd, self.glora_Cu = self.make_param((in_features, 1), f"LoRA_{r}")
        self.glora_D = nn.Parameter(torch.zeros(out_features))
        self.glora_E = nn.Parameter(torch.zeros(out_features))
        self.eval_config = None
        nn.init.kaiming_uniform_(self.glora_Au, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.glora_Bu, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.glora_Cu, a=math.sqrt(5))
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs
        config_A_B = [f"LoRA_{r}", "vector", "constant", "none"]
        config_C = [f"LoRA_{r}", "vector", "none"]
        config_D_E = ["constant", "none", "vector"]
        self.configs = []
        for A in config_A_B:
            for B in config_A_B:
                for C in config_C:
                    for D in config_D_E:
                        for E in config_D_E:
                            config = {"A": A, "B": B, "C": C, "D": D, "E": E}
                            self.configs.append(config)

    def make_param(self, shape, config=None):
        if "LoRA" in config:
            out_feature = shape[0]
            in_feature = shape[1]
            try:
                rank = int(config.split("_")[1])
            except:
                rank = 4
            return nn.Parameter(torch.zeros(out_feature, rank)), nn.Parameter(torch.zeros(rank, in_feature))
        return nn.Parameter(torch.zeros(*shape))

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep
