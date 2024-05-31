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
from dataclasses import dataclass
from typing import Optional

from peft.config import PeftConfig
from peft.utils.peft_types import PeftType


@dataclass
class XLoraConfig(PeftConfig):
    r"""
    This is the configuration class to store the configuration of a `XLoraModel`. When the config is reloaded, the
    paths of the `adapters` field is disregarded in favor of the saved adapters. As such, only the keys matter during
    loading.

    Args:
        hidden_size (`int`):
            Hidden size of the base model.
        adapters (`dict`):
            Mapping of adapter names to the LoRA adapter id, as per PeftModel.load_adapter. *They will be automatically
            loaded*, to use as LoRA experts. When using from_pretrained, pass the new adapters dict as a keyword
            argument.
        enable_softmax (`bool`, *optional*, defaults to `True`):
            Enable softmax application for the X-LoRA classifier.
        enable_softmax_topk (`bool`, *optional*, defaults to `False`):
            Enable softmax application for the top-k LoRA adapters. Mutually exclusive to `enable_softmax` and must
            only be set if `top_k_lora` is.
        softmax_temperature (`float`, *optional*, defaults to 1.0):
            Softmax temperature, lower yields sharper predictions
        layerwise_scalings (`bool`, *optional*, defaults to `False`):
            If True, generate scalings for each LoRA adapter (each layer). If this is False, then scalings will be
            broadcasted, the same, to each layer.
        top_k_lora (`int`, *optional*, defaults to None):
            Sparsely select the top_k LoRA experts instead of the default dense method.
        xlora_depth (`int`, *optional*, defaults to 1):
            Depth of the X-LoRA classifier.
        xlora_size (`int`, *optional*, defaults to 2048):
            Hidden size of the X-LoRA classifier, irrelevant if `xlora_depth=1`.
        xlora_dropout_p (`float`, *optional*, defaults to 0.2):
            Dropout probability of the X-LoRA classifier, irrelevant if `xlora_depth=1`.
        use_trainable_adapters (`bool`, *optional*, defaults to False):
            Make the adapters trainable.
        scaling_pass_value (`float`, *optional*, defaults to 0):
            Scaling pass value.
        global_scaling_weight (`float`, *optional*, defaults to 1):
            Weight to multiply output of each LoRA adapter by.
    """

    hidden_size: int = None  # type: ignore
    adapters: dict[str, str] = None  # type: ignore
    enable_softmax: bool = True
    enable_softmax_topk: bool = False
    layerwise_scalings: bool = False
    xlora_depth: int = 1
    xlora_size: int = 2048
    xlora_dropout_p: float = 0.2
    use_trainable_adapters: bool = False
    softmax_temperature: float = 1.0
    top_k_lora: Optional[int] = None
    scaling_pass_value: float = 0.0
    global_scaling_weight: float = 1.0

    def __post_init__(self):
        self.peft_type = PeftType.XLORA

        if self.hidden_size is None:
            warnings.warn(
                "No value was provided for `hidden_size`. This will be set to 4096 by default, please ensure that this is correct."
            )
            self.hidden_size = 4096
        if self.adapters is None:
            warnings.warn(
                "No value was provided for for `adapters`. This will be set to empty, please ensure that this is correct."
            )
            self.adapters = {}

        if self.enable_softmax_topk and self.top_k_lora is None:
            warnings.warn("`enable_softmax_topk` enabled `top_k_lora` is not set")

        if self.enable_softmax_topk and self.enable_softmax:
            warnings.warn(
                "`enable_softmax_topk` and `enable_softmax` are both enabled. This will result in worse performance."
            )

        if self.top_k_lora is not None and self.top_k_lora < 1:
            warnings.warn("`top_k_lora` value must be at least 1.")
