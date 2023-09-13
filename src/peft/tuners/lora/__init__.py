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

from peft.import_utils import is_bnb_4bit_available, is_bnb_available

from .config import LoraConfig
from .gptq import QuantLinear
from .layer import Conv2d, Embedding, Linear, LoraLayer
from .model import LoraModel


__all__ = ["LoraConfig", "Conv2d", "Embedding", "LoraLayer", "Linear", "LoraModel", "QuantLinear"]


if is_bnb_available():
    from .bnb import Linear8bitLt

    __all__ += ["Linear8bitLt"]

if is_bnb_4bit_available():
    from .bnb import Linear4bit

    __all__ += ["Linear4bit"]
