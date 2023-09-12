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
from typing import TYPE_CHECKING
from ...import_utils import _LazyModule, OptionalDependencyNotAvailable
from peft.import_utils import is_bnb_4bit_available, is_bnb_available, is_auto_gptq_available

_import_structure = {
    "config": ["AdaLoraConfig"],
    "layer": ["AdaLoraLayer", "RankAllocator", "SVDLinear"],
    "model": ["AdaLoraModel"],
}

try:
    if not is_bnb_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["bnb"] = ["SVDLinear8bitLt"]

try:
    if not is_bnb_4bit_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["bnb"] = ["SVDLinear4bit"]

try:
    if not is_auto_gptq_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["gptq"] = ["SVDQuantLinear"]



if TYPE_CHECKING:
    from .config import AdaLoraConfig
    from .layer import AdaLoraLayer, RankAllocator, SVDLinear
    from .model import AdaLoraModel


    try:
        if not is_auto_gptq_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .gptq import SVDQuantLinear


    try:
        if not is_bnb_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .bnb import SVDLinear8bitLt

    try:
        if not is_bnb_4bit_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .bnb import SVDLinear4bit
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)