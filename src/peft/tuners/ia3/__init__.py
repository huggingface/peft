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
from peft.import_utils import is_bnb_available

_import_structure = {
    "config": ["IA3Config"],
    "layer": ["IA3Layer", "Linear"],
    "model": ["IA3Model"],
}

try:
    if not is_bnb_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["bnb"] = ["Linear8bitLt"]

if TYPE_CHECKING:
    from .config import IA3Config
    from .layer import IA3Layer, Linear
    from .model import IA3Model
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
