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

from .config import LoKrConfig
from .layer import Conv2d, Linear, LoKrLayer
from .model import LoKrModel

from peft.utils import register_peft_method


__all__ = ["LoKrConfig", "LoKrModel", "Conv2d", "Linear", "LoKrLayer"]

register_peft_method(name="lokr", config_cls=LoKrConfig, model_cls=LoKrModel, is_mixed_compatible=True)
