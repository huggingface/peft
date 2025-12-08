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

from peft.utils import register_peft_method

from .config import LilyConfig
from .layer import Conv2d, Linear, LilyLayer
from .model import LilyModel


__all__ = ["Linear", "LilyConfig", "LilyLayer", "LilyModel"]

register_peft_method(name="lily", config_cls=LilyConfig, model_cls=LilyModel, prefix="lily_", is_mixed_compatible=True)
