# Copyright 2026-present the HuggingFace Inc. team.
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

from .config import DummyConfig
from .layer import Conv1d, Conv2d, Conv3d, DummyLayer, Linear
from .model import DummyModel


__all__ = ["Conv1d", "Conv2d", "Conv3d", "DummyConfig", "DummyLayer", "DummyModel", "Linear"]

register_peft_method(
    name="dummy",
    config_cls=DummyConfig,
    model_cls=DummyModel,
    prefix="dummy_",
    is_mixed_compatible=True,
)
