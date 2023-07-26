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


class BaseTunerMixin:
    _is_plugable = False
    active_adapter = None

    @property
    def peft_is_plugable(self):
        return self._is_plugable

    def __post_init__(self):
        if self.active_adapter is None:
            raise ValueError("active_adapter must be set in the subclass")
