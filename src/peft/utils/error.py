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


class PeftError(Exception):
    """Base PEFT error"""


class NoMatchingPeftModuleError(PeftError, ValueError):
    """The adapter being injected matched no module or parameter of the base model.

    Raised e.g. when the `target_modules` of the PEFT config matched nothing, which most often points at a
    misconfiguration. It subclasses `ValueError` for backwards compatibility with code that intercepted the generic
    error raised previously. Code that adds such an adapter intentionally can catch this error (or `PeftError`) and
    proceed.
    """
