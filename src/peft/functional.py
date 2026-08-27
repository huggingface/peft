# Copyright 2025-present the HuggingFace Inc. team.
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
"""
Functions that are useful for integration with non-PeftModel models, e.g. transformers or diffusers.

The functions provided here can be considered "public API" of PEFT and hence are safe to be used by packages that
provide PEFT integrations.
"""

from peft.mapping import inject_adapter_in_model
from peft.tuners.tuners_utils import cast_adapter_dtype, delete_adapter, set_adapter, set_requires_grad
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict


__all__ = [
    "cast_adapter_dtype",
    "delete_adapter",
    "get_peft_model_state_dict",
    "inject_adapter_in_model",
    "set_adapter",
    "set_peft_model_state_dict",
    "set_requires_grad",
]
