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

import torch

from peft.import_utils import is_inc_available
from peft.tuners.tuners_utils import BaseTunerLayer

from .layer import Linear, Conv2d


def dispatch_inc(target: torch.nn.Module, adapter_name: str, **kwargs):
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if is_inc_available():
        from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import (
            PatchedLinear,
            PatchedConv2d,
        )
        if isinstance(target_base_layer, PatchedLinear):
            new_module = Linear(target, adapter_name, **kwargs)
        elif isinstance(target_base_layer, torch.nn.Conv2d):
            new_module = Conv2d(target, adapter_name, **kwargs)

    return new_module
