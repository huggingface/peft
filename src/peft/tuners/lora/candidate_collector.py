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

"""LoRA candidate enumerator. Registered for `PeftType.LORA`.

The supported-types tuple mirrors `peft.tuners.lora.layer.dispatch_default`
but this module does NOT call dispatch_default itself: dispatch_default has
side effects on `config.fan_in_fan_out` (lines 2495-2500 and 2502-2508 of
layer.py at commit 2859358). If LoRA gains support for a new module type,
both this tuple and dispatch_default need updating.
"""

from torch import nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.target_suggester import register_candidate_collector
from peft.tuners.tuners_utils import BaseTunerLayer

_LORA_SUPPORTED_TYPES = (
    nn.Linear,
    nn.Embedding,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.MultiheadAttention,
    Conv1D,
)


@register_candidate_collector("LORA")
def lora_candidates(model):
    """Yield full module names that LoRA could validly adapt.

    Skips:
        - the root module (empty name)
        - modules already wrapped by PEFT (BaseTunerLayer instances)
    """
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, BaseTunerLayer):
            continue
        if isinstance(module, _LORA_SUPPORTED_TYPES):
            yield name
