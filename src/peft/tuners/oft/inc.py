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

# NOTE: PEFT tests related to INC are handled under Optimum-Habana repository:
# - LLMs: https://github.com/huggingface/optimum-habana/blob/main/tests/test_peft_inference.py
# - Diffusers: https://github.com/huggingface/optimum-habana/blob/main/tests/test_diffusers.py

from typing import Optional

import torch

from peft.import_utils import is_inc_available
from peft.tuners.tuners_utils import BaseTunerLayer

from .layer import Linear


if is_inc_available():

    class IncOFTLinear(Linear):
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            **kwargs,
        ):
            super().__init__(base_layer, adapter_name, **kwargs)

        def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
            """
            Merge the active adapter weights into the base weights

            Args:
                safe_merge (`bool`, *optional*):
                    If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                    before merging the weights. This is useful if you want to check if the merge operation will produce
                    NaNs. Defaults to `False`.
                adapter_names (`list[str]`, *optional*):
                    The list of adapter names that should be merged. If None, all active adapters will be merged.
                    Defaults to `None`.
            """
            raise NotImplementedError("Merging OFT with INC layers is not yet implemented")

        def unmerge(self) -> None:
            """
            This method unmerges all merged adapter layers from the base weights.
            """
            raise NotImplementedError("Unmerging OFT from INC layers is not yet implemented")


def dispatch_inc(target: torch.nn.Module, adapter_name: str, **kwargs):
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if is_inc_available():
        from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import (
            PatchedLinear,
        )

        if isinstance(target_base_layer, PatchedLinear):
            new_module = IncOFTLinear(target, adapter_name, **kwargs)

    return new_module
