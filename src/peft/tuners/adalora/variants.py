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

"""
AdaDoRA variant: DoRA variant for AdaLoRA's SVD-based adaptation.

This module follows the standard LoraVariant pattern used in PEFT, providing static methods for initialization, forward
pass, and merging operations specific to AdaLoRA's 3-factor decomposition.
"""

from typing import Any

import torch

from peft.tuners.lora.layer import LoraVariant
from peft.utils.other import transpose

from .dora import AdaDoraLinearLayer


class AdaDoraLinearVariant(LoraVariant):
    """
    DoRA variant for AdaLoRA's SVD-based adaptation.

    Follows the standard LoraVariant pattern while accounting for AdaLoRA's 3-factor decomposition (A, E, B matrices)
    instead of standard LoRA's 2-factor decomposition (A, B).

    The variant pattern provides:
    - init: Creates AdaDoraLinearLayer and stores in lora_magnitude_vector
    - forward: Handles DoRA-aware forward pass with SVD decomposition
    - merge_safe/merge_unsafe: Handles weight merging with DoRA scaling
    - unmerge: Reverses the merge operation
    """

    @staticmethod
    def init(module, adapter_name: str, **kwargs: Any) -> None:
        """
        Initialize AdaDoRA layer and store in lora_magnitude_vector.

        Creates an AdaDoraLinearLayer with the magnitude vector initialized based on ||W + ΔW|| where ΔW uses AdaLoRA's
        SVD decomposition.

        Args:
            module: The SVDLinear module
            adapter_name: Name of the adapter being initialized
            **kwargs: Additional arguments (unused)
        """
        if not module.lora_magnitude_vector:
            # first dora layer being added, add lora_magnitude_vector to learnable params
            module.adapter_layer_names = module.adapter_layer_names[:] + ("lora_magnitude_vector",)

        dora_layer = AdaDoraLinearLayer(fan_in_fan_out=getattr(module, "fan_in_fan_out", False))

        lora_A = module.lora_A[adapter_name]
        lora_B = module.lora_B[adapter_name]
        lora_E = module.lora_E[adapter_name]
        scaling = module.scaling[adapter_name]
        ranknum = module.ranknum[adapter_name].item()

        dora_layer.update_layer(
            base_layer=module.get_base_layer(),
            lora_A=lora_A,
            lora_B=lora_B,
            lora_E=lora_E,
            scaling=scaling,
            ranknum=ranknum,
            place_on_cpu=False,
        )
        module.lora_magnitude_vector[adapter_name] = dora_layer

    @staticmethod
    def forward(
        module,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        DoRA forward pass for AdaLoRA.

        Computes the DoRA-adjusted output by delegating to AdaDoraLinearLayer.

        Args:
            module: The SVDLinear module
            active_adapter: Name of the active adapter
            x: Input tensor (already cast to correct dtype)
            result: Result from base layer forward pass
            **kwargs: Additional arguments (unused)

        Returns:
            DoRA-adjusted output tensor
        """
        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]
        lora_E = module.lora_E[active_adapter]
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]
        ranknum = module.ranknum[active_adapter].item()

        # apply dropout in training mode
        if not isinstance(dropout, torch.nn.Identity) and module.training:
            x = dropout(x)
            base_result = None
        else:
            base_result = result

        result = result + module.lora_magnitude_vector[active_adapter](
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            lora_E=lora_E,
            scaling=scaling,
            ranknum=ranknum,
            base_layer=module.get_base_layer(),
            base_result=base_result,
        )
        return result

    @staticmethod
    def merge_safe(module, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        """
        Safe merge for AdaDoRA.

        Merges adapter weights into base weights with DoRA magnitude scaling. Returns a new tensor (does not modify
        orig_weight in-place).

        The merged weight is computed as: W' = (m / ||W + ΔW||) * (W + ΔW)

        Args:
            module: The SVDLinear module
            active_adapter: Name of the adapter to merge
            orig_weight: Original base layer weight

        Returns:
            Merged weight tensor
        """
        delta_weight = module.get_delta_weight(active_adapter)

        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]
        lora_E = module.lora_E[active_adapter]
        scaling = module.scaling[active_adapter]
        ranknum = module.ranknum[active_adapter].item()

        # get weight norm using the AdaDoRA layer
        weight_norm = (
            module.lora_magnitude_vector[active_adapter]
            .get_weight_norm(orig_weight, lora_A, lora_B, lora_E, scaling, ranknum)
            .detach()
        )

        # cache weight_norm for potential unmerge operation
        module._cache_store(f"{active_adapter}-weight_norm", weight_norm)

        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm
        dora_factor = transpose(dora_factor.view(-1, 1), module.fan_in_fan_out)
        new_weight = dora_factor * (orig_weight + delta_weight)
        return new_weight.to(orig_weight.dtype)

    @staticmethod
    def merge_unsafe(module, active_adapter: str, orig_weight: torch.Tensor) -> None:
        """
        Unsafe merge for AdaDoRA.

        Merges adapter weights into base weights with DoRA magnitude scaling. Modifies orig_weight in-place.

        Args:
            module: The SVDLinear module
            active_adapter: Name of the adapter to merge
            orig_weight: Original base layer weight (modified in-place)
        """
        delta_weight = module.get_delta_weight(active_adapter)

        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]
        lora_E = module.lora_E[active_adapter]
        scaling = module.scaling[active_adapter]
        ranknum = module.ranknum[active_adapter].item()

        weight_norm = (
            module.lora_magnitude_vector[active_adapter]
            .get_weight_norm(orig_weight, lora_A, lora_B, lora_E, scaling, ranknum)
            .detach()
        )

        # cache weight_norm for potential unmerge operation
        module._cache_store(f"{active_adapter}-weight_norm", weight_norm)

        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm
        dora_factor = transpose(dora_factor.view(-1, 1), module.fan_in_fan_out)
        new_weight = dora_factor * (orig_weight.data + delta_weight)
        orig_weight.data = new_weight.to(orig_weight.dtype)

    @staticmethod
    def unmerge(module, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        """
        Unmerge for AdaDoRA.

        Reverses the merge operation by removing the DoRA-scaled adapter contribution from the merged weights.

        The unmerged weight is computed as: W = W' / (m / ||W + ΔW||) - ΔW

        Args:
            module: The SVDLinear module
            active_adapter: Name of the adapter to unmerge
            orig_weight: Merged weight tensor

        Returns:
            Unmerged (original) weight tensor
        """
        delta_weight = module.get_delta_weight(active_adapter)
        weight_norm = module._cache_pop(f"{active_adapter}-weight_norm")
        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm
        new_weight = orig_weight.data / dora_factor.view(-1, 1) - delta_weight
        return new_weight.to(orig_weight.dtype)
