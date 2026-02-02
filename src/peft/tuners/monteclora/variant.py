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

from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.lora.layer import Linear, LoraVariant
from peft.tuners.tuners_utils import MonteCLoRASampler


class MonteCLoraLinearVariant(LoraVariant):
    """
    MonteCLoRA (Monte Carlo Low-Rank Adaptation) variant implementation.

    This variant adds variational inference to LoRA by introducing Monte Carlo sampling
    to the adapter weights during training. The sampling is controlled by a MonteCLoRASampler
    that maintains variational parameters and produces perturbations to the LoRA weights.
    """

    @staticmethod
    def init(module: Linear, adapter_name: str, **kwargs: Any) -> None:
        """
        Initialize MonteCLoRA for a LoRA layer.

        This adds a MonteCLoRASampler to the layer that will be used during forward passes
        to sample variational perturbations for the LoRA A matrix.

        Args:
            module: The Linear LoRA layer to add MonteCLoRA to
            adapter_name: Name of the adapter
            **kwargs: Must contain 'monteclora_config' with MonteCLoRA configuration
        """
        monteclora_config = kwargs.get("monteclora_config")
        if monteclora_config is None:
            raise ValueError("MonteCLoraLinearVariant.init() requires 'monteclora_config' in kwargs")

        if not hasattr(module, "lora_monteclora_sampler"):
            module.adapter_layer_names = module.adapter_layer_names[:] + ("lora_monteclora_sampler",)
            module.lora_monteclora_sampler = nn.ModuleDict()

        lora_A = module.lora_A[adapter_name]
        r = lora_A.out_features
        in_features = module.in_features

        sampler = MonteCLoRASampler(
            in_features=in_features,
            out_features=r,
            monteclora_n=monteclora_config.monteclora_n,
            use_entropy=monteclora_config.use_entropy,
            dirichlet_prior=monteclora_config.dirichlet_prior,
            sample_scaler=monteclora_config.sample_scaler,
            kl_loss_weight=monteclora_config.kl_loss_weight,
            mc_training=monteclora_config.mc_training,
            buffer_size=monteclora_config.buffer_size,
            device=lora_A.weight.device,
        )

        module.lora_monteclora_sampler[adapter_name] = sampler

    @staticmethod
    def merge_safe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        """
        Safely merge MonteCLoRA adapter weights into base weights.

        For merging, we ignore the MC sampling and just use the base LoRA weights (lora_A and lora_B).
        This is equivalent to merging a standard LoRA adapter.

        Args:
            module: The Linear LoRA layer
            active_adapter: Name of the adapter to merge
            orig_weight: Original base layer weight

        Returns:
            New weight with adapter merged
        """
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)
        new_weight = orig_weight + delta_weight.to(orig_dtype)
        return new_weight

    @staticmethod
    def merge_unsafe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> None:
        """
        Merge MonteCLoRA adapter weights into base weights (in-place).

        For merging, we ignore the MC sampling and just use the base LoRA weights (lora_A and lora_B).
        This is equivalent to merging a standard LoRA adapter.

        Args:
            module: The Linear LoRA layer
            active_adapter: Name of the adapter to merge
            orig_weight: Original base layer weight (modified in-place)
        """
        delta_weight = module.get_delta_weight(active_adapter)
        orig_weight.data += delta_weight

    @staticmethod
    def unmerge(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        """
        Unmerge MonteCLoRA adapter weights from base weights.

        For unmerging, we ignore the MC sampling and just use the base LoRA weights (lora_A and lora_B).
        This is equivalent to unmerging a standard LoRA adapter.

        Args:
            module: The Linear LoRA layer
            active_adapter: Name of the adapter to unmerge
            orig_weight: Current weight with adapter merged

        Returns:
            Weight with adapter unmerged
        """
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)
        new_weight = orig_weight - delta_weight.to(orig_dtype)
        return new_weight

    @staticmethod
    def forward(
        module: Linear,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with MonteCLoRA sampling.

        This samples variational perturbations from the MonteCLoRASampler and applies them
        to the LoRA A weights before computing the LoRA update.

        Args:
            module: The Linear LoRA layer
            active_adapter: Name of the active adapter
            x: Input tensor
            result: Output from the base layer
            **kwargs: Additional arguments (unused)

        Returns:
            result + LoRA update with MonteCLoRA sampling applied
        """
        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]  #lora_B is zero in the beginning. test for stochasticity of outputs might fail because of this
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]
        sampler = module.lora_monteclora_sampler[active_adapter]

        x = x.to(lora_A.weight.dtype)
        if isinstance(dropout, nn.Identity) or not module.training:
            x_dropped = x
        else:
            x_dropped = dropout(x)

        current_weight_A = lora_A.weight

        if module.training and hasattr(sampler, "mc_training") and sampler.mc_training:
            lora_A_vars, lora_A_wts = sampler()
            if not isinstance(lora_A_vars, int):
                if torch.isnan(lora_A_vars).any() or torch.isnan(lora_A_wts).any():
                    warnings.warn("MonteCLoRA sampling produced NaNs, using base weights.")
                else:
                    noise = torch.nan_to_num(lora_A_vars, nan=0.0)
                    base_w = lora_A.weight.T
                    perturbed_w = (
                        base_w + noise
                    )
                    averaged_w = torch.einsum("n,nij->ij", lora_A_wts, perturbed_w)
                    current_weight_A = averaged_w.T
        out_A = F.linear(x_dropped, current_weight_A)
        result = result + lora_B(out_A) * scaling
        return result
