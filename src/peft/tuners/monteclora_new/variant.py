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

        # Register the sampler container if it doesn't exist
        if not hasattr(module, "lora_monteclora_sampler"):
            # Add to adapter_layer_names so it's recognized as trainable
            module.adapter_layer_names = module.adapter_layer_names[:] + ("lora_monteclora_sampler",)
            module.lora_monteclora_sampler = nn.ModuleDict()

        # Get the rank from the lora_A layer
        lora_A = module.lora_A[adapter_name]
        r = lora_A.out_features
        in_features = module.in_features

        # Create the MonteCLoRA sampler
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
        MonteCLoRA cannot be safely merged because it involves stochastic sampling.
        """
        raise NotImplementedError("MonteCLoRA does not support safe merging due to its stochastic nature.")

    @staticmethod
    def merge_unsafe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> None:
        """
        MonteCLoRA cannot be merged because it involves stochastic sampling.
        """
        raise NotImplementedError("MonteCLoRA does not support merging due to its stochastic nature.")

    @staticmethod
    def unmerge(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        """
        MonteCLoRA cannot be unmerged because it involves stochastic sampling.
        """
        raise NotImplementedError("MonteCLoRA does not support unmerging due to its stochastic nature.")

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
        lora_B = module.lora_B[active_adapter]
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]
        sampler = module.lora_monteclora_sampler[active_adapter]

        # Apply dropout
        x = x.to(lora_A.weight.dtype)
        if isinstance(dropout, nn.Identity) or not module.training:
            x_dropped = x
        else:
            x_dropped = dropout(x)

        # Get the base LoRA A weight
        current_weight_A = lora_A.weight

        # Apply MonteCLoRA sampling during training
        if module.training and hasattr(sampler, "mc_training") and sampler.mc_training:
            # Sample from the variational distribution
            lora_A_vars, lora_A_wts = sampler()

            # Check if sampling was successful (returns -1 when not training)
            if not isinstance(lora_A_vars, int):
                # Check for NaN values
                if torch.isnan(lora_A_vars).any() or torch.isnan(lora_A_wts).any():
                    warnings.warn("MonteCLoRA sampling produced NaNs, using base weights.")
                else:
                    # Apply the variational perturbation
                    # lora_A_vars shape: [monteclora_n, in_features, out_features]
                    # We need to transpose to match lora_A.weight shape [out_features, in_features]
                    noise = torch.nan_to_num(lora_A_vars, nan=0.0)

                    # Transpose the base weight to match noise shape
                    base_w = lora_A.weight.T  # [in_features, out_features]

                    # Add noise to create perturbed weights for each sample
                    perturbed_w = (
                        base_w + noise
                    )  # Broadcasting: [in_features, out_features] + [n, in_features, out_features]

                    # Weighted average over Monte Carlo samples
                    averaged_w = torch.einsum("n,nij->ij", lora_A_wts, perturbed_w)

                    # Transpose back to LoRA A weight shape
                    current_weight_A = averaged_w.T

        # Compute LoRA update: x @ A^T @ B^T * scaling
        out_A = F.linear(x_dropped, current_weight_A)
        result = result + lora_B(out_A) * scaling

        return result
