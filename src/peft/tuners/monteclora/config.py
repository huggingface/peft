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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MonteCLoraConfig:
    """
    This is the sub-configuration class to store the configuration for MonteCLoRA (Monte Carlo Low-Rank Adaptation).
    MonteCLoRA introduces variational inference into LoRA by adding Monte Carlo sampling to the adapter weights.

    Args:
        monteclora_n (`int`):
            Number of Monte Carlo samples to use. Default is 8.
        monteclora_m (`Optional[int]`):
            Additional parameter for the sampler. Default is None.
        use_entropy (`bool`):
            Whether to use entropy regularization in the variational loss. Default is False.
        dirichlet_prior (`float`):
            Prior parameter for Dirichlet distribution used in expert weight sampling. Default is 0.1.
        sample_scaler (`float`):
            Scaling factor for the Monte Carlo samples. Controls the magnitude of variational perturbations.
            Default is 1e-4.
        kl_loss_weight (`float`):
            Weight for the KL divergence loss component in the variational objective. Default is 1e-5.
        mc_training (`bool`):
            Whether to enable Monte Carlo training mode. Default is True.
        buffer_size (`int`):
            Size of the buffer for the Monte Carlo sampler. Default is 150.
    """

    monteclora_n: int = field(
        default=8,
        metadata={"help": "Number of Monte Carlo samples to use."},
    )
    monteclora_m: Optional[int] = field(
        default=None,
        metadata={"help": "Additional parameter for the sampler."},
    )
    use_entropy: bool = field(
        default=False,
        metadata={"help": "Whether to use entropy regularization in the variational loss."},
    )
    dirichlet_prior: float = field(
        default=0.1,
        metadata={"help": "Prior parameter for Dirichlet distribution used in expert weight sampling."},
    )
    sample_scaler: float = field(
        default=1e-4,
        metadata={
            "help": "Scaling factor for the Monte Carlo samples. Controls the magnitude of variational perturbations."
        },
    )
    kl_loss_weight: float = field(
        default=1e-5,
        metadata={"help": "Weight for the KL divergence loss component in the variational objective."},
    )
    mc_training: bool = field(
        default=True,
        metadata={"help": "Whether to enable Monte Carlo training mode."},
    )
    buffer_size: int = field(
        default=150,
        metadata={"help": "Size of the buffer for the Monte Carlo sampler."},
    )

    def __post_init__(self):
        if self.monteclora_n <= 0:
            raise ValueError("`monteclora_n` must be greater than 0.")
        if self.dirichlet_prior <= 0:
            raise ValueError("`dirichlet_prior` must be greater than 0.")
        if self.buffer_size <= 0:
            raise ValueError("`buffer_size` must be greater than 0.")
