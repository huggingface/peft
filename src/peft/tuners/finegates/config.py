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
from typing import Literal, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class FineGatesConfig(PeftConfig):
    """
    Configuration for FineGates adapters.

    FineGates learns structured row and column gates for frozen linear layers. During training, gates are sampled via a
    hard-sigmoid relaxation; during evaluation and merging, deterministic gate values are used.

    Args:
        target_modules (`Optional[Union[list[str], str]]`):
            Modules to adapt. If omitted, PEFT selects defaults based on the architecture.
        exclude_modules (`Optional[Union[list[str], str]]`):
            Modules to exclude from adaptation.
        fan_in_fan_out (`bool`):
            Set this to `True` for layers like `transformers.pytorch_utils.Conv1D` that store weights as
            `(fan_in, fan_out)`.
        modules_to_save (`Optional[list[str]]`):
            Additional modules to keep trainable and save with the adapter checkpoint.
        bias (`str`):
            Bias training mode. Set to `"finegates_only"` to mirror the reference implementation and train biases for
            wrapped sparse layers in addition to the gate parameters.
        target_sparsity (`float`):
            Target fraction of pruned rows/columns. This is used by the auxiliary sparsity loss.
        sparsity_loss_weight (`float`):
            Multiplier for the auxiliary sparsity loss returned by `FineGatesModel._get_finegates_loss()`.
        gate_init_mean (`float`):
            Mean of the normal initialization for gate logits.
        gate_init_std (`float`):
            Standard deviation of the normal initialization for gate logits.
        gate_noise_std (`float`):
            Standard deviation of the training-time Gaussian noise added before applying the hard sigmoid relaxation.
    """

    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with FineGates. "
                "For example, ['q_proj', 'v_proj'] or '.*decoder.*(q_proj|v_proj)$'. "
                "This can also be 'all-linear' to target all linear/Conv1D modules except the output head on "
                "transformers models."
            )
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from FineGates."},
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the target layer stores weight like (fan_in, fan_out)."},
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from FineGates layers to be set as trainable and saved in the final checkpoint."
            )
        },
    )
    init_weights: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Compatibility flag used by PEFT's generic test harness. If set to False, FineGates uses a "
                "non-identity random initialization to exercise adapter behavior in generic tests."
            )
        },
    )
    bias: Literal["none", "all", "finegates_only"] = field(
        default="none",
        metadata={"help": "Bias type for FineGates. Can be 'none', 'all', or 'finegates_only'."},
    )
    target_sparsity: float = field(
        default=0.2,
        metadata={"help": "Target structured sparsity level used by the FineGates auxiliary loss."},
    )
    sparsity_loss_weight: float = field(
        default=1e-2,
        metadata={"help": "Weight applied to the FineGates auxiliary sparsity loss."},
    )
    gate_init_mean: float = field(
        default=0.9,
        metadata={"help": "Mean of the normal initialization used for gate logits."},
    )
    gate_init_std: float = field(
        default=0.1,
        metadata={"help": "Standard deviation of the normal initialization used for gate logits."},
    )
    gate_noise_std: float = field(
        default=0.0,
        metadata={"help": "Training-time Gaussian noise standard deviation used in the hard-sigmoid relaxation."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.FINEGATES
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )

        if not 0.0 <= self.target_sparsity < 1.0:
            raise ValueError(f"`target_sparsity` must be in [0, 1), got {self.target_sparsity}.")
        if self.sparsity_loss_weight < 0:
            raise ValueError(f"`sparsity_loss_weight` must be non-negative, got {self.sparsity_loss_weight}.")
        if self.bias not in {"none", "all", "finegates_only"}:
            raise ValueError(f"`bias` must be one of 'none', 'all', or 'finegates_only', got {self.bias}.")
        if self.gate_init_std <= 0:
            raise ValueError(f"`gate_init_std` must be > 0, got {self.gate_init_std}.")
        if self.gate_noise_std < 0:
            raise ValueError(f"`gate_noise_std` must be >= 0, got {self.gate_noise_std}.")
