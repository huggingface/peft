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
from typing import Optional, Union

from peft.tuners.lora.config import LoraConfig
from peft.utils import PeftType, TaskType


@dataclass
class MonteCLoraConfig(LoraConfig):
    """
    Configuration class for MonteCLoRA.
    """

    # MonteCLoRA toggles
    use_monteclora: bool = True

    # Where to apply MonteCLoRA injection: ["lora_A"], ["lora_B"], etc.
    monteclora_at: Union[str, list[str]] = field(default_factory=lambda: ["lora_A"])

    # Specific modules to target (e.g. ["q_proj", "v_proj"]).
    # If None, defaults to LoraConfig.target_modules in post_init.
    monteclora_targets: Optional[Union[list[str], str]] = None

    # Monte Carlo parameters
    monteclora_n: int = 8
    monteclora_m: Optional[int] = None  # Added: Required by layer.py sampler
    use_entropy: bool = False
    dirichlet_prior: float = 0.1
    sample_scaler: float = 1e-4
    kl_loss_weight: float = 1e-5
    mc_training: bool = True
    buffer_size: int = 150

    # Required for PEFT saving/loading
    task_type: Union[str, TaskType] = TaskType.CAUSAL_LM

    def __post_init__(self):
        super().__post_init__()

        # Normalize list fields
        if isinstance(self.monteclora_at, str):
            self.monteclora_at = [self.monteclora_at]

        # Sync targets: If monteclora_targets is not set, apply to all LoRA targets
        if self.monteclora_targets is None:
            self.monteclora_targets = self.target_modules

        # CRITICAL: Handle PeftType
        # If you haven't modified the library source, use a string or existing type
        try:
            self.peft_type = PeftType.MONTECLORA
        except AttributeError:
            # Fallback if MONTECLORA isn't in the Enum yet
            self.peft_type = "MONTECLORA"

    @property
    def monteclora_config(self):
        """
        Backward compatibility helper. Allows model.py to access `config.monteclora_config` even though this object IS
        the config.
        """
        return self
