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

import warnings
from dataclasses import dataclass, field
from typing import Optional

from peft.tuners.lora import LoraConfig
from peft.utils import PeftType


@dataclass
class AdaLoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.AdaLora`].

    AdaLoRA has three phases defined by `tinit`, `tfinal` and `total_step`.

    The initial phase can be understood as a step for pre-training the adapters so that when reducing their rank, there
    is already some information encoded that can be reduced instead of random matrices. This phase is defined by
    supplying `tinit`.

    After the initial phase is over (`tinit` steps have passed) and the final phase has not begun, AdaLoRA reduces the
    budget of how much rank each layer is allowed to have with each step. This is where the reduction of rank is
    happening. This goes on until `total_step - tfinal` steps are reached.

    The last phase, beginning once `total_step - tfinal` steps are reached, does not change the layer ranks anymore but
    fine-tunes the reduced-rank layers that resulted from the previous phase.

    A practical example: `tinit` is 10, `tfinal` is 20, `total_step` is 100. We spend 10 steps doing pre-training
    without rank reduction because our budget is constant (init phase), then we spend 80 (100-20) steps in the
    reduction phase where our budget decreases step-wise and, finally, 20 steps in the final fine-tuning stage without
    reduction.

    Args:
        target_r (`int`): The target average rank of incremental matrix.
        init_r (`int`): The initial rank for each incremental matrix.
        tinit (`int`): The steps of initial fine-tuning warmup.
        tfinal (`int`): The number of steps of final fine-tuning.
        deltaT (`int`): The time internval between two budget allocations.
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
        orth_reg_weight (`float`): The coefficient of orthogonal regularization.
        total_step (`int`): The total training steps that should be specified before training.
        rank_pattern (`list`): The allocated rank for each weight matrix by RankAllocator.
    """

    target_r: int = field(default=8, metadata={"help": "Target Lora matrix dimension."})
    init_r: int = field(default=12, metadata={"help": "Initial Lora matrix dimension."})
    tinit: int = field(default=0, metadata={"help": "The steps of initial warmup."})
    tfinal: int = field(default=0, metadata={"help": "The steps of final warmup."})
    deltaT: int = field(default=1, metadata={"help": "Step interval of rank allocation."})
    beta1: float = field(default=0.85, metadata={"help": "Hyperparameter of EMA."})
    beta2: float = field(default=0.85, metadata={"help": "Hyperparameter of EMA."})
    orth_reg_weight: float = field(default=0.5, metadata={"help": "The orthogonal regularization coefficient."})
    total_step: Optional[int] = field(default=None, metadata={"help": "The total training steps."})
    rank_pattern: Optional[dict] = field(default=None, metadata={"help": "The saved rank pattern."})

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.ADALORA

        if self.use_dora:
            raise ValueError(f"{self.peft_type} does not support DoRA.")

        if self.loftq_config:
            raise ValueError(f"{self.peft_type} does not support LOFTQ.")

        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )
        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")

        # Check if 'r' has been set to a non-default value
        if self.r != 8:  # 8 is the default value for 'r' in LoraConfig
            warnings.warn(
                "Note that `r` is not used in AdaLora and will be ignored."
                "If you intended to set the initial rank, use `init_r` instead."
            )

        if self.total_step is None or self.total_step <= 0:
            raise ValueError("AdaLoRA does not work when `total_step` is None, supply a value > 0.")

        if self.tinit >= (self.total_step - self.tfinal):
            raise ValueError(
                "The supplied schedule values don't allow for a budgeting phase. Decrease `tfinal`/`tinit` or "
                "increase `total_step`."
            )
