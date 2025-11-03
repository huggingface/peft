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

from peft.utils import PeftType

from ..lora import LoraConfig


@dataclass
class BdLoraConfig(LoraConfig):
    """
    This configuration is used to instantiate a BD-LoRA adapter, which is a variant of LoRA proposed by Wang et al.
    2025, https://openreview.net/forum?id=1cjLvtFOmL. In BD-LoRA, some LoRA factors are saved as a block-diagonal
    matrix.

    In the paper, this block-diagonal structure is leveraged to allow for a faster multi-LoRA serving strategy on
    multiple GPUs by to eliminating communication overheads.

    BD-LoRA is intended to be served with the megatron sharding strategy. If we take the Llama architecture as an
    example, then we would shard the q,k,v, and up projections in a column-parallel manner, while setting the LoRA-B
    factor to be block-diagonal. Similarly, the down and out projections would be sharded in a row-parallel manner,
    with the LoRA-A factor being block-diagonal.

    At the same time, this LoRA variant is not tied to a specific sharding strategy and can be used in a general way to
    create LoRA adapters with block-diagonal factors.
    """

    prefix: str = field(default="lora_")
    lora_a_is_blockdiagonal: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "Modules where the LoRA-A is block-diagonal. Matches each pattern in the list against the module name via `pattern is in target_name`. Example: ['up_proj', 'q_proj', 'v_proj', 'k_proj']"
        },
    )
    lora_b_is_blockdiagonal: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "Modules where the LoRA-B is block-diagonal. Matches each pattern in the list against the module name via `pattern is in target_name`. Example: ['out_proj', 'down_proj']"
        },
    )
    nblocks: int = field(
        default=1,
        metadata={
            "help": "Number of blocks each block-diagonal matrix has. If using BD-LoRA to speed up inference, set it to be equal to the desired sharding degree during serving."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.BDLORA
