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
    TODO : add docstring
    """

    prefix: str = field(default="lora_")
    # Add fields for the user to specify module types
    row_sharded_modules: Optional[list[str]] = field(default=None, metadata={"help": "TODO"})
    column_sharded_modules: Optional[list[str]] = field(default=None, metadata={"help": "TODO"})
    nblocks: int = field(default=None, metadata={"help": "TODO"})

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.BDLORA
