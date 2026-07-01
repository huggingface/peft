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

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class ShadowConfig(PeftConfig):
    """
    Configuration class for [`ShadowModel`].

    TODO
    """

    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "Transformer blocks to wrap with the shadow mechanism (whole blocks, not linear layers).",
            "example": r"For example, '.*\.layers\.\d+$' to target every decoder block.",
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex of module names to exclude from the shadow mechanism."},
    )
    r: int = field(default=8, metadata={"help": "Rank of the low-rank injection bottleneck W_down / W_up."})
    shadow_alpha: float = field(
        default=1.0, metadata={"help": "Strength of the injected correction added onto the block input (Eq. 4)."}
    )
    shadow_dropout: float = field(
        default=0.0, metadata={"help": "Dropout applied to the discrepancy signal before the bottleneck (Eq. 3)."}
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Zero-initialize W_up so the injection is a no-op at the start of training. Don't change this unless "
                "you know what you're doing."
            )
        },
    )
    # --- shadow backbone (the model-level component that produces s^(0)) ---
    shadow_model: Literal["auto"] | str = field(
        default="mirror",
        metadata={
            "help": (
                "How to build the shadow backbone: 'auto' means that the same architecture as the base model is being "
                "used; if another string is passed, load that model with Transformers `AutoModel`."
            )
        },
    )
    shadow_num_hidden_layers: Optional[int] = field(
        default=None, metadata={"help": "Number of layers L_s of the auto-built shadow backbone."}
    )
    shadow_hidden_size: Optional[int] = field(
        default=None, metadata={"help": "Hidden size d_s of the shadow backbone (may differ from the base size d)."}
    )
    shadow_num_attention_heads: Optional[int] = field(
        default=None, metadata={"help": "Number of attention heads of the auto-built shadow backbone."}
    )
    shadow_intermediate_size: Optional[int] = field(
        default=None, metadata={"help": "Feed-forward width of the auto-built shadow backbone."}
    )
    share_embeddings: bool = field(
        default=True, metadata={"help": "Reuse the frozen base input embeddings in the shadow backbone."}
    )
    # --- shadow update networks ---
    update_hidden_size: Optional[int] = field(
        default=None, metadata={"help": "Hidden width of the T (candidate) and G (gate) update MLPs. Defaults to r."}
    )
    auxiliary_loss_weight: float = field(
        default=0.05, metadata={"help": "Weight lambda of the auxiliary shadow loss (Eq. 8-9). 0 disables it."}
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None, metadata={"help": "Block indices to transform. If omitted, every matched block is transformed."}
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "Name of the nn.ModuleList holding the blocks (often 'layers' or 'h')."},
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Extra modules to set as trainable and save in the final checkpoint (e.g. a classifier)."},
    )

    def __post_init__(self):
        super().__post_init__()
        # NOTE: boilerplate -- a real integration must add `SHADOW` to the `PeftType` enum and register the method.
        self.peft_type = PeftType.SHADOW
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )

        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified.")
