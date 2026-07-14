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
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class ShadowConfig(PeftConfig):
    """
    Configuration class for [`ShadowModel`] (ShadowPEFT).

    ShadowPEFT augments a frozen base decoder-only model with a small, trainable parallel *shadow* network. A shadow
    backbone produces an initial shadow state `s^(0)` that then rides the base model's decoder loop: at every targeted
    block the discrepancy between the base hidden states and the shadow state is injected back into the block input
    (Eq. 2-4), and the shadow state is advanced by a gated residual update from the block output (Eq. 5-7). Only the
    shadow components are trained; the base model stays frozen. Because the adaptation is an input-dependent trajectory
    in layer space rather than a static weight delta, ShadowPEFT cannot be merged into the base weights.

    Args:
        target_modules (`Optional[Union[list[str], str]]`):
            The transformer blocks to wrap with the shadow mechanism (whole decoder blocks, not linear layers). Can be
            a list of module names, or a regex, e.g. `r'.*\\.layers\\.\\d+$'` to target every decoder block. Defaults to
            `None`, in which case every decoder block of the base model is wrapped. Note that the wrapped blocks must be
            contiguous, because the shadow state rides the decoder loop from the first wrapped block to the last.
        exclude_modules (`Optional[Union[list[str], str]]`):
            The names of the modules to not wrap with the shadow mechanism, given as a list or a regex. Defaults to
            `None`.
        r (`int`):
            The rank of the low-rank injection bottleneck `W_down` / `W_up`. Defaults to `8`.
        shadow_alpha (`float`):
            The strength of the injected correction added onto the block input (Eq. 4). Defaults to `1.0`.
        shadow_dropout (`float`):
            The dropout probability applied to the discrepancy signal before the bottleneck (Eq. 3). Defaults to `0.0`.
        init_weights (`bool`):
            Whether to zero-initialize `W_up` so the injection is a no-op at the start of training (mirroring LoRA's
            `B=0` convention). Don't change this unless you know what you are doing. Defaults to `True`.
        shadow_model (`str`):
            How to build the shadow backbone. `"mirror"` (default) builds a fresh backbone of the same architecture as
            the base model but with fewer/smaller layers (see the `shadow_*` overrides below). Any other string is
            treated as a model id or local path: a plain model is loaded with Transformers `AutoModel`, and a
            "projected" shadow checkpoint (`model_type == "causal_lm_with_hidden_projection"`, e.g.
            `shadow-llm/Qwen3-0.6B-H8B`) loads its pretrained backbone together with its trained
            shadow-hidden -> base-hidden projection.
        shadow_num_hidden_layers (`Optional[int]`):
            The number of layers of the auto-built (`"mirror"`) shadow backbone. Defaults to `None` (`1` layer).
        shadow_hidden_size (`Optional[int]`):
            The hidden size of the auto-built shadow backbone; may differ from the base hidden size (a projection is
            inserted automatically). Defaults to `None` (same as the base model).
        shadow_num_attention_heads (`Optional[int]`):
            The number of attention heads of the auto-built shadow backbone. Defaults to `None` (same as the base
            model).
        shadow_intermediate_size (`Optional[int]`):
            The feed-forward width of the auto-built shadow backbone. Defaults to `None` (same as the base model).
        share_embeddings (`bool`):
            Whether to reuse the frozen base input embeddings to feed the shadow backbone (via `inputs_embeds`) instead
            of the shadow backbone's own embedding table. Defaults to `True`.
        update_hidden_size (`Optional[int]`):
            The hidden width of the `T` (candidate) and `G` (gate) update MLPs. Defaults to `None` (uses `r`).
        auxiliary_loss_weight (`float`):
            The weight `lambda` of the auxiliary shadow loss (Eq. 8-9) that is added to the task loss when `labels` are
            passed. Set to `0` to disable it. Defaults to `0.05`.
        layers_to_transform (`Optional[Union[list[int], int]]`):
            The block indices to transform. If a list is passed, the shadow mechanism is applied to the blocks at those
            indices. If a single integer is passed, it is applied at that index only. Defaults to `None` (every matched
            block is transformed).
        layers_pattern (`Optional[Union[list[str], str]]`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`. This is the name of the
            `nn.ModuleList` that holds the decoder blocks (often `"layers"` or `"h"`). Defaults to `None`.
        modules_to_save (`Optional[list[str]]`):
            The extra modules to set as trainable and save in the final checkpoint (e.g. a classifier head). The special
            name `"shadow_lm_head"` unfreezes the (otherwise frozen) copy of the base LM head used for the auxiliary
            loss. Defaults to `None`.
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
    r: int = field(
        default=8, metadata={"help": "Rank of the low-rank injection bottleneck W_down / W_up. Default: 8."}
    )
    shadow_alpha: float = field(
        default=1.0,
        metadata={"help": "Strength of the injected correction added onto the block input (Eq. 4). Default: 1.0."},
    )
    shadow_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout applied to the discrepancy signal before the bottleneck (Eq. 3). Default: 0.0."},
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Zero-initialize W_up so the injection is a no-op at the start of training (default: True). Don't "
                "change this unless you know what you're doing."
            )
        },
    )
    # --- shadow backbone (the model-level component that produces s^(0)) ---
    shadow_model: str = field(
        default="mirror",
        metadata={
            "help": (
                "How to build the shadow backbone: 'mirror' (default) builds the same architecture as the base model; "
                "any other string is loaded as a model id/path with Transformers `AutoModel`."
            )
        },
    )
    shadow_num_hidden_layers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of layers of the auto-built shadow backbone (default: None = 1 layer)."},
    )
    shadow_hidden_size: Optional[int] = field(
        default=None,
        metadata={"help": "Hidden size of the shadow backbone, may differ from the base size (default: None = base)."},
    )
    shadow_num_attention_heads: Optional[int] = field(
        default=None,
        metadata={"help": "Number of attention heads of the auto-built shadow backbone (default: None = base)."},
    )
    shadow_intermediate_size: Optional[int] = field(
        default=None,
        metadata={"help": "Feed-forward width of the auto-built shadow backbone (default: None = base)."},
    )
    share_embeddings: bool = field(
        default=True,
        metadata={"help": "Reuse the frozen base input embeddings in the shadow backbone (default: True)."},
    )
    # --- shadow update networks ---
    update_hidden_size: Optional[int] = field(
        default=None,
        metadata={"help": "Hidden width of the T (candidate) and G (gate) update MLPs (default: None = r)."},
    )
    auxiliary_loss_weight: float = field(
        default=0.05,
        metadata={"help": "Weight lambda of the auxiliary shadow loss (Eq. 8-9), 0 disables it. Default: 0.05."},
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={"help": "Block indices to transform. If omitted, every matched block is transformed (default)."},
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "Name of the nn.ModuleList holding the blocks (often 'layers' or 'h'). Default: None."},
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Extra modules to set as trainable and save in the final checkpoint. Default: None."},
    )

    def __post_init__(self):
        super().__post_init__()
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

    @property
    def is_shadow(self) -> bool:
        return True
