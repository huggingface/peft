# Copyright 2024-present the HuggingFace Inc. team.
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
from typing import Literal, Optional

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class ShadowConfig(PeftConfig):
    """
    Configuration for the [`ShadowModel`] (ShadowPEFT).

    ShadowPEFT augments a frozen base decoder-only model with a small, trainable parallel *shadow* network. The shadow
    network runs alongside the base model and injects learned corrections into each decoder layer, enabling effective
    adaptation with a small fraction of the parameters. The shadow module can be trained, stored and deployed as a
    standalone component.

    Args:
        num_shadow_layers (`int`):
            Number of transformer layers in the *implicit* shadow model. Ignored when an explicit ``shadow_model`` is
            supplied to [`get_peft_model`].
        injection_hidden_size (`int`):
            Bottleneck dimension of the injection adapter. Larger values are more expressive but add parameters.
        gate_hidden_size (`int`):
            Hidden dimension of the shadow-update gate.
        alpha (`float`):
            Scale factor applied to the injection delta: ``hidden' = hidden + alpha * injection_delta``.
        dropout (`float`):
            Dropout applied inside the injection and update adapters.
        shadow_intermediate_size (`Optional[int]`):
            Override the MLP intermediate size of the implicit shadow model (``None`` = same as the base model).
        shadow_num_attention_heads (`Optional[int]`):
            Override the number of attention heads of the implicit shadow model (``None`` = same as base).
        shadow_num_key_value_heads (`Optional[int]`):
            Override the number of key/value heads (GQA) of the implicit shadow model (``None`` = same as base).
        shadow_head_dim (`Optional[int]`):
            Override the per-head dimension of the implicit shadow model (``None`` = same as base).
        shadow_loss_weight (`float`):
            Weight of the auxiliary shadow-path cross-entropy loss added to the base loss when ``labels`` are passed:
            ``loss = base_loss + shadow_loss_weight * shadow_loss``. Set to ``0`` to disable the auxiliary loss.
        shadow_inference_mode (`Literal["base_shadow", "shadow_only"]`):
            ``"base_shadow"`` (default) returns the base model logits in ``logits`` and the shadow-path logits in
            ``shadow_logits``. ``"shadow_only"`` runs only the lightweight shadow path (no base forward pass) and
            returns the shadow logits in both fields.
        modules_to_save (`Optional[list[str]]`):
            Extra modules to make trainable and persist alongside the adapter. ``"shadow_lm_head"`` /
            ``"shadow_classifier_head"`` toggle training of the shadow task head; any other names refer to base-model
            modules (e.g. ``"lm_head"``, ``"score"``, ``"classifier"``) which will be unfrozen and saved as well.
    """

    num_shadow_layers: int = field(default=1, metadata={"help": "Number of layers in the implicit shadow model."})
    injection_hidden_size: int = field(default=16, metadata={"help": "Bottleneck dimension of the injection adapter."})
    gate_hidden_size: int = field(default=10, metadata={"help": "Hidden dimension of the shadow-update gate."})
    alpha: float = field(default=0.1, metadata={"help": "Scale factor for the injection delta."})
    dropout: float = field(default=0.2, metadata={"help": "Dropout inside the injection/update adapters."})

    shadow_intermediate_size: Optional[int] = field(
        default=None, metadata={"help": "Override implicit shadow MLP intermediate size (None = same as base)."}
    )
    shadow_num_attention_heads: Optional[int] = field(
        default=None, metadata={"help": "Override implicit shadow attention heads (None = same as base)."}
    )
    shadow_num_key_value_heads: Optional[int] = field(
        default=None, metadata={"help": "Override implicit shadow KV heads (None = same as base)."}
    )
    shadow_head_dim: Optional[int] = field(
        default=None, metadata={"help": "Override implicit shadow head dimension (None = same as base)."}
    )

    shadow_loss_weight: float = field(
        default=0.05, metadata={"help": "Weight of the auxiliary shadow-path loss added to the base loss."}
    )
    shadow_inference_mode: Literal["base_shadow", "shadow_only"] = field(
        default="base_shadow",
        metadata={"help": "Inference mode: 'base_shadow' (base + shadow logits) or 'shadow_only'."},
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Extra modules to train/save alongside the shadow adapter."},
    )
    explicit_shadow_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Hub path or local directory of the explicit shadow model used to initialize the shadow backbone "
                "architecture at load time. Fine-tuned shadow weights are stored in the adapter checkpoint."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.SHADOW
        if self.shadow_inference_mode not in ("base_shadow", "shadow_only"):
            raise ValueError(
                f"shadow_inference_mode must be 'base_shadow' or 'shadow_only', got {self.shadow_inference_mode!r}."
            )

    @property
    def is_shadow(self) -> bool:
        return True
