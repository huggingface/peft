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

from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class SupertuningConfig(PeftConfig):
    """Configuration class for [`SupertuningModel`].

    Super-Tuning (arXiv:2607.09287) freezes the base weight and trains only a sparse support of scalar entries,
    selected by weight magnitude. Setting ``r`` additionally allocates a LoRA-style low-rank adapter on top of the
    sparse support (the paper's "Supra" hybrid).

    The default (magnitude scoring, ``r=None``) reproduces the paper's best-reported single-mechanism configuration:
    on Meta-Llama-3-8B, ``magnitude-topk`` at 79.02% average beats ``Wanda`` at 78.66% AND requires no calibration
    pass.

    Args:
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. String → regex match; list → suffix / exact match; None
            → model-architecture default.
        modules_to_save (`Optional[List[str]]`):
            Modules outside the Supertuning layers that should also be trainable and saved in the final checkpoint
            (e.g. randomly-initialized classifier heads).
        sparsity (`float`):
            Target sparsity ratio in ``[0.0, 1.0)``. ``0.9`` = 10% of weight entries are trainable. Defaults to ``0.5``.
        selection_direction (`str`):
            Which end of the magnitude score to keep as the trainable support. ``"top"`` keeps the largest-magnitude
            entries (paper's Super / Supra); ``"bottom"`` keeps the smallest (paper's ``-bottom`` variants). Paper
            reports the best direction is model- and task-dependent. Defaults to ``"top"``.
        r (`Optional[int]`):
            LoRA rank for the "Supra" hybrid. When ``None`` (default), pure Super — sparse support only. When set to
            a positive integer, additionally allocates LoRA ``A`` (``[r, in_features]``) and ``B``
            (``[out_features, r]``) parameters whose contribution is added to the sparse support in the forward pass.
        lora_alpha (`Optional[float]`):
            LoRA scaling factor for Supra mode. If ``None`` and ``r`` is set, defaults to ``2 * r``. Ignored when
            ``r is None``.
        lora_dropout (`float`):
            LoRA dropout probability for Supra mode. Defaults to ``0.0``. Ignored when ``r is None``.
        init_weights (`bool`):
            Whether to initialize the trainable sparse values to zero (an identity update) at construction. Defaults
            to ``True``. LoRA ``A`` uses Kaiming-uniform, ``B`` uses zero (standard LoRA init) regardless.

    Paper: https://arxiv.org/abs/2607.09287
    """

    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "Module names or regex to replace with Supertuning layers. E.g. ['q_proj', 'v_proj'] or "
                "'.*self_attn.*(q_proj|v_proj)$'. None uses the model-architecture default."
            ),
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "Modules outside Supertuning layers to set as trainable and save in the final checkpoint "
                "(e.g. randomly-initialized classifier heads in sequence-classification tasks)."
            ),
        },
    )
    sparsity: float = field(
        default=0.5,
        metadata={
            "help": (
                "Target sparsity ratio in [0.0, 1.0). E.g. 0.9 = 10% of weight entries are trainable."
            ),
        },
    )
    selection_direction: str = field(
        default="top",
        metadata={
            "help": (
                "Which end of the magnitude score to keep as the trainable support. 'top' keeps the "
                "largest-magnitude entries (paper's Super/Supra); 'bottom' keeps the smallest ('-bottom' variants)."
            )
        },
    )
    r: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "LoRA rank for the 'Supra' hybrid. When None (default), pure Super — sparse support only. When set, "
                "additionally allocates LoRA A/B parameters composed additively with the sparse support."
            )
        },
    )
    lora_alpha: Optional[float] = field(
        default=None,
        metadata={"help": "LoRA scaling for Supra mode. Defaults to 2*r when r is set. Ignored when r is None."},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "LoRA dropout probability for Supra mode. Ignored when r is None."},
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the trainable sparse values to zero (identity update). LoRA A/B init is "
                "unaffected — Kaiming-uniform for A, zeros for B, matching the LoRA convention."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.SUPERTUNING
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )

        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError(f"sparsity must be in [0.0, 1.0), got {self.sparsity}")

        if self.selection_direction not in ("top", "bottom"):
            raise ValueError(f"selection_direction must be 'top' or 'bottom', got {self.selection_direction!r}")

        if self.r is not None:
            if not isinstance(self.r, int) or self.r <= 0:
                raise ValueError(f"r must be a positive integer or None, got {self.r!r}")
            if self.lora_alpha is None:
                self.lora_alpha = 2 * self.r
        else:
            if self.lora_alpha is not None:
                raise ValueError("lora_alpha is set but r is None — Supra mode requires an explicit r.")

        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError(f"lora_dropout must be in [0.0, 1.0), got {self.lora_dropout}")
