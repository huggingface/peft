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
from typing import Literal, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class PSOFTConfig(PeftConfig):
    """
    Configuration for PSOFT (Efficient Orthogonal Fine-Tuning with Principal Subspace Adaptation).

    PSOFT inserts an r*r orthogonal transformation R between LoRA A and B,
    so the low-rank update is ΔW = B @ (R-I) @ A. Only R (and optional tunable vectors) are trained;
    A and B are initialized with psoft_init (SVD-based, row-orthogonal A) and frozen.

    Args:
        r (`int`): PSOFT rank, which controls the adapter capacity via an r*r transformation R).
        target_modules: Module names to apply PSOFT to.
        psoft_alpha (`int`): Scaling factor related (same as LoRA alpha).
        psoft_dropout (`float`): Dropout for the adapter path.
        fan_in_fan_out (`bool`): Set True if the layer stores weight as (fan_in, fan_out).
        init_psoft_weights (`Literal["psoft_init", "pissa_init"]`):
            - `"psoft_init"`: SVD-based initialization with row-orthogonal A (recommended).
            - `"pissa_init"`: SVD-based initialization with symmetrical A and B (Standard PiSSA).
        psoft_svd (`Literal["full", "lowrank"]`):
            - `"full"`: uses torch.linalg.svd.
            - `"lowrank"`: torch.svd_lowrank (may be faster for large matrices / large models).
        psoft_svd_lowrank_niter (`int`): Number of power iterations for torch.svd_lowrank when psoft_svd='lowrank'.
        psoft_orth (`bool`):
            If True, R is constrained to be orthogonal via Cayley parameterization,
            which helps preserve the geometric relationships among the pre-trained weight vectors.
            If False, R is a free r*r matrix without orthogonality constraints.
        psoft_mag_b (`bool`):
            If True, learns a diagonal scaling vector on the output side of R.
            Typically used together with psoft_mag_a to improve task adaptability,
            at the cost of slightly distorting the geometric structure of the pre-trained weight space.
        psoft_mag_a (`bool`):
            If True, learns a diagonal scaling vector on the input side of R.
            Usually paired with psoft_mag_b to improve task adaptability,
            at the cost of slightly distorting the geometric structure of the pre-trained weight space.
        use_cayley_neumann (`bool`): Use Cayley-Neumann series for orthogonal R (faster for large ranks).
        num_cayley_neumann_terms (`int`): Number of terms in Cayley-Neumann series, default is 5.
        cayley_neumann_eps (`float`):
            Upper bound on `||Q||_F`. If exceeded, `Q` is rescaled to this value to stabilize the
            truncated Cayley-Neumann approximation, typically improving convergence and numerical stability.
        init_weights (`bool`): Non-zero initialization of R.
        modules_to_save: Additional module names to set trainable and save.
    """

    r: int = field(
        default=32,
        metadata={
            "help": (
                "PSOFT rank (low-rank dimension), which controls the adapter capacity via an r*r transformation R."
                "For simple tasks, small ranks 32-128 would be recommanded. For more complex tasks, large ranks 64-256 would be recommanded."
                "Small ranks may underfit complex tasks, while larger ranks increase expressiveness at the cost of additional parameters and computation."
                "For typical tasks, recommended rank values can be found in the paper: https://openreview.net/forum?id=FSHrinMArK"
            )
        },
    )

    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex to replace with PSOFT. Same semantics as LoRA."},
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex to exclude from PSOFT."},
    )
    psoft_alpha: int = field(default=32, metadata={"help": "PSOFT scaling factor."})
    psoft_dropout: float = field(default=0.0, metadata={"help": "Dropout for PSOFT path."})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set True if the layer stores weight like (fan_in, fan_out)."},
    )
    init_psoft_weights: Literal["psoft_init", "pissa_init"] = field(
        default="psoft_init",
        metadata={
            "help": (
                "Initialization method for PSOFT."
                "- 'psoft_init': SVD-based initialization with row-orthogonal A (recommended)."
                "- 'pissa_init': SVD-based initialization with symmetrical A and B (standard PiSSA)."
            )
        },
    )
    psoft_svd: Literal["full", "lowrank"] = field(
        default="full",
        metadata={
            "help": "SVD backend used for init: 'full' uses torch.linalg.svd, 'lowrank' uses torch.svd_lowrank."
        },
    )
    psoft_svd_lowrank_niter: int = field(
        default=10,
        metadata={"help": "Number of power iterations for torch.svd_lowrank when psoft_svd='lowrank'."},
    )
    psoft_orth: bool = field(
        default=True,
        metadata={
            "help": (
                "If True, R is constrained to be orthogonal via Cayley parameterization, "
                "which helps preserve the geometric relationships among the pre-trained weight vectors. "
                "If False, R is a free matrix without orthogonality constraints."
            )
        },
    )

    psoft_mag_b: bool = field(
        default=True,
        metadata={
            "help": (
                "If True, learns a diagonal scaling vector on the output side of R. "
                "Typically used together with psoft_mag_a to improve task adaptability, "
                "at the cost of slightly distorting the geometric structure of the pre-trained weight space."
            )
        },
    )

    psoft_mag_a: bool = field(
        default=True,
        metadata={
            "help": (
                "If True, learns a diagonal scaling vector on the input side of R. "
                "Usually paired with psoft_mag_b to improve task adaptability, "
                "at the cost of slightly distorting the geometric structure of the pre-trained weight space."
            )
        },
    )
    use_cayley_neumann: bool = field(
        default=False,
        metadata={"help": "Use Cayley-Neumann series for orthogonal R (faster, approximate orthogonality)."},
    )
    num_cayley_neumann_terms: int = field(
        default=5,
        metadata={"help": "Number of terms in Cayley-Neumann series. Higher = better orthogonality."},
    )
    cayley_neumann_eps: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "The Neumann-series approximation is most reliable when the generator matrix `Q` has a sufficiently small"
                "norm (heuristically, `||Q|| < 1`). This parameter enforces that condition by projecting `Q` onto an"
                "`||Q||_F <= cayley_neumann_eps` ball: when `||Q||_F` exceeds the threshold, `Q` is uniformly scaled down."
                "Larger values apply weaker shrinkage; smaller values apply stronger shrinkage, typically improving"
                "convergence and numerical stability at the cost of smaller update magnitude."
                "Details please refer to https://spherelab.ai/oftv2/"
            )
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "List of modules apart from PSOFT layers to set as trainable and save."},
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the PSOFT layers with their default initialization. Don't change ",
                "this setting, except if you know exactly what you're doing.",
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={"help": "Layer indices to transform. Same semantics as LoRA."},
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "Layer pattern name when using layers_to_transform."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.PSOFT
