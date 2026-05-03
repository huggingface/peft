# Copyright 2025-present the HuggingFace Inc. team.
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
class GloraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`GloraModel`].

    Glora modifies a frozen linear layer W0 as:
        W_eff = W0 + W0 * A + B
        b_eff = b0 + b0 * D + E + W0 @ C

    Each matrix (A, B, C, D, E) can be parameterized independently. The config values control
    how many parameters are used and what shapes they can express:

    - `lora`: Low-rank decomposition `Xd @ Xu` with shapes `(out, r)` and `(r, in)`. Uses
      `r * (out + in)` parameters and can express any rank-r correction. Like standard LoRA.
    - `vector`: A single column vector of shape `(out, 1)`, broadcast across the full matrix.
      Uses `out` parameters; only per-output-channel scaling or shifts.
    - `constant`: A single scalar shared across all elements. Uses 1 parameter; most constrained.
    - `none`: Zeros, no trainable parameters. Effectively disables this path.

    Args:
        r (`int`): Rank of the low-rank decomposition used when a config is set to `lora`.
        target_modules (`Optional[Union[List[str], str]]`): The names of the modules to apply Glora to.
        config_A_B (`str`): Parameterization for the A and B matrices (weight multiplicative and additive
            corrections). Valid values: `lora`, `vector`, `constant`, `none`.
        config_C (`str`): Parameterization for the C matrix (weight-to-bias coupling: b += W0 @ C).
            Valid values: `lora`, `vector`, `none`.
        config_D_E (`str`): Parameterization for the D and E scalars (bias multiplicative and additive
            corrections). Does not support `lora` since D and E are bias-sized vectors, not matrices.
            Valid values: `vector`, `constant`, `none`.
        init_weights (`bool`): If True (default), initialize GLoRA as a no-op (zeros). If False,
            use kaiming initialization so the adapter is not a no-op.
    """

    r: int = field(
        default=8, metadata={"help": "Default rank of the LoRA matrices if the config contains lora parametrization."}
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Glora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    bias: str = field(
        default="none",
        metadata={
            "help": (
                "Bias handling: 'none', 'all', or 'glora_only' (train bias on GLoRA layers when adapters are active)."
            )
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from GLoRA layers to be set as trainable and saved in the final checkpoint."
            )
        },
    )

    config_A_B: Literal["lora", "vector", "constant", "none"] = field(
        default="lora",
        metadata={
            "help": (
                "Parameterization for A and B (weight multiplicative and additive corrections: "
                "W_eff = W0 + W0*A + B). "
                "'lora': low-rank Xd@Xu (out x r and r x in), r*(out+in) params, most expressive. "
                "'vector': per-output-channel scalar (out params). "
                "'constant': single scalar (1 param). "
                "'none': disabled (0 params). "
            )
        },
    )

    config_C: Literal["lora", "vector", "none"] = field(
        default="lora",
        metadata={
            "help": (
                "Parameterization for C (weight-to-bias coupling: b += W0 @ C). "
                "'lora': low-rank column Xd@Xu (in x r and r x 1), couples bias to a rank-r projection of W0. "
                "'vector': single (in x 1) column, couples each bias element to a weighted sum of W0 rows. "
                "'none': disabled (0 params). "
            )
        },
    )

    config_D_E: Literal["vector", "constant", "none"] = field(
        default="constant",
        metadata={
            "help": (
                "Parameterization for D and E (bias multiplicative and additive corrections: "
                "b_eff = b0 + b0*D + E). "
                "'vector': per-output-channel value (out params). "
                "'constant': single scalar shared across all bias elements (1 param). "
                "'none': disabled (0 params). "
                "Does not support 'lora' since D and E are bias-length vectors, not matrices. "
            )
        },
    )

    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "If True, initialize GLoRA as a no-op (zeros for down-projection). "
                "If False, use kaiming initialization so the adapter output is not identity."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.GLORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )

        valid_A_B = {"lora", "vector", "constant", "none"}
        valid_C = {"lora", "vector", "none"}
        valid_D_E = {"vector", "constant", "none"}

        if self.config_A_B not in valid_A_B:
            raise ValueError(
                f"Invalid config_A_B value: {self.config_A_B!r}. Valid values are: {', '.join(sorted(valid_A_B))}."
            )
        if self.config_C not in valid_C:
            raise ValueError(
                f"Invalid config_C value: {self.config_C!r}. Valid values are: {', '.join(sorted(valid_C))}."
            )
        if self.config_D_E not in valid_D_E:
            raise ValueError(
                f"Invalid config_D_E value: {self.config_D_E!r}. Valid values are: {', '.join(sorted(valid_D_E))}."
            )
