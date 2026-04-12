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
from peft.utils.peft_types import PeftType


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
    """

    _VALID_A_B_CONFIGS = {"lora", "vector", "constant", "none"}
    _VALID_C_CONFIGS = {"lora", "vector", "none"}
    _VALID_D_E_CONFIGS = {"constant", "none", "vector"}

    r: int = field(
        default=8, metadata={"help": "Default rank of the LoRA matrices if the config contains lora parametrization."}
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )

    config_A_B: Literal["lora", "vector", "constant", "none"] = field(
        default="lora",
        metadata={
            "help": (
                "Parameterization for A and B (weight multiplicative and additive corrections: W_eff = W0 + W0*A + B). "
                "'lora': low-rank Xd@Xu (out x r and r x in), r*(out+in) params, most expressive. "
                "'vector': per-output-channel scalar (out params). "
                "'constant': single scalar (1 param). "
                "'none': disabled (0 params). "
                f"Valid values: {', '.join(sorted(_VALID_A_B_CONFIGS))}. 'lora' is post-processed to lora_<rank>."
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
                f"Valid values: {', '.join(sorted(_VALID_C_CONFIGS))}. 'lora' is post-processed to lora_<rank>."
            )
        },
    )

    config_D_E: Literal["vector", "constant", "none"] = field(
        default="constant",
        metadata={
            "help": (
                "Parameterization for D and E (bias multiplicative and additive corrections: b_eff = b0 + b0*D + E). "
                "'vector': per-output-channel value (out params). "
                "'constant': single scalar shared across all bias elements (1 param). "
                "'none': disabled (0 params). "
                "Does not support 'lora' since D and E are bias-length vectors, not matrices. "
                f"Valid values: {', '.join(sorted(_VALID_D_E_CONFIGS))}."
            )
        },
    )

    def _validate_and_process_config(
        self, config_value: str, valid_configs: set, config_name: str, allow_lora: bool = True
    ) -> str:
        """
        Validate and process a configuration value.

        Args:
            config_value: The configuration value to validate
            valid_configs: Set of valid configuration values
            config_name: Name of the configuration (for error messages)
            allow_lora: Whether LoRA configuration is allowed

        Returns:
            Processed configuration value

        Raises:
            ValueError: If the configuration value is invalid
        """
        config_value = str(config_value).lower()

        if config_value and "lora" in config_value:
            if not allow_lora:
                raise ValueError(
                    f"Invalid {config_name} value: {config_value}. lora is not supported for {config_name}."
                )
            return f"lora_{self.r}"

        if config_value not in valid_configs:
            raise ValueError(
                f"Invalid {config_name} value: {config_value}. Valid values are: {', '.join(sorted(valid_configs))}."
            )

        return config_value

    def __post_init__(self):
        self.peft_type = PeftType.GLORA

        # Validate and process each configuration
        self.config_A_B = self._validate_and_process_config(self.config_A_B, self._VALID_A_B_CONFIGS, "config_A_B")  # type: ignore[assignment]

        self.config_C = self._validate_and_process_config(self.config_C, self._VALID_C_CONFIGS, "config_C")  # type: ignore[assignment]

        self.config_D_E = self._validate_and_process_config(  # type: ignore[assignment]
            self.config_D_E, self._VALID_D_E_CONFIGS, "config_D_E", allow_lora=False
        )
