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
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils.peft_types import PeftType


@dataclass
class GLoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`GLoraModel`].

    Args:
        r (`int`): GLora attention dimension (rank of the LoRA matrices).
        target_modules (`Optional[Union[List[str], str]]`): The names of the modules to apply GLora to.
        config_A_B (`str`): Configuration for A and B matrices. Valid values: 'LoRA', 'vector', 'constant', 'none'.
        config_C (`str`): Configuration for C matrix. Valid values: 'LoRA', 'vector', 'none'.
        config_D_E (`str`): Configuration for D and E matrices. Valid values: 'constant', 'none', 'vector'.
    """

    _VALID_A_B_CONFIGS = {"LoRA", "vector", "constant", "none"}
    _VALID_C_CONFIGS = {"LoRA", "vector", "none"}
    _VALID_D_E_CONFIGS = {"constant", "none", "vector"}

    r: int = field(
        default=4, metadata={"help": "Default rank of the LoRA matrices if the config contains LoRA parametrization."}
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )

    config_A_B: str = field(
        default="LoRA",
        metadata={
            "help": "Configuration for A and B matrices in GLora."
            f"Valid values: {', '.join(_VALID_A_B_CONFIGS)}. "
            "For LoRA, it will be post-processed to LoRA_<rank>."
        },
    )

    config_C: str = field(
        default="LoRA",
        metadata={
            "help": "Configuration for C matrix in GLora."
            f"Valid values: {', '.join(_VALID_C_CONFIGS)}. "
            "For LoRA, it will be post-processed to LoRA_<rank>."
        },
    )

    config_D_E: str = field(
        default="constant",
        metadata={
            "help": f"Configuration for D and E matrices in GLora. Valid values: {', '.join(_VALID_D_E_CONFIGS)}."
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
        if config_value and "LoRA" in config_value:
            if not allow_lora:
                raise ValueError(
                    f"Invalid {config_name} value: {config_value}. LoRA is not supported for {config_name}."
                )
            return f"LoRA_{self.r}"

        if config_value not in valid_configs:
            raise ValueError(
                f"Invalid {config_name} value: {config_value}. Valid values are: {', '.join(sorted(valid_configs))}."
            )

        return config_value

    def __post_init__(self):
        self.peft_type = PeftType.GLORA

        # Validate and process each configuration
        self.config_A_B = self._validate_and_process_config(self.config_A_B, self._VALID_A_B_CONFIGS, "config_A_B")

        self.config_C = self._validate_and_process_config(self.config_C, self._VALID_C_CONFIGS, "config_C")

        self.config_D_E = self._validate_and_process_config(
            self.config_D_E, self._VALID_D_E_CONFIGS, "config_D_E", allow_lora=False
        )
