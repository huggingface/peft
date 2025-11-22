"""
=============================================================================
HEAVILY ANNOTATED: LoraConfig.__post_init__() - Configuration Validation
=============================================================================

File Location: src/peft/tuners/lora/config.py (LoraConfig class)

WHAT: Validates LoraConfig parameters after initialization and sets up
      sub-configurations (LoftQConfig, EvaConfig, etc.)

WHY: Catch configuration errors early before attempting injection
     Prevents invalid combinations of parameters

KEY VALIDATIONS:
1. loftq_config properly initialized
2. eva_config properly initialized
3. use_dora incompatibilities
4. target_modules vs target_parameters mutual exclusivity
5. Runtime configuration setup

Source: src/peft/tuners/lora/config.py:250-799
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Union


@dataclass
class LoraConfig(PeftConfig):
    """Configuration for LoRA adapters with extensive validation"""

    # Core parameters (see references/lora_config_reference.md for full list)
    r: int = 8
    lora_alpha: int = 8
    target_modules: Optional[Union[list[str], str]] = None
    lora_dropout: float = 0.0
    # ... (30+ more parameters)

    def __post_init__(self):
        """
        Validate configuration after initialization.

        Called automatically after __init__().
        Performs validation and sets up sub-configs.
        """

        # ===================================================================
        # PARENT VALIDATION
        # ===================================================================
        # Call parent's __post_init__ first
        super().__post_init__()

        # ===================================================================
        # LOFTQ CONFIG VALIDATION
        # ===================================================================
        # WHAT: Ensure loftq_config is proper LoftQConfig instance
        # WHY: User might pass dict instead of LoftQConfig
        # ===================================================================
        if self.loftq_config is not None:
            if not isinstance(self.loftq_config, LoftQConfig):
                # Convert dict to LoftQConfig
                self.loftq_config = LoftQConfig(**self.loftq_config)

            # Validate loftq_config parameters
            if self.loftq_config.loftq_bits not in [2, 4, 8]:
                raise ValueError("loftq_bits must be 2, 4, or 8")

        # ===================================================================
        # EVA CONFIG VALIDATION
        # ===================================================================
        # WHAT: Ensure eva_config is proper EvaConfig instance
        # WHY: EVA needs specific configuration for data-driven init
        # ===================================================================
        if self.init_lora_weights == "eva":
            if self.eva_config is None:
                raise ValueError("eva_config must be provided when using EVA initialization")

            if not isinstance(self.eva_config, EvaConfig):
                # Convert dict to EvaConfig
                self.eva_config = EvaConfig(**self.eva_config)

            # Validate EVA parameters
            if self.eva_config.rho <= 0:
                raise ValueError("EVA rho must be positive")

        # ===================================================================
        # DORA INCOMPATIBILITY CHECKS
        # ===================================================================
        # WHAT: DoRA incompatible with certain features
        # WHY: Technical limitations of DoRA implementation
        # ===================================================================
        if self.use_dora:
            # DoRA + rsLoRA incompatible
            if self.use_rslora:
                raise ValueError("DoRA does not support use_rslora")

            # DoRA + loftq incompatible
            if self.init_lora_weights == "loftq":
                raise ValueError("DoRA does not support loftq initialization")

            # DoRA + rank patterns requires special handling
            if self.rank_pattern:
                # Allowed, but warn
                warnings.warn("DoRA with rank_pattern is experimental")

        # ===================================================================
        # TARGET MODULES VS TARGET PARAMETERS
        # ===================================================================
        # WHAT: Ensure only one targeting method used
        # WHY: Different injection mechanisms, can't use both
        # ===================================================================
        if self.target_modules is not None and self.target_parameters is not None:
            raise ValueError(
                "Cannot specify both target_modules and target_parameters. "
                "Use target_modules for nn.Module, target_parameters for nn.Parameter"
            )

        # ===================================================================
        # RUNTIME CONFIG SETUP
        # ===================================================================
        # WHAT: Initialize runtime-only configuration
        # WHY: Some settings shouldn't be saved with adapter
        # ===================================================================
        if self.runtime_config is None:
            self.runtime_config = LoraRuntimeConfig()
        elif not isinstance(self.runtime_config, LoraRuntimeConfig):
            self.runtime_config = LoraRuntimeConfig(**self.runtime_config)

        # ===================================================================
        # CORDA CONFIG VALIDATION
        # ===================================================================
        if isinstance(self.init_lora_weights, str) and self.init_lora_weights.startswith("corda"):
            if self.corda_config is None:
                raise ValueError("corda_config required for CorDA initialization")
            if not isinstance(self.corda_config, CordaConfig):
                self.corda_config = CordaConfig(**self.corda_config)

        # ===================================================================
        # ARROW CONFIG VALIDATION
        # ===================================================================
        if self.arrow_config is not None:
            if not isinstance(self.arrow_config, ArrowConfig):
                self.arrow_config = ArrowConfig(**self.arrow_config)

            # Validate arrow parameters
            if self.arrow_config.top_k <= 0:
                raise ValueError("Arrow top_k must be positive")


# =============================================================================
# KEY VALIDATION CHECKS
# =============================================================================
# 1. Sub-config Instantiation: dict → Config object
# 2. Incompatibility Checks: DoRA + rsLoRA, etc.
# 3. Required Configs: EVA needs eva_config
# 4. Mutual Exclusivity: target_modules XOR target_parameters
# 5. Parameter Ranges: bits in [2,4,8], rho > 0, etc.
#
# See: references/lora_config_reference.md for all parameters
# =============================================================================
