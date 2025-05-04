# Copyright 2025-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Union, Sequence

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class UIOrthoLoRAConfig(PeftConfig):
    """
    Configuration for [`UIOrthoLoRAModel`].

    Args:
        sigma_regularization (`float` or `Sequence[float]`, *optional*, defaults to `1e-3`):
            Diagonal values used to regularize Σ (see Eq. ?? in your notes).
        num_of_svectors_to_adapt (`Union[int, Sequence[int]]`, *optional*, defaults to `8`):
            How many singular *vectors* per adapted layer are trainable (k in the paper).
        num_of_svalues_to_adapt (`Union[int, Sequence[int]]`, *optional*, defaults to `4`):
            How many smallest singular *values* per adapted layer are trainable (m ≤ k).
        scaling_factor (`float`, *optional*, defaults to `1.0`):
            Overall multiplier on the generated adapter patch ΔW.
        E_init_value / D_init_value / adapter_init_value (`float`):
            Tiny constants for initialising the learned diagonals and Σ entries.
        layers_to_transform (`Optional[Union[Sequence[int], int]]`):
            Indices of the backbone’s `nn.ModuleList` (often `'layers'` or `'h'`) that
            should receive the UIOrthoLoRA transformation.
        layers_pattern (`Optional[Union[Sequence[str], str]]`):
            If your model’s layer list has a non-standard name, set it here.
        enforce_sv_positive (`bool`):
            If *True* the adapted singular values are rectified (ReLU) to stay ≥0.
        fan_in_fan_out (`bool`):
            Set *True* if weight matrices are stored as **(in, out)** (GPT-2 style).
        bias (`str`):
            Same semantics as VeRA / LoRA: `'none'`, `'all'`, `'uiortholora_only'`.
    """

    sigma_regularization: Union[float, Sequence[float]] = field(
        default=1e-3, metadata={"help": "Regularisation term for Σ diagonal."}
    )
    num_of_svectors_to_adapt: Union[int, Sequence[int]] = field(
        default=8, metadata={"help": "Number of singular vectors adapted per layer."}
    )
    num_of_svalues_to_adapt: Union[int, Sequence[int]] = field(
        default=4, metadata={"help": "Number of singular values adapted per layer."}
    )
    scaling_factor: float = field(default=1, metadata={"help": "ΔW scaling factor."})
    E_init_value: float = field(default=1, metadata={"help": "Init for left diag."})
    D_init_value: float = field(default=1, metadata={"help": "Init for right diag."})
    adapter_init_value: float = field(
        default=1e-7, metadata={"help": "Init for trainable Σ entries."}
    )
    layers_to_transform: Optional[Union[Sequence[int], int]] = field(
        default=None,
        metadata={
            "help": "Layer indices to transform. If None, every Linear is adapted."
        },
    )
    layers_pattern: Optional[Union[Sequence[str], str]] = field(
        default=None,
        metadata={
            "help": "ModuleList attribute that holds the layers (e.g. 'layers', 'h')."
        },
    )
    enforce_sv_positive: bool = field(
        default=False,
        metadata={"help": "Clamp adapted singular values to be non-negative."},
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={
            "help": "Set True if weight layout is (fan_in, fan_out) instead of (out, in)."
        },
    )
    bias: str = field(
        default="none",
        metadata={
            "help": "Bias handling: 'none', 'all', or 'uiortholora_only' (train only ΔW bias)."
        },
    )

    # --- <optional housekeeping fields for parity with other tuners> ---
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "Extra modules (e.g. classifier) to keep trainable & save alongside adapters."
            )
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "If False, skip adapter weight init (assumes weights already loaded)."
            )
        },
    )

    # --------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()
        # Register the new peft_type (extend PeftType Enum beforehand)
        self.peft_type = PeftType.OTHER  # or add PeftType.UIORTHOLORA
        # Ensure list→set conversion for fast membership tests
        if isinstance(self.layers_to_transform, list):
            self.layers_to_transform = set(self.layers_to_transform)

        # Sanity checks
        if self.layers_pattern and self.layers_to_transform is None:
            raise ValueError("When `layers_pattern` is set, `layers_to_transform` must also be provided.")

        if isinstance(self.sigma_regularization, (list, tuple)) and any(s <= 0 for s in self.sigma_regularization):
            warnings.warn("Some sigma_regularization values are non-positive; this may cause numerical issues.")
