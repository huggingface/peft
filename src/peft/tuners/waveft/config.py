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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType

from .constants import WAVELET_REDUCTIONS


@dataclass
class WaveFTConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`WaveFTModel`]. It is used to define the
    parameters for Wavelet-based Fine-Tuning (WaveFT), an approach that leverages the sparsity of wavelet transforms
    for parameter-efficient fine-tuning of pretrained models.

    Args:
        n_frequency (`int`):
            Number of learnable wavelet coefficients for the Discrete Wavelet Transform (DWT). 'n_frequency' is an
            integer that is greater than 0 and less than or equal to the total number of elements in the original
            weight matrix (d_out * d_in). This parameter directly controls the number of trainable parameters for each
            adapted layer. A higher 'n_frequency' generally leads to better performance but also increases GPU memory
            usage, with a minor impact on training speed.
        scaling (`float`):
            The scaling factor applied to the reconstructed delta W matrix. This is a crucial hyperparameter, analogous
            to `lora_alpha` in LoRA. It can be tuned during hyperparameter search. Our default value for SDXL
            personalization is 25.
        wavelet_family (`str`):
            The wavelet family (e.g., 'db1', 'sym2', 'coif1') to use for the DWT and Inverse DWT (IDWT). Defaults to
            'db1' (Haar wavelet). Different wavelet families have varying filter lengths which affect the training time
            substantially
        use_idwt (`bool`):
            Set to False for efficient adaptation. Whether to use the Inverse Discrete Wavelet Transform (IDWT) to
            reconstruct the delta weights from the learned wavelet coefficients. If `True` (default), the IDWT is
            applied. If `False`, the learned coefficients are directly used to form a sparse delta weight matrix, which
            is faster but performs worse for the SDXL personalization task.
        random_loc_seed (`int`):
            Seed for determining the random locations of the `n_frequency` learnable wavelet coefficients within the
            full wavelet coefficient matrix.
        target_modules (`Union[list[str],str]`):
            List of module names or a regex expression identifying the modules to be adapted with WaveFT. For example,
            `['q_proj', 'v_proj']` or `'.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'`. Currently, only linear
            layers (`torch.nn.Linear`) are supported.
        exclude_modules (`Optional[Union[List[str], str]]`):
            List of module names or a regex expression for modules to exclude from WaveFT adaptation.
        fan_in_fan_out (`bool`):
            Set to `True` if the weights of the layer to be replaced are stored in `(fan_in, fan_out)` format. Default
            is `False`.
        bias (`str`):
            Bias type for WaveFT. Can be 'none', 'all', or 'waveft_only'. ('fourier_only' was likely a typo and has
            been corrected to 'waveft_only' if it implies bias only on adapted parameters) If 'waveft_only', biases are
            added only to the WaveFT components. If 'all', biases are added to both base and WaveFT components. If
            'none', no new biases are added.
        modules_to_save (`list[str]`):
            List of modules, in addition to WaveFT layers, that should be marked as trainable and saved in the final
            checkpoint. Useful for layers like classifiers in sequence or token classification tasks that are randomly
            initialized and need training.
        layers_to_transform (`Union[list[int],int]`):
            Specific layer indices to transform. If provided, PEFT will only adapt layers at these indices. If a single
            integer is given, only that layer is transformed.
        layers_pattern (`Optional[Union[List[str], str]]`):
            Pattern for layer names, used if `layers_to_transform` is specified and the layer pattern is not standard
            (e.g., not 'layers' or 'h'). This should target the `nn.ModuleList` attribute in the model.
        n_frequency_pattern (`dict`):
            A dictionary mapping layer names (or regex) to specific `n_frequency` values, overriding the global
            `n_frequency`. Example: `{"model.decoder.layers.0.encoder_attn.k_proj": 1000}`.
        init_weights (`bool`):
            Initialization strategy for the learnable wavelet coefficients (spectrum). If `True` (default),
            coefficients are initialized to zeros. If `False`, coefficients are initialized from a standard normal
            distribution scaled by a small factor.
        proportional_parameters (`bool`):
            If `True`, `n_frequency` is allocated proportionally to each layer's `input_dim * output_dim`. Default is
            `False`. Note: This option is included for experimental thoroughness to allow researchers to reproduce
            paper results, rather than for practical utility, as no beneficial scenarios have been identified.
    """

    n_frequency: int = field(
        default=2592,  # Default value might need adjustment based on common use cases or paper findings
        metadata={
            "help": (
                "Number of learnable wavelet coefficients for the Discrete Wavelet Transform (DWT). "
                "'n_frequency' is an integer that is greater than 0 and less than or equal to the "
                "total number of elements in the original weight matrix (d_out * d_in). "
                "This parameter directly controls the number of trainable parameters for each adapted layer. "
                "A higher 'n_frequency' generally leads to better performance but also increases "
                "GPU memory usage, with a minor impact on training speed."
            )
        },
    )
    scaling: float = field(
        default=25.0,  # Default value seems low based on typical examples, might need adjustment
        metadata={
            "help": (
                "The scaling factor applied to the reconstructed delta W matrix. This is a crucial "
                "hyperparameter, analogous to 'lora_alpha' in LoRA. It can be tuned during hyperparameter "
                "search. Default value for SDXL personalization is 25. "
            )
        },
    )
    wavelet_family: str = field(
        default="db1",
        metadata={
            "help": (
                "The wavelet family (e.g., 'db1', 'sym2', 'coif1') to use for the DWT and Inverse DWT (IDWT). "
                "Defaults to 'db1' (Haar wavelet). Different wavelet families have varying filter lengths "
                "which affect the training time substantially. Size differences are handled automatically "
                "if use_idwt is True."
            )
        },
    )
    use_idwt: bool = field(
        default=True,
        metadata={
            "help": (
                "Set to False for efficient adaptation. "
                "Whether to use the Inverse Discrete Wavelet Transform (IDWT) to reconstruct the delta "
                "weights from the learned wavelet coefficients. If True (default), the IDWT is applied. "
                "If False, the learned coefficients are directly used to form a sparse delta weight matrix, "
                "which is faster but performs worse for the SDXL personalization task."
            )
        },
    )
    random_loc_seed: int = field(
        default=777,
        metadata={
            "help": (
                "Seed for determining the random locations of the 'n_frequency' learnable wavelet "
                "coefficients within the full wavelet coefficient matrix."
            )
        },
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={
            "help": (
                "Set to True if the weights of the layer to be replaced are stored in (fan_in, fan_out) "
                "format. Default is False."
            )
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or a regex expression identifying the modules to be adapted with WaveFT. "
                "For example, ['q_proj', 'v_proj'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "Currently, only linear layers (torch.nn.Linear) are supported."
            )
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex for modules to exclude from WaveFT adaptation."},
    )
    bias: str = field(
        default="none",
        metadata={
            "help": (
                "Bias type for WaveFT. Can be 'none', 'all', or 'waveft_only'. "
                "If 'waveft_only', biases are added only to the WaveFT components. "
                "If 'all', biases are added to both base and WaveFT components. "
                "If 'none', no new biases are added."
            )
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules, in addition to WaveFT layers, that should be marked as trainable "
                "and saved in the final checkpoint. Useful for layers like classifiers in sequence "
                "or token classification tasks that are randomly initialized and need training."
            )
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "Specific layer indices to transform. If provided, PEFT will only adapt layers at these "
                "indices. If a single integer is given, only that layer is transformed."
            )
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "Pattern for layer names, used if `layers_to_transform` is specified and the layer "
                "pattern is not standard (e.g., not 'layers' or 'h'). This should target the "
                "`nn.ModuleList` attribute in the model."
            )
        },
    )
    n_frequency_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "A dictionary mapping layer names (or regex) to specific `n_frequency` values, "
                'overriding the global `n_frequency`. Example: {"model.decoder.layers.0.encoder_attn.k_proj": 1000}.'
            )
        },
    )
    proportional_parameters: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, 'n_frequency' is allocated proportionally to each layer's "
                "input_dim * output_dim. Default is False. Note: This option is included "
                "for experimental thoroughness to allow researchers to reproduce paper results, "
                "rather than for practical utility, as no beneficial scenarios have been identified."
            )
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Initialization strategy for the learnable wavelet coefficients (spectrum). "
                "If True (default), coefficients are initialized to zeros. "
                "If False, coefficients are initialized from a standard normal distribution scaled by a small factor."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.WAVEFT
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )
        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")
        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")

        if self.wavelet_family not in WAVELET_REDUCTIONS:
            raise ValueError(
                f"Wavelet family {self.wavelet_family} not supported. Supported wavelet families are: {list(WAVELET_REDUCTIONS.keys())}"
            )
