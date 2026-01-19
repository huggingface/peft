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
class AdamssConfig(PeftConfig):
    """
    Configuration class for Adamss (Adaptive Multi-Subspaces) method.

    AdaMSS is a parameter-efficient fine-tuning method that decomposes weight matrices
    using SVD and clusters the decomposed space into multiple trainable subspaces.
    It learns low-rank updates within these subspaces while keeping the original weights frozen.

    Args:
        r (`int`):
            Total rank for SVD decomposition (denoted as R in the paper). This determines
            how many singular vectors are used to represent the weight matrix before clustering.
            Higher values capture more information from the original weights but require more
            computation and memory. Lower values provide stronger regularization.
            Typical values range from 50 to 500. Default is 100.

        num_subspaces (`int`):
            Number of subspaces (K) to cluster the SVD-decomposed space into. Each subspace
            learns independent low-rank updates. Increasing this value allows finer-grained
            adaptation but increases the number of trainable parameters proportionally.
            When using ASA (Adaptive Subspace Allocation), this determines the initial number
            of subspaces before pruning. Typical values range from 3 to 10. Default is 5.

        subspace_rank (`int`):
            The rank (r_i) for each trainable subspace. This controls the capacity of each
            subspace to learn adaptations. Higher values increase expressiveness but also
            increase trainable parameters. Total trainable parameters scale as
            O(num_subspaces * subspace_rank * (in_dim + out_dim) / num_subspaces).
            For most tasks, values of 1-4 work well. Default is 1.

        target_modules (`Optional[Union[list[str], str]]`):
            The names of the modules to apply AdaMSS to. If specified, only these modules
            will be adapted. Can be a list of exact module names or a regex expression.
            For example, `['q_proj', 'v_proj']` for attention layers, or
            `'.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'` for regex matching.

        modules_to_save (`Optional[list[str]]`):
            List of modules apart from AdaMSS layers to be set as trainable and saved in
            the final checkpoint. These modules will be fully fine-tuned (not just low-rank).
            Required for randomly initialized heads like `classifier` or `score` in
            classification tasks.

        init_weights (`Literal["orthogonal"]`):
            Initialization method for AdaMSS trainable weights. Currently only "orthogonal"
            is supported, which uses orthogonal initialization for the B matrices (output
            projection). The A matrices are initialized to zero to ensure the model starts
            from the pretrained weights. Set to None to skip initialization when loading
            from a checkpoint. Default is "orthogonal".

        layers_to_transform (`Optional[Union[list[int], int]]`):
            Specific layer indices to apply AdaMSS to. If specified, only these layers
            will be adapted, useful for experimenting with which layers benefit most from
            adaptation. Can be a single integer or a list of integers.

        layers_pattern (`Optional[Union[list[str], str]]`):
            Pattern to match layer names when `layers_to_transform` is specified. Used to
            extract layer indices from module names that don't follow the common pattern.

        use_asa (`bool`):
            Whether to enable Adaptive Subspace Allocation (ASA). When enabled, ASA
            dynamically prunes less important subspaces during training based on gradient
            information, reducing the effective number of parameters while maintaining
            performance. Requires integration with a training callback. Default is False.

        asa_target_subspaces (`int`):
            Target total number of active subspaces across all layers when ASA is enabled.
            ASA will progressively prune subspaces until this target is reached. Lower
            values result in more aggressive pruning and fewer trainable parameters.
            Should be less than `num_subspaces * num_target_modules`. Typical values
            range from 20 to 100 depending on model size. Default is 50.

        init_warmup (`int`):
            Number of training steps to wait before starting ASA pruning. During warmup,
            all subspaces remain active to allow importance scores to stabilize. Higher
            values give more time for accurate importance estimation but delay pruning.
            Typical values range from 50 to 200. Default is 50.

        final_warmup (`int`):
            Training step at which ASA completes pruning and reaches `asa_target_subspaces`
            active subspaces. The pruning is distributed between `init_warmup` and
            `final_warmup`. Should be set based on total training steps; typically 1/3
            to 1/2 of total training steps. Default is 1000.

        mask_interval (`int`):
            Number of training steps between ASA mask updates. Lower values allow more
            frequent adaptation but increase overhead. Higher values provide more stable
            importance estimates between updates. Typical values range from 50 to 200.
            Default is 100.

        asa_importance_beta (`float`):
            Exponential moving average (EMA) coefficient for smoothing subspace importance
            scores. Higher values (closer to 1.0) give more weight to historical importance,
            providing stability. Lower values make importance more responsive to recent
            gradients. Typical values range from 0.8 to 0.95. Default is 0.85.

        asa_uncertainty_beta (`float`):
            EMA coefficient for smoothing importance uncertainty estimates. Controls how
            quickly uncertainty adapts to gradient variance. Similar to asa_importance_beta,
            higher values provide more stable estimates. Typical values range from 0.8 to 0.95.
            Default is 0.85.

        asa_schedule_exponent (`float`):
            Schedule exponent controlling the decay rate from total subspaces to
            `asa_target_subspaces` during ASA warmup. Higher values result in faster initial
            pruning (more aggressive early reduction), while lower values provide a more
            gradual, linear-like decay. The formula is:
            current_active_subspaces = asa_target_subspaces + (asa_total_subspaces - asa_target_subspaces) * (progress ** exponent).
            Typical values range from 1.0 (linear) to 5.0 (aggressive). Default is 3.0.

        use_dynamic_rank (`bool`):
            Whether to dynamically determine subspace ranks based on singular value magnitudes.
            When True, each subspace's rank is determined by counting singular values above
            a threshold, allowing different subspaces to have different effective ranks.
            When False, all subspaces use the fixed `subspace_rank`. Default is False.

        svd_threshold (`float`):
            Threshold ratio for dynamic rank selection, only used when `use_dynamic_rank=True`.
            A singular value is considered significant if it exceeds `threshold * max_singular_value`.
            Higher values result in lower effective ranks (more aggressive truncation).
            Typical values range from 0.05 to 0.2. Default is 0.1 (10% of max).
    """

    r: int = field(
        default=100,
        metadata={
            "help": (
                "Total rank for SVD decomposition (R in the paper). Higher values capture more "
                "information but require more computation. The actual rank is clamped to "
                "min(r, in_features, out_features). Typical values: 50-500. Default: 100."
            )
        },
    )
    num_subspaces: int = field(
        default=5,
        metadata={
            "help": (
                "Number of subspaces (K) to cluster the SVD space into. Each subspace learns "
                "independent low-rank updates. Increasing this allows finer-grained adaptation "
                "but increases trainable parameters. Typical values: 3-10. Default: 5."
            )
        },
    )
    subspace_rank: int = field(
        default=1,
        metadata={
            "help": (
                "Rank (r_i) for each trainable subspace. Higher values increase expressiveness "
                "but also increase parameters. For most tasks, values of 1-4 work well. Default: 1."
            )
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with AdaMSS. "
                "For example, ['q_proj', 'v_proj'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "For Vision Transformers, typically ['query', 'value']."
            )
        },
    )
    init_weights: Optional[Literal["orthogonal"]] = field(
        default="orthogonal",
        metadata={
            "help": (
                "Initialization method for AdaMSS trainable weights. Currently only 'orthogonal' "
                "is supported, which uses orthogonal initialization for B matrices. A matrices "
                "are initialized to zero to start from pretrained weights. Set to None to skip "
                "initialization when loading from a checkpoint. Default: 'orthogonal'."
            )
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from AdaMSS layers to be set as trainable and saved in the final checkpoint. "
                "For example, in Sequence Classification or Token Classification tasks, "
                "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "Specific layer indices to apply AdaMSS to. If specified, only these layers will "
                "be adapted, useful for experimenting with which layers benefit most. Can be a "
                "single integer or a list of integers. Default: None (adapt all matching layers)."
            )
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "Pattern to match layer names when `layers_to_transform` is specified. Used to "
                "extract layer indices from module names that don't follow common patterns."
            )
        },
    )
    # ASA (Adaptive Subspace Allocation) parameters
    use_asa: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable Adaptive Subspace Allocation (ASA). When enabled, ASA dynamically "
                "prunes less important subspaces during training based on gradient information, "
                "reducing parameters while maintaining performance. Requires a training callback. "
                "Default: False."
            )
        },
    )
    asa_target_subspaces: int = field(
        default=50,
        metadata={
            "help": (
                "Target total number of active subspaces across all layers when ASA is enabled. "
                "ASA progressively prunes subspaces until this target is reached. Lower values "
                "result in more aggressive pruning. Typical values: 20-100. Default: 50."
            )
        },
    )
    init_warmup: int = field(
        default=50,
        metadata={
            "help": (
                "Training steps to wait before starting ASA pruning. During warmup, all subspaces "
                "remain active to allow importance scores to stabilize. Higher values give more "
                "time for accurate estimation. Typical values: 50-200. Default: 50."
            )
        },
    )
    final_warmup: int = field(
        default=1000,
        metadata={
            "help": (
                "Training step at which ASA completes pruning and reaches asa_target_subspaces. "
                "Should be set based on total training steps; typically 1/3 to 1/2 of total steps. "
                "Default: 1000."
            )
        },
    )
    mask_interval: int = field(
        default=100,
        metadata={
            "help": (
                "Training steps between ASA mask updates. Lower values allow more frequent "
                "adaptation but increase overhead. Higher values provide more stable estimates. "
                "Typical values: 50-200. Default: 100."
            )
        },
    )
    asa_importance_beta: float = field(
        default=0.85,
        metadata={
            "help": (
                "EMA coefficient for smoothing subspace importance scores during ASA. Higher "
                "values (closer to 1.0) give more weight to historical importance, providing "
                "stability. Lower values make importance more responsive to recent gradients. "
                "Typical values: 0.8-0.95. Default: 0.85."
            )
        },
    )
    asa_uncertainty_beta: float = field(
        default=0.85,
        metadata={
            "help": (
                "EMA coefficient for smoothing importance uncertainty estimates during ASA. "
                "Controls how quickly uncertainty adapts to gradient variance. Higher values "
                "provide more stable estimates. Typical values: 0.8-0.95. Default: 0.85."
            )
        },
    )
    asa_schedule_exponent: float = field(
        default=3.0,
        metadata={
            "help": (
                "Schedule exponent controlling the decay rate from total subspaces to "
                "asa_target_subspaces. Higher values result in faster initial pruning (aggressive "
                "early reduction), lower values provide gradual linear-like decay. Formula: "
                "current_active_subspaces = asa_target_subspaces + (total - target) * (progress ** exponent). "
                "Typical values: 1.0 (linear) to 5.0 (aggressive). Default: 3.0."
            )
        },
    )
    # Dynamic rank selection parameters
    use_dynamic_rank: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to dynamically determine subspace ranks based on singular value magnitudes. "
                "When True, each subspace's rank is determined by counting singular values above "
                "a threshold, allowing different subspaces to have different effective ranks. "
                "When False (default), all subspaces use the fixed subspace_rank. Default: False."
            )
        },
    )
    svd_threshold: float = field(
        default=0.1,
        metadata={
            "help": (
                "Threshold ratio for dynamic rank selection (only used when use_dynamic_rank=True). "
                "A singular value is significant if it exceeds threshold * max_singular_value. "
                "Higher values result in lower effective ranks (more aggressive truncation). "
                "Typical values: 0.05-0.2. Default: 0.1 (10% of max)."
            )
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.ADAMSS
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        
        # Normalize init_weights: convert False to None for backward compatibility
        if self.init_weights is False:
            self.init_weights = None
        
        # Validate initialization method
        if self.init_weights not in ["orthogonal", None]:
            raise ValueError(f"init_weights must be 'orthogonal' or None, got {self.init_weights}")
