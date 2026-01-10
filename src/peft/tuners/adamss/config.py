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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class AdaMSSConfig(PeftConfig):
    """
    Configuration class for AdaMSS (Adaptive Multi-Subspaces) method.
    
    AdaMSS is a parameter-efficient fine-tuning method that decomposes weight matrices
    using SVD and clusters the decomposed space into multiple trainable subspaces.
    
    Args:
        r (`int`): 
            Rank for SVD decomposition (adamss_R). Default is 500.
        num_subspaces (`int`):
            Number of subspaces for clustering (adamss_K). Default is 5.
        subspace_rank (`int`):
            Rank for each subspace (adamss_ri). Default is 1.
        target_modules (`Optional[Union[list[str], str]]`):
            The names of the modules to apply AdaMSS to. If specified, only these modules
            will be adapted. Can be a list or a regex expression.
        modules_to_save (`Optional[list[str]]`):
            List of modules apart from AdaMSS layers to be set as trainable and saved in the final checkpoint.
            For example, in Sequence Classification task, the classification head is randomly initialized and 
            needs to be trained.
        init_weights (`Literal["orthogonal"]`):
            Initialization method for AdaMSS weights. Currently only supports "orthogonal".
            Default is "orthogonal".
        layers_to_transform (`Optional[Union[list[int], int]]`):
            The layer indices to transform. If specified, only these layer indices will be adapted.
        use_asa (`bool`):
            Whether to enable Adaptive Subspace Allocation. If True, dynamically selects important
            subspaces during training. Default is False.
        target_kk (`int`):
            Target number of active subspaces when ASA is enabled. Default is 50.
        init_warmup (`int`):
            Initial warmup steps before starting ASA masking. Default is 50.
        final_warmup (`int`):
            Final warmup steps when ASA reaches target_kk. Default is 1000.
        mask_interval (`int`):
            Steps between ASA updates. Default is 100.
        beta1 (`float`):
            EMA coefficient for importance averaging. Default is 0.85.
        beta2 (`float`):
            EMA coefficient for uncertainty averaging. Default is 0.85.
        tt (`float`):
            Temperature parameter for importance scoring. Default is 0.01.
        layers_pattern (`Optional[Union[list[str], str]]`):
            The layer pattern name for layer index matching if `layers_to_transform` is specified.
    """

    r: int = field(default=500, metadata={"help": "Rank for SVD decomposition (adamss_R)"})
    num_subspaces: int = field(default=5, metadata={"help": "Number of subspaces for clustering (adamss_K)"})
    subspace_rank: int = field(
        default=1, metadata={"help": "Rank for each trainable subspace (adamss_ri)"}
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
    init_weights: Literal["orthogonal"] = field(
        default="orthogonal",
        metadata={
            "help": (
                "Initialization method for AdaMSS weights. Currently only 'orthogonal' is supported, "
                "which uses orthogonal initialization for subspace matrices."
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
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        },
    )
    # ASA (Adaptive Subspace Allocation) parameters
    use_asa: bool = field(default=False, metadata={"help": "Enable Adaptive Subspace Allocation"})
    target_kk: int = field(default=50, metadata={"help": "Target number of active subspaces for ASA"})
    init_warmup: int = field(default=50, metadata={"help": "Initial warmup steps for ASA"})
    final_warmup: int = field(default=1000, metadata={"help": "Final warmup steps for ASA"})
    mask_interval: int = field(default=100, metadata={"help": "Steps between ASA updates"})
    beta1: float = field(default=0.85, metadata={"help": "EMA coefficient for importance"})
    beta2: float = field(default=0.85, metadata={"help": "EMA coefficient for uncertainty"})
    tt: float = field(
        default=3.0,
        metadata={
            "help": (
                "Schedule exponent controlling decay from total_kk to target_kk during ASA warmup. "
                "Aligned with adamss_pkg where curr_kk = target_kk + (total_kk - target_kk) * (mul_coeff ** tt)."
            )
        },
    )
    # Dynamic rank selection parameters
    use_dynamic_rank: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use dynamic rank selection based on singular value threshold. "
                "If True, adaptively determines rank per subspace using SVD threshold. "
                "If False (default, matches adamss_pkg), uses fixed subspace_rank for all subspaces."
            )
        },
    )
    svd_threshold: float = field(
        default=0.1,
        metadata={
            "help": (
                "Singular value threshold ratio for dynamic rank selection. "
                "Only used when use_dynamic_rank=True. "
                "Rank is determined by counting singular values > threshold * largest_singular_value. "
                "Default is 0.1 (10% of largest singular value)."
            )
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.ADAMSS
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # Validate initialization method
        if self.init_weights not in ["orthogonal"]:
            raise ValueError(f"init_weights must be 'orthogonal', got {self.init_weights}")
