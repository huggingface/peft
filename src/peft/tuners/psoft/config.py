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

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class PsoftConfig(PeftConfig):
    """
    Configuration for PSOFT (Efficient Orthogonal Fine-Tuning with Principal Subspace Adaptation).

    PSOFT inserts an r*r orthogonal transformation R between low-rank matrices A and B, so the low-rank update is ΔW =
    B @ (R-I) @ A. Only R (and optional tunable vectors) are trained; A and B are initialized with psoft_init
    (SVD-based, row-orthogonal A) and frozen.

    Args:
        r (`int`):
            Defaults to 32. PSOFT rank (r) controls the adapter capacity through an r*r transformation R. Smaller ranks
            32-128 are typically sufficient for simple tasks, More complex tasks may benefit from 64-256, increasing
            expressiveness at the cost of additional parameters and computation. See the paper for empirically
            validated settings: https://openreview.net/forum?id=FSHrinMArK.
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D modules are chosen (if
            the model is a PreTrainedModel, the output layer excluded). If this is not specified, modules will be
            chosen according to the model architecture. If the architecture is not known, an error will be raised -- in
            this case, you should specify the target modules manually.
        exclude_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to not apply the adapter. When passing a string, a regex match will be performed.
            When passing a list of strings, either an exact match will be performed or it is checked if the name of the
            module ends with any of the passed strings.
        psoft_alpha (`int`): Defaults to 32. It controls PSOFT scaling factor. Same semantics as LoRA alpha.
        psoft_dropout (`float`): Defaults to 0.0. Dropout for PSOFT path. Same semantics as LoRA dropout.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        ab_svd_init (`Literal["psoft_init", "pissa_init"]`):
            Defaults to 'psoft_init'. Initialization strategy for A and B used to construct the principal subspace in
            PSOFT. 'psoft_init': SVD-based initialization with row-orthogonal A, ensuring strict orthogonality (PSOFT).
            'pissa_init': SVD-based initialization with symmetric A and B (standard PiSSA).
        psoft_svd (`Literal["full", "lowrank"]`):
            Defaults to 'full'. SVD backend for initialization: 'full' uses torch.linalg.svd; 'lowrank' uses
            torch.svd_lowrank.
        psoft_svd_lowrank_niter (`int`):
            Only used when psoft_svd='lowrank'. Defaults to 10. Number of power iterations used by torch.svd_lowrank
            when psoft_svd='lowrank'.
        psoft_orth (`bool`):
            Defaults to 'True'. If True, constrains R to be orthogonal via Cayley parameterization, preserving the
            geometric relationships among column of the pre-trained weight vectors. If False, R is a free matrix
            without orthogonality constraints.
        psoft_mag_b (`bool`):
            Defaults to 'True'. If True, learns a diagonal scaling vector on the 'output' side of R. Commonly paired
            with psoft_mag_a to increase task adaptability, with slight distortion to the pre-trained geometry.
        psoft_mag_a (`bool`):
            Defaults to 'True'. If True, learns a diagonal scaling vector on the 'input' side of R. Commonly paired
            with psoft_mag_b to increase task adaptability, with slight distortion to the pre-trained geometry.
        use_cayley_neumann (`bool`):
            Defaults to 'False'. Whether to use the Cayley-Neumann formulation of PSOFT or not. Set to True to improve
            computational efficiency but comes at costs of bigger approximation error for orthogonality.
        num_cayley_neumann_terms (`int`):
            Defaults to 5. Only used when use_cayley_neumann=True. Number of Cayley-Neumann terms to use. Higher number
            results in less approximation error for orthogonality.
        cayley_neumann_eps (`optional[float]`):
            Defaults to 'None'. Only used when use_cayley_neumann=True. Optional Frobenius-norm bound for the generator
            matrix Q in the Cayley-Neumann approximation. If None (default), no rescaling is applied. If set to a value
            in (0, 1) (e.g., 0.9), Q is rescaled whenever ||Q||_F exceeds the threshold to improve numerical stability.
            See https://spherelab.ai/oftv2/ for details.
        init_weights (`bool`):
            Defaults to 'True'. Whether to initialize the weights of the PSOFT layers with their default
            initialization. Don't change this setting, except if you know exactly what you're doing.
        modules_to_save (`List[str]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
            that are specified in this list. If a single integer is passed, it will apply the transformations on the
            layer at this index.
        layers_pattern (`Optional[Union[List[str], str]]`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`. This should target the
            `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`.
    """

    r: int = field(
        default=32,
        metadata={
            "help": (
                "PSOFT rank (r) controls the adapter capacity through an r*r transformation R. "
                "Smaller ranks 32-128 are typically sufficient for simple tasks, More complex tasks may benefit from 64-256, "
                "increasing expressiveness at the cost of additional parameters and computation. "
                "See the paper for empirically validated settings: https://openreview.net/forum?id=FSHrinMArK. "
            )
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with PSOFT. "
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D "
                "(if the model is a PreTrainedModel, the output layer excluded). "
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually. "
            ),
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from PSOFT. "},
    )
    psoft_alpha: int = field(
        default=32, metadata={"help": "It controls PSOFT scaling factor. Same semantics as LoRA alpha. "}
    )
    psoft_dropout: float = field(
        default=0.0, metadata={"help": "Dropout for PSOFT path. Same semantics as LoRA dropout. "}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out). "},
    )
    ab_svd_init: Literal["psoft_init", "pissa_init"] = field(
        default="psoft_init",
        metadata={
            "help": (
                "Initialization strategy for A and B used to construct the principal subspace in PSOFT. "
                "- 'psoft_init': SVD-based initialization with row-orthogonal A (asymmetric A and B), ensuring strict orthogonality (PSOFT). "
                "- 'pissa_init': SVD-based initialization with symmetric A and B, without strict orthogonality constraint (standard PiSSA). "
            )
        },
    )
    psoft_svd: Literal["full", "lowrank"] = field(
        default="full",
        metadata={
            "help": "SVD backend for initialization: 'full' uses torch.linalg.svd; 'lowrank' uses torch.svd_lowrank. "
        },
    )
    psoft_svd_lowrank_niter: int = field(
        default=10,
        metadata={
            "help": "Number of power iterations used by torch.svd_lowrank when psoft_svd='lowrank'. Only used when psoft_svd='lowrank'. "
        },
    )
    psoft_orth: bool = field(
        default=True,
        metadata={
            "help": (
                "If True, constrains R to be orthogonal via Cayley parameterization, preserving the geometric relationships among column of the pre-trained weight vectors. "
                "If False, R is a free matrix without orthogonality constraints. "
            )
        },
    )
    psoft_mag_b: bool = field(
        default=True,
        metadata={
            "help": (
                "If True, learns a diagonal scaling vector on the 'output' side of R. "
                "Commonly paired with psoft_mag_a to increase task adaptability, with slight distortion to the pre-trained geometry. "
            )
        },
    )
    psoft_mag_a: bool = field(
        default=True,
        metadata={
            "help": (
                "If True, learns a diagonal scaling vector on the 'input' side of R. "
                "Commonly paired with psoft_mag_b to increase task adaptability, with slight distortion to the pre-trained geometry. "
            )
        },
    )
    use_cayley_neumann: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the Cayley-Neumann Formulation of PSOFT or not. Set to True to improve computational efficiency but comes at costs of bigger approximation error for orthogonality. "
        },
    )
    num_cayley_neumann_terms: int = field(
        default=5,
        metadata={
            "help": "Number of Cayley-Neumann terms to use. Higher number results in less approximation error for orthogonality. Only used when use_cayley_neumann=True."
        },
    )
    cayley_neumann_eps: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional Frobenius-norm bound for the generator matrix Q in the Cayley-Neumann approximation. Only used when use_cayley_neumann=True. "
                "If None (default), no rescaling is applied. "
                "If set to a value in (0, 1) (e.g., 0.9), Q is rescaled whenever ||Q||_F exceeds the threshold to improve numerical stability. "
                "See https://spherelab.ai/oftv2/ for details. "
            )
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from PSOFT layers to be set as trainable and saved in the final checkpoint. "
                "For example, in Sequence Classification or Token Classification tasks, "
                "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved. "
            )
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the PSOFT layers with their default initialization. "
                "Don't change this setting, except if you know exactly what you're doing. "
            )
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index. "
                "This only works when target_modules is a list of str."
            )
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern. "
                "This only works when target_modules is a list of str. This should target the `nn.ModuleList` of the "
                "model, which is often called `'layers'` or `'h'`. "
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.PSOFT
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

        if self.r <= 0:
            raise ValueError(f"`r` must be a positive integer; got {self.r}.")

        allowed_inits = {"psoft_init", "pissa_init"}
        if self.ab_svd_init not in allowed_inits:
            raise ValueError(f"`ab_svd_init` must be one of {sorted(allowed_inits)}; got {self.ab_svd_init!r}.")

        allowed_svd_backends = {"full", "lowrank"}
        if self.psoft_svd not in allowed_svd_backends:
            raise ValueError(f"`psoft_svd` must be one of {sorted(allowed_svd_backends)}; got {self.psoft_svd!r}.")

        DEFAULT_LOW_RANK_NITER = self.__dataclass_fields__["psoft_svd_lowrank_niter"].default
        if self.psoft_svd != "lowrank" and self.psoft_svd_lowrank_niter != DEFAULT_LOW_RANK_NITER:
            warnings.warn(
                "`psoft_svd_lowrank_niter` is only used when `psoft_svd='lowrank'`. "
                f"Got psoft_svd={self.psoft_svd!r}, so psoft_svd_lowrank_niter="
                f"{self.psoft_svd_lowrank_niter} will be ignored.",
                UserWarning,
            )

        DEFAULT_NUM_CAYLEY_NEUMANN_TERMS = self.__dataclass_fields__["num_cayley_neumann_terms"].default
        if self.use_cayley_neumann:
            if self.num_cayley_neumann_terms <= 0:
                raise ValueError(
                    f"`num_cayley_neumann_terms` must be a positive integer; got {self.num_cayley_neumann_terms}."
                )
            if self.cayley_neumann_eps is not None and not (0.0 < self.cayley_neumann_eps < 1.0):
                raise ValueError(f"`cayley_neumann_eps` must be in (0, 1) when set; got {self.cayley_neumann_eps}.")
        else:
            if self.num_cayley_neumann_terms != DEFAULT_NUM_CAYLEY_NEUMANN_TERMS:
                warnings.warn(
                    "`num_cayley_neumann_terms` is only used when `use_cayley_neumann=True`. "
                    f"Since `use_cayley_neumann=False`, `num_cayley_neumann_terms={self.num_cayley_neumann_terms}` will be ignored.",
                    UserWarning,
                )
            if self.cayley_neumann_eps is not None:
                warnings.warn(
                    "`cayley_neumann_eps` is only used when `use_cayley_neumann=True`. "
                    f"Since `use_cayley_neumann=False`, `cayley_neumann_eps={self.cayley_neumann_eps}` will be ignored.",
                    UserWarning,
                )
