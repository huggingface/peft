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

from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class PacaConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`PacaModel`].

    PaCA (Partial Connection Adaptation, https://huggingface.co/papers/2503.01905) fine-tunes `r` randomly selected
    input columns (partial connections) of each targeted pretrained weight. The only trainable parameters are the
    updates of these columns; unlike LoRA, PaCA adds no extra low-rank matrices. Right before the matmul, the adapted
    columns are swapped into the pretrained weight and restored right after it, so forward and backward each run a
    single GEMM, and only the `r` selected input channels are kept in memory for the backward pass.

    Args:
        r (`int`, *optional*, defaults to `8`):
            Number of input columns (partial connections) that are fine-tuned in each targeted layer. For a weight of
            shape `(out_features, in_features)`, PaCA trains `r * out_features` parameters, the same number as the `B`
            matrix of a LoRA adapter with rank `r`. Must not exceed `in_features`.
        paca_alpha (`int`, *optional*, defaults to `8`):
            Scaling factor. The update of the selected columns is `paca_alpha / r * delta`. Like `lora_alpha`, this
            acts as a multiplier of the effective learning rate of the adapter.
        target_modules (`Union[list[str], str]`, *optional*):
            List of module names or regex expression of the module names to replace with PaCA. For example, `['q_proj',
            'v_proj']` or `'.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'`. If this is specified as
            `'all-linear'`, all linear modules except the output layer are chosen. Only `nn.Linear` layers are
            supported.
        exclude_modules (`Union[list[str], str]`, *optional*):
            The names of the modules to not apply the adapter. When passing a string, a regex match will be performed.
            When passing a list of strings, either an exact match will be performed or it is checked if the name of the
            module ends with any of the passed strings.
        layers_to_transform (`Union[list[int], int]`, *optional*):
            The layer indexes to transform. If this argument is specified, PEFT will transform only the layers indexes
            that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at
            this index.
        layers_pattern (`Optional[Union[list[str], str]]`, *optional*):
            The layer pattern name, used only if `layers_to_transform` is different from `None`. This should target the
            `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`.
        random_seed (`int`, *optional*, defaults to `None`):
            Seed used to select the partial connections. Each layer uses a different selection derived from this seed
            and the layer name. If `None`, the global torch random number generator is used. The selected indices are
            always stored in the adapter checkpoint, so the seed is not needed to load a trained adapter.
        paca_indices_pattern (`dict`, *optional*):
            Optional mapping from layer names or regexp expressions to the input column indices that are fine-tuned in
            the matching layers instead of `r` randomly selected ones, for example
            `{'^model.layers.0.self_attn.q_proj': [0, 2, 4]}`. This also allows a different number of columns per
            layer. The indices of a layer must be unique and smaller than `in_features`. The scaling is `paca_alpha /
            r` with the `r` of this config in all layers, independent of the number of given indices. Targeted layers
            that match no key use `r` randomly selected columns.
        init_weights (`bool`, *optional*, defaults to `True`):
            If `True`, the update of the selected columns is initialized to zero, i.e. the adapted model is identical
            to the base model at initialization. Set to `False` to initialize with random values; this is only intended
            for testing.
        modules_to_save (`list[str]`, *optional*):
            List of modules apart from PaCA layers to be set as trainable and saved in the final checkpoint. For
            example, in Sequence Classification or Token Classification tasks, the final layer `classifier/score` are
            randomly initialized and as such need to be trainable and saved.
    """

    r: int = field(
        default=8,
        metadata={
            "help": (
                "Number of input columns (partial connections) that are fine-tuned in each targeted layer. For a "
                "weight of shape `(out_features, in_features)`, PaCA trains `r * out_features` parameters, the same "
                "number as the `B` matrix of a LoRA adapter with rank `r`. Must not exceed `in_features`."
            )
        },
    )
    paca_alpha: int = field(
        default=8,
        metadata={
            "help": (
                "Scaling factor. The update of the selected columns is `paca_alpha / r * delta`. Like `lora_alpha`, "
                "this acts as a multiplier of the effective learning rate of the adapter."
            )
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with PaCA. For example, "
                "`['q_proj', 'v_proj']` or `'.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'`. If this is "
                "specified as `'all-linear'`, all linear modules except the output layer are chosen. Only `nn.Linear` "
                "layers are supported."
            )
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "The names of the modules to not apply the adapter. When passing a string, a regex match will be "
                "performed. When passing a list of strings, either an exact match will be performed or it is checked "
                "if the name of the module ends with any of the passed strings."
            )
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform. If this argument is specified, PEFT will transform only the layers "
                "indexes that are specified inside this list. If a single integer is passed, PEFT will transform only "
                "the layer at this index."
            )
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different from `None`. This should "
                "target the `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`."
            )
        },
    )
    random_seed: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Seed used to select the partial connections. Each layer uses a different selection derived from this "
                "seed and the layer name. If `None`, the global torch random number generator is used. The selected "
                "indices are always stored in the adapter checkpoint, so the seed is not needed to load a trained "
                "adapter."
            )
        },
    )
    paca_indices_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Optional mapping from layer names or regexp expressions to the input column indices that are "
                "fine-tuned in the matching layers instead of `r` randomly selected ones, for example "
                "`{'^model.layers.0.self_attn.q_proj': [0, 2, 4]}`. This also allows a different number of columns per "
                "layer. The indices of a layer must be unique and smaller than `in_features`. The scaling is "
                "`paca_alpha / r` with the `r` of this config in all layers, independent of the number of given "
                "indices. Targeted layers that match no key use `r` randomly selected columns."
            )
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "If `True`, the update of the selected columns is initialized to zero, i.e. the adapted model is "
                "identical to the base model at initialization. Set to `False` to initialize with random values; this "
                "is only intended for testing."
            )
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from PaCA layers to be set as trainable and saved in the final checkpoint. For "
                "example, in Sequence Classification or Token Classification tasks, the final layer `classifier/score` "
                "are randomly initialized and as such need to be trainable and saved."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.PACA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )
        if self.layers_pattern and self.layers_to_transform is None:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")
        if self.r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {self.r}")
        for key, indices in self.paca_indices_pattern.items():
            # the upper bound (in_features) is checked when the layer is created
            if not isinstance(indices, (list, tuple)) or not all(isinstance(i, int) for i in indices):
                raise ValueError(
                    f"The indices in `paca_indices_pattern` must be a list of int (e.g. `tensor.tolist()`), got {key!r}: "
                    f"{indices!r}"
                )
            if not indices:
                raise ValueError(f"`paca_indices_pattern` must not contain an empty list of indices, got {key!r}: []")
            if len(set(indices)) != len(indices):
                raise ValueError(f"The indices in `paca_indices_pattern` must be unique, got {key!r}: {indices}")
            if min(indices) < 0:
                raise ValueError(f"The indices in `paca_indices_pattern` must not be negative, got {key!r}: {indices}")
