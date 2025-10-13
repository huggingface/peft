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


@dataclass
class DeloraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`DeloraModel`].

    Args:
        r (`int`):
            The rank of the DeLoRA adapter.
        delora_lambda (`int`):
            The initial value of the boundary of the DeLoRA adapter. This variable sets an upper bound to the Frobenius
            norm of the weight change, avoiding the finetuned model to deviate too much from the original model.
        module_dropout (`float`):
            The dropout probability for disabling DeLoRA modules during training.
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D modules are chosen,
            excluding the output layer. If this is not specified, modules will be chosen according to the model
            architecture. If the architecture is not known, an error will be raised -- in this case, you should specify
            the target modules manually.
        exclude_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to not apply the adapter. When passing a string, a regex match will be performed.
            When passing a list of strings, either an exact match will be performed or it is checked if the name of the
            module ends with any of the passed strings.
        bias (`str`):
            Bias type for DeLoRA. Can be 'none', 'all' or 'delora_only'. If 'all' or 'delora_only', the corresponding
            biases will be updated during training. Be aware that this means that, even when disabling the adapters,
            the model will not produce the same output as the base model would have without adaptation.
        init_weights (`bool`):
            Whether to perform initialization of adapter weights. If `True` (default): A is initialized with kaiming
            uniform initialization, while B is initialized with zeros. If `False`: A and B are both initialized with
            kaiming uniform, immediately contributing a non-zero delta. This is generally discouraged for normal use.
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
            that are specified in this list. If a single integer is passed, it will apply the transformations on the
            layer at this index.
        layers_pattern (`Optional[Union[List[str], str]]`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`. This should target the
            `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`.
        rank_pattern (`dict`):
            The mapping from layer names or regexp expression to ranks which are different from the default rank
            specified by `r`. For example, `{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`.
        lambda_pattern (`dict`):
            The mapping from layer names or regexp expression to lambdas which are different from the default lambda
            specified by `delora_lambda`. For example, `{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`.
        modules_to_save (`Optional[List[str]]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "DeLoRA rank"})
    delora_lambda: int = field(
        default=15,
        metadata={
            "help": "The initial value of the boundary of the DeLoRA adapter. This variable sets an upper bound to the "
            "Frobenius norm of the weight change, avoiding the finetuned model to deviate too much from the original model."
        },
    )
    module_dropout: float = field(
        default=0.0, metadata={"help": "The dropout probability for disabling DeLoRA modules during training"}
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with DeLoRA."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
            "This can also be a wildcard 'all-linear' which matches all linear layers except the output layer."
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from DeLoRA."},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for DeLoRA. Can be 'none' or 'all'"})
    init_weights: bool = field(
        default=True,
        metadata={
            "help": "Whether to perform initialization of adapter weights. If `True` (default): A is initialized with kaiming uniform "
            "initialization, while B is initialized with zeros. If `False`: A and B are both initialized with kaiming uniform, "
            "immediately contributing a non-zero delta. This is generally discouraged for normal use."
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that "
            "are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the "
            "common layers pattern. This should target the `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`."
        },
    )
    rank_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": "The mapping from layer names or regexp expression to ranks which are different from the default rank specified "
            "by `r`. For example, `{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`."
        },
    )
    lambda_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": "The mapping from layer names or regexp expression to lambdas which are different from the default lambda specified by `delora_lambda`."
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from DeLoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, the final layer `classifier/score` "
            "are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        # PeftType enum members are uppercase; use DELORA
        self.peft_type = PeftType.DELORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")
