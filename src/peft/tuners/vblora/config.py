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
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class VBLoRAConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`VBLoRAConfig`].

    Paper: https://huggingface.co/papers/2405.15179

    Args:
        r (`int`):
            The rank of incremental matrices.
        num_vectors (`int`):
            Number of vectors in the vector bank. Use higher values when the model size increases.
        vector_length (`int`):
            The length of the vectors in the vector bank. The length of the vectors should be divisible by the hidden
            dimension of the model.
        topk (`int`):
            The K value for top-K selection. A larger value of K increases the size of the saved model. In practice,
            setting K=2 typically provides the best performance and parameter efficiency. For more details, refer to
            the discussion in the paper.
        target_modules (`Union[List[str], str]`):
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
        save_only_topk_weights (`bool`):
            Whether to only save the topk weights. Setting `save_only_topk_weights = True` significantly reduces
            storage space. However, models saved in this mode can be used for merging or inference only, not for
            resuming training.
        vblora_dropout (`float`):
            The dropout probability for VBLoRA layers.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        bias (`str`):
            Bias type for VBLoRA. Can be 'none', 'all' or 'vblora_only'. If 'all' or 'vblora_only', the corresponding
            biases will be updated during training. Be aware that this means that, even when disabling the adapters,
            the model will not produce the same output as the base model would have without adaptation.
        modules_to_save (`List[str]`):
            List of modules apart from VBLoRA layers to be set as trainable and saved in the final checkpoint.
        init_vector_bank_bound (`float`):
            The vector bank is initialized with a uniform distribution between -init_vector_bank_bound and
            init_vector_bank_bound. Avoid initializing the vector bank with all zeros to prevent zero gradients. A
            small value, such as 0.02, is typically effective. Initializing with a large value may cause training
            instability.
        init_logits_std (`float`):
            The logits are initialized with a normal distribution with a standard deviation of init_logits_std. Default
            is 0.1.
        layers_to_transform (`Union[List[int],int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
            that are specified in this list. If a single integer is passed, it will apply the transformations on the
            layer at this index.
        layers_pattern (`Optional[Union[List[str], str]]`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`. This should target the
            `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`.
    """

    r: int = field(default=4, metadata={"help": "The rank of incremental matrices."})
    num_vectors: int = field(
        default=256,
        metadata={"help": "Number of vectors in the vector bank. Use higher values when the model size increases."},
    )
    vector_length: int = field(
        default=256,
        metadata={
            "help": "The length of the vectors in the vector bank. The length of the vectors should be divisible by "
            "the hidden dimension of the model."
        },
    )
    topk: int = field(
        default=2,
        metadata={
            "help": "The K value for top-K selection. A larger value of K increases the size of the saved model. "
            "In practice, setting K=2 typically provides the best performance and parameter efficiency. "
            "For more details, refer to the discussion in the paper."
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually."
            )
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from VBLoRA."},
    )
    save_only_topk_weights: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only save the topk weights. Setting `save_only_topk_weights = True` significantly reduces "
                "storage space. However, models saved in this mode can be used for merging or inference only, not for "
                "resuming training."
            )
        },
    )
    vblora_dropout: float = field(default=0.0, metadata={"help": "VBLoRA dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for VBLoRA. Can be 'none', 'all' or 'vblora_only'"})
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from VBLoRA layers to be set as trainable and saved in the final checkpoint. For"
                " example, in Sequence Classification or Token Classification tasks, the final layer"
                " `classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        },
    )
    init_vector_bank_bound: float = field(
        default=0.02,
        metadata={
            "help": (
                "The vector bank is initialized with a uniform distribution between -init_vector_bank_bound and"
                " init_vector_bank_bound. Avoid initializing the vector bank with all zeros to prevent zero gradients."
                " A small value, such as 0.02, is typically effective. Initializing with a large value may cause"
                " training instability."
            ),
        },
    )
    init_logits_std: float = field(
        default=0.1,
        metadata={
            "help": (
                "The logits are initialized with a normal distribution with a standard deviation of init_logits_std. "
                "Default value 0.1 typically works well."
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index. "
            "This only works when target_modules is a list of str. This should target the `nn.ModuleList` of the "
            "model, which is often called `'layers'` or `'h'`."
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
            "This only works when target_modules is a list of str."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.VBLORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )
        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")
