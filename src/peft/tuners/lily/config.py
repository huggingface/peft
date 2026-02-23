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
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class LilyConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`LilyModel`].

    Args:
        r (`int`):
            Lily's rank. Determines the inner hidden dimension of each adapter and the rank of the
            weight update ``A @ B``. In Lily, since the number of adapters is typically smaller than in LoRA,
            each adapter needs to carry more capacity, so it is recommended to use a larger ``r`` than
            in LoRA — typically ``2x``, ``3x``, or ``4x`` the LoRA rank you would normally use.
            The total number of trainable parameters scales with ``r * (num_A + num_B)``, so increasing
            ``r`` while keeping ``num_A`` and ``num_B`` small is the recommended trade-off.
        num_A (`int`):
            The number of shared A adapters. Lily groups the model's layers into ``num_A`` consecutive
            blocks, where each block of ``total_layers / num_A`` adjacent layers shares one A adapter.
            The A adapter compresses the input into a low-rank representation of size ``r``.
            For best results, ``num_A`` should evenly divide the
            total number of layers in your model. It is recommended that ``num_A`` should ideally
            evenly divide the total number of layers.
            Suggested values: ``total_layers / 2``, ``total_layers / 3``, or ``total_layers / 4``.
            Keeping ``num_A`` small and increasing ``r`` instead leads to
            better performance than the opposite trade-off (large ``num_A``, small ``r``).
        num_B (`int`):
            The number of shared B adapters. Unlike A adapters (which are grouped by layer),
            all B adapters are shared globally across every layer. For each forward pass, a router
            computes a weighted combination of all ``num_B`` B adapters (using softmax-normalized
            weights) to produce a single combined B adapter, which then projects the low-rank
            representation back to the original dimension. It is recommended to set
            ``num_B`` equal to ``num_A``. Suggested values: ``total_layers / 2``, ``total_layers / 3``, or
            ``total_layers / 4``. Similar to ``num_A``, prefer smaller ``num_B`` with larger ``r``
            over larger ``num_B`` with smaller ``r``.
            NOTE: to train router, you need at least 2 B adapters (i.e. ``num_B >= 2``), since the router learns to compute a weighted combination of the B adapters.
             If you set ``num_B = 1``, the router will not be trained.
        target_modules (`Union[List[str], str]`, *optional*):
            The names of the modules to apply Lily to. Can be a list of module name strings (e.g.
            ``['q_proj', 'v_proj']``) or a regex pattern (e.g.
            ``'.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'``). If not specified, Lily will be
            applied to all supported linear layers.
        scaling (`float`):
            A scalar multiplier applied to the combined adapter output (``scaling * A @ combined_B``)
            before adding it to the frozen weight's forward pass. Unlike LoRA, Lily does not use an
            ``alpha / r`` formulation; instead, ``scaling`` is a direct multiplier. This design
            makes it straightforward to sweep over values on a log scale (e.g. ``0.01``, ``0.1``,
            ``1.0``, ``10.0``). The optimal value is task-dependent and should be treated as a
            hyperparameter. We recommend starting with ``1.0``.
        modules_to_save (`List[str]`, *optional*):
            List of modules apart from FourierFT layers to be set as trainable and saved in the final checkpoint. For
            example, in Sequence Classification or Token Classification tasks, the final layer `classifier/score` are
            randomly initialized and as such need to be trainable and saved.
        exclude_modules (`Union[List[str], str]`, *optional*):
            The names of the modules to not apply the adapter. When passing a string, a regex match will be performed.
            When passing a list of strings, either an exact match will be performed or it is checked if the name of the
            module ends with any of the passed strings.
        layers_to_transform (`Union[list[int],int]`):
            The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes
            that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at
            this index.
        layers_pattern (`Optional[Union[List[str], str]]`):
            The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is
            not in the common layers pattern. This should target the `nn.ModuleList` of the model, which is often
            called `'layers'` or `'h'`.
        init_weights (`bool`):
            Whether to initialize Lily adapter weights using the default initialization scheme: A
            matrices are initialized with Kaiming uniform, and B matrices are initialized to zero,
            ensuring that the adapter output is zero at the start of training and does not disturb
            the pretrained model. It is strongly recommended to keep this as ``True`` unless you have
            a specific reason to change it.
    """

    r: int = field(default=4, metadata={"help": "Lily's rank"})
    scaling: float = field(default=1.0, metadata={"help": "scaling factor for lily"})
    num_A: int = field(default=1, metadata={"help": "Lily's number of adapter A"})
    num_B: int = field(default=1, metadata={"help": "Lily's number of adapter B"})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lily."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from Lily."},
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from Lily layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
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
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern. "
            "This should target the `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`."
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from Lily layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the Lily layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.LILY
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )
        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")
