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
class CortexConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`CortexModel`].

    Cortex adds trainable channels to the feed-forward networks (FFNs) of a frozen model. Each added channel works like
    a native FFN channel: it has its own gate, up, and down weights, and it uses the activation of the FFN it extends.

    Args:
        k (`int`):
            The number of channels added to each targeted FFN, i.e. the added intermediate width. For a gated FFN with
            hidden size `d`, this adds `3 * d * k` parameters.
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the FFN modules to adapt. Target whole FFNs, not their projections. Two kinds of modules are
            supported: dense FFNs computing `down_proj(act_fn(gate_proj(x)) * up_proj(x))` (or `down_proj(act_fn(
            up_proj(x)))` without `gate_proj`), like the MLPs of most Transformers decoder models, and packed MoE
            experts following the Transformers experts interface, like the `experts` module of Qwen3-MoE or GPT-OSS.
            The shared experts of MoE models are dense FFNs, e.g. `["shared_experts"]`. When passing a string, a regex
            match will be performed. When passing a list of strings, either an exact match will be performed or it is
            checked if the name of the module ends with any of the passed strings. If this is not specified, modules
            will be chosen according to the model architecture: the dense FFNs of dense models, and the shared experts
            of MoE models, or their packed experts if they have no shared experts. With `moe_target="routed"`, the
            packed experts are chosen instead, and with `moe_target="both"`, they are chosen as well. If the
            architecture is not known, an error will be raised -- in this case, you should specify the target modules
            manually.
        exclude_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to not apply the adapter. When passing a string, a regex match will be performed.
            When passing a list of strings, either an exact match will be performed or it is checked if the name of the
            module ends with any of the passed strings.
        moe_target (`Literal["shared", "routed", "both"]`):
            Where to add channels in MoE models. `"shared"` (default) extends the native shared experts, like the
            `shared_experts` of DeepSeek-V3. Packed experts in MoE blocks without native shared experts instead get `k`
            channels that are common to all experts: they are applied to every token, and their output is multiplied by
            the sum of the selected routing weights of the token, which is the same as adding the channels to every
            expert. `"routed"` adds `k_routed` channels to every expert, which are applied like the native expert
            channels, i.e. only when the router selects that expert and weighted by its routing weight. `"both"` does
            both. Targeted dense FFNs that are not shared experts are extended regardless of this argument.
        k_routed (`Optional[int]`):
            The number of channels added to each expert for the routed branch. Defaults to `k`. Note that the number of
            added parameters is multiplied by the number of experts.
        cortex_bias (`Literal["auto", "none"]`):
            Whether the projections of the added channels have biases. With `"auto"` (default), each of them has a bias
            if the corresponding projection of the base FFN has one, e.g. for GPT-OSS, so that merging does not change
            the architecture of the model. With `"none"`, they have no biases. Unlike the gate and up biases, the down
            bias does not belong to the added channels: like the down bias of the base FFN, it is added to the output
            of the FFN, and merging adds it to the down bias of the base FFN.
        init_weights (`Union[bool, Literal["random", "sample"]]`):
            How to initialize the gate and up weights of the added channels. `"random"` (default, or `True`) uses the
            default initialization of `nn.Linear`. `"sample"` copies the gate and up weights of randomly sampled
            channels of the base FFN; routed channels are sampled from their own expert, channels common to all experts
            from all experts. In both cases, the down weights and biases are initialized to zero, so that the adapter
            initially does not change the output of the model. `False` initializes the down weights and biases randomly
            as well and should only be used for testing.
        init_noise_std (`float`):
            The standard deviation of the Gaussian noise added to the sampled weights when `init_weights="sample"`.
        modules_to_save (`Optional[List[str]]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
    """

    k: int = field(
        default=16,
        metadata={
            "help": "The number of channels added to each targeted FFN, i.e. the added intermediate width. For a "
            "gated FFN with hidden size `d`, this adds `3 * d * k` parameters."
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the FFN modules to adapt with Cortex. Target whole "
            "FFNs, not their projections, e.g. ['mlp'] for dense models, ['experts'] for MoE models, or "
            "['shared_experts'] for the shared experts of MoE models. If not specified, the dense FFNs of dense "
            "models and the shared experts of MoE models are targeted, or their packed experts if they have no shared "
            "experts. With moe_target='routed', the packed experts are targeted instead, and with moe_target='both', "
            "they are targeted as well."
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from Cortex."},
    )
    moe_target: Literal["shared", "routed", "both"] = field(
        default="shared",
        metadata={
            "help": "Where to add channels in MoE models. 'shared' (default) extends the native shared experts, like "
            "the `shared_experts` of DeepSeek-V3. Packed experts in MoE blocks without native shared experts instead "
            "get `k` channels that are common to all experts: they are applied to every token, and their output is "
            "multiplied by the sum of the selected routing weights of the token, which is the same as adding the "
            "channels to every expert. 'routed' adds `k_routed` channels to every expert, which are applied like the "
            "native expert channels, i.e. only when the router selects that expert and weighted by its routing "
            "weight. 'both' does both. Targeted dense FFNs that are not shared experts are extended regardless of "
            "this argument."
        },
    )
    k_routed: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of channels added to each expert for the routed branch. Defaults to `k`. Note that "
            "the number of added parameters is multiplied by the number of experts."
        },
    )
    cortex_bias: Literal["auto", "none"] = field(
        default="auto",
        metadata={
            "help": "Whether the projections of the added channels have biases. With 'auto' (default), each of them "
            "has a bias if the corresponding projection of the base FFN has one, e.g. for GPT-OSS, so that merging "
            "does not change the architecture of the model. With 'none', they have no biases. Unlike the gate and up "
            "biases, the down bias does not belong to the added channels: like the down bias of the base FFN, it is "
            "added to the output of the FFN, and merging adds it to the down bias of the base FFN."
        },
    )
    init_weights: Union[bool, Literal["random", "sample"]] = field(
        default="random",
        metadata={
            "help": "How to initialize the gate and up weights of the added channels. 'random' (default, or `True`) "
            "uses the default initialization of `nn.Linear`. 'sample' copies the gate and up weights of randomly "
            "sampled channels of the base FFN; routed channels are sampled from their own expert, channels common to "
            "all experts from all experts. In both cases, the down weights and biases are initialized to zero, so "
            "that the adapter initially does not change the output of the model. `False` initializes the down weights "
            "and biases randomly as well and should only be used for testing."
        },
    )
    init_noise_std: float = field(
        default=0.01,
        metadata={
            "help": "The standard deviation of the Gaussian noise added to the sampled weights when "
            "`init_weights='sample'`."
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from Cortex layers to be set as trainable and saved in the final "
            "checkpoint. For example, in Sequence Classification or Token Classification tasks, the final layer "
            "`classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.CORTEX
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )
