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

from .scoring import SCORING_FUNCTIONS


@dataclass
class MoeToDenseConfig(PeftConfig):
    """
    Configuration class of [`MoeToDenseModel`], which converts a Mixture-of-Experts (MoE) model into a dense model.

    The method follows "Pruning and Distilling Mixture-of-Experts into Dense Language Models" (Kim et al., 2026,
    https://huggingface.co/papers/2605.28207): for each MoE layer, the experts are scored by importance using routing
    statistics collected on calibration data (Section 3.2 of the paper), the top scoring experts are concatenated into
    a single dense feed-forward network (Section 3.1), and the resulting dense model is refined by distilling the
    logits of the original MoE model, the teacher, into it (Section 3.5). Of the paper's design space, the
    implementation covers the pure pruning configuration (`K = k` in the paper's notation: each kept expert is copied,
    no experts are merged), which the paper found to perform best (Section 4.2).

    Args:
        target_modules (`Optional[Union[list[str], str]]`):
            The names of the *experts* modules of the MoE layers to convert, i.e. the modules that hold the weights of
            all experts as 3D tensors (e.g. `Qwen3MoeExperts`). If not specified, all such modules are detected
            automatically.
        exclude_modules (`Optional[Union[list[str], str]]`):
            The names of the modules to not convert. When passing a string, a regex match will be performed. When
            passing a list of strings, either an exact match will be performed or it is checked if the name of the
            module ends with any of the passed strings.
        num_experts_to_keep (`Optional[int]`):
            The number of experts to keep per MoE layer (`K` in the paper's notation). The dense FFN has an
            intermediate size of `num_experts_to_keep` times the intermediate size of a single expert. Defaults to the
            number of experts the router activates per token (top-k), which results in a dense FFN with the same number
            of active parameters as the MoE layer; this matches the paper's best configuration `K = k` (Section 4.2).
            Note that other values change the width of the dense FFN, whereas the paper instead merges `K > k` experts
            into `k` groups, which is not implemented.
        scoring (`str`):
            The method used to score the importance of the experts. Currently, only `"conditional_prob"` is supported:
            the conditional probability (CP) scoring of Section 3.2, Eq. 3 of the paper, which ranks experts by their
            average routing probability over the tokens for which they were selected. Among the scoring methods that
            can be computed from the router outputs alone, CP performed best in the paper (Section 4.2, Figure 4); the
            paper's top-scoring methods ACP and DO-ACP additionally require the outputs of all experts on the
            calibration data and are not implemented.
        modules_to_save (`Optional[Union[list[str], str]]`):
            Names of additional modules to fully fine-tune together with the dense FFNs, e.g. the attention projections
            and the norms (`["q_proj", "k_proj", "v_proj", "o_proj", "input_layernorm", "post_attention_layernorm"]`).
            Trainable copies of these modules are created and saved with the adapter, while the originals are still
            used for the teacher (i.e. when the adapters are disabled). This corresponds to the paper's setup of
            distilling the whole student rather than only the FFNs, and it lets the rest of the network adapt to the
            replaced MoE layers. The non-expert parameters of an MoE model are usually a small fraction of the total,
            so the additional cost is low. Do not include the embeddings or the language modeling head, which are
            shared between teacher and student.
    """

    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "The names of the experts modules of the MoE layers to convert. If not specified, all modules that "
                "look like transformers MoE experts modules are detected automatically."
            ),
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from conversion."},
    )
    num_experts_to_keep: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The number of experts to keep per MoE layer (`K` in the paper's notation). The dense FFN has an intermediate "
                "size of `num_experts_to_keep` times the intermediate size of a single expert. Defaults to the number of "
                "experts the router activates per token (top-k), which results in a dense FFN with the same number of active "
                "parameters as the MoE layer; this matches the paper's best configuration `K = k` (Section 4.2).  Note that "
                "other values change the width of the dense FFN, whereas the paper instead merges `K > k` experts into `k` "
                "groups, which is not implemented."
            )
        },
    )
    scoring: str = field(
        default="conditional_prob",
        metadata={
            "help": (
                'The method used to score the importance of the experts. Currently, only `"conditional_prob"` is supported: '
                "the conditional probability (CP) scoring of Section 3.2, Eq. 3 of the paper, which ranks experts by their "
                "average routing probability over the tokens for which they were selected. Among the scoring methods that can "
                "be computed from the router outputs alone, CP performed best in the paper (Section 4.2, Figure 4); the "
                "paper's top-scoring methods ACP and DO-ACP additionally require the outputs of all experts on the "
                "calibration data and are not implemented."
            ),
        },
    )
    modules_to_save: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "Names of additional modules to fully fine-tune together with the dense FFNs, e.g. the attention projections "
                'and the norms (`["q_proj", "k_proj", "v_proj", "o_proj", "input_layernorm", "post_attention_layernorm"]`). '
                "Trainable copies of these modules are created and saved with the adapter, while the originals are still used "
                "for the teacher (i.e. when the adapters are disabled). This corresponds to the paper's setup of distilling "
                "the whole student rather than only the FFNs, and it lets the rest of the network adapt to the replaced MoE "
                "layers. The non-expert parameters of an MoE model are usually a small fraction of the total, so the "
                "additional cost is low. Do not include the embeddings or the language modeling head, which are shared "
                "between teacher and student."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.MOE_TO_DENSE
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )
        if self.scoring not in SCORING_FUNCTIONS:
            raise ValueError(f"Unknown scoring method '{self.scoring}', choose one of {sorted(SCORING_FUNCTIONS)}.")
        if (self.num_experts_to_keep is not None) and (self.num_experts_to_keep < 1):
            raise ValueError(f"`num_experts_to_keep` must be a positive integer, got {self.num_experts_to_keep}.")
