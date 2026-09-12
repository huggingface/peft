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

"""
Expert scoring for the MoE-to-dense method.

Scoring methods rank the experts of an MoE layer by importance (Section 3.2 of the paper); the top scoring experts are
kept when building the dense FFN. All scoring methods implemented here are computed from routing statistics that are
collected from the router outputs during normal forward passes of the MoE model, i.e. they don't require a separate
calibration procedure. This covers the paper's frequency-based and conditional probability scorings; its
activation-weighted scorings (ACP and DO-ACP) additionally require the outputs of all experts on the calibration tokens
and are not implemented. To add a new scoring method, add a function that maps `ExpertRoutingStats` to a 1D tensor of
scores to `SCORING_FUNCTIONS`.
"""

from collections.abc import Callable

import torch


class ExpertRoutingStats:
    """
    Accumulates routing statistics of a single MoE layer.

    For each expert, the number of tokens for which the expert was selected (`selected_count`) and the sum of its
    routing probabilities over those tokens (`prob_sum`) are tracked, as well as the total number of tokens seen.
    """

    def __init__(self, num_experts: int, device: torch.device | str | None = None) -> None:
        self.num_experts = num_experts
        self.selected_count = torch.zeros(num_experts, dtype=torch.float64, device=device)
        self.prob_sum = torch.zeros(num_experts, dtype=torch.float64, device=device)
        self.num_tokens = 0

    @torch.no_grad()
    def update(self, probs: torch.Tensor, selected_indices: torch.Tensor) -> None:
        """
        Update the statistics with the routing result of a batch of tokens.

        Args:
            probs (`torch.Tensor`):
                Routing probabilities over all experts, shape `[num_tokens, num_experts]`.
            selected_indices (`torch.Tensor`):
                Indices of the selected experts, shape `[num_tokens, top_k]`.
        """
        probs = probs.reshape(-1, self.num_experts)
        selected_indices = selected_indices.reshape(probs.shape[0], -1)
        if self.selected_count.device != probs.device:
            self.selected_count = self.selected_count.to(probs.device)
            self.prob_sum = self.prob_sum.to(probs.device)

        selected = torch.zeros_like(probs, dtype=torch.bool).scatter_(1, selected_indices.long(), True)
        self.selected_count += selected.sum(0).to(torch.float64)
        self.prob_sum += (probs.to(torch.float64) * selected).sum(0)
        self.num_tokens += probs.shape[0]

    def reset(self) -> None:
        self.selected_count.zero_()
        self.prob_sum.zero_()
        self.num_tokens = 0


def conditional_prob_scores(stats: ExpertRoutingStats) -> torch.Tensor:
    """
    Conditional probability (CP) scoring (Section 3.2, Eq. 3 of the paper): the average routing probability of an
    expert over the tokens for which it was selected. In contrast to frequency-based scoring, CP is not diluted by how
    often an expert is chosen, so specialist experts that are rarely selected but confidently routed score highly.
    Experts that are never selected get a score of 0.
    """
    return (stats.prob_sum / stats.selected_count.clamp(min=1)).to(torch.float32)


SCORING_FUNCTIONS: dict[str, Callable[[ExpertRoutingStats], torch.Tensor]] = {
    "conditional_prob": conditional_prob_scores,
}
