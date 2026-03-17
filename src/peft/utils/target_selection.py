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

import torch
import torch.nn as nn
from typing import List, Optional, Dict

class KappaTuneSelector:
    """
    Lightweight utility to compute per-module condition numbers (κ = σ_max / σ_min)
    and return the best LoRA target modules (lowest κ = most stable / least anisotropic).
    Use it before creating LoraConfig (no dependency on the full KappaTune optimizer).
    """
    def __init__(self, model: nn.Module, max_dim_size_to_analyze: int = 16384):
        """
        Args:
            model: The base model (e.g. AutoModelForCausalLM).
            max_dim_size_to_analyze: Skip any weight with a dimension > this value
                                    (safety for embeddings, very large matrices, etc.).
        """
        self.model = model
        self.max_dim_size_to_analyze = max_dim_size_to_analyze
        self._condition_numbers: Optional[Dict[str, float]] = None  # module_name -> kappa

    def _compute_kappas(self) -> None:
        """Compute condition number for every nn.Linear module's weight."""
        if self._condition_numbers is not None:
            return

        condition_numbers: Dict[str, float] = {}
        logger = None  # optional: you can add logging if you want

        for module_name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            weight = module.weight.detach()

            # Skip if too large
            if any(dim > self.max_dim_size_to_analyze for dim in weight.shape):
                continue

            # SVD (GPU if possible)
            try:
                if weight.is_cuda:
                    _, s, _ = torch.linalg.svd(weight, full_matrices=False)
                else:
                    _, s, _ = torch.linalg.svd(weight.cpu(), full_matrices=False)

                kappa = (s[0] / s[-1]).item() if s[-1] > 1e-8 else float("inf")
                condition_numbers[module_name] = kappa
            except (torch.linalg.LinAlgError, RuntimeError):
                condition_numbers[module_name] = float("inf")  # treat as bad target

        self._condition_numbers = condition_numbers

    def get_best_targets(
        self,
        top_p: Optional[float] = None,      # e.g. 0.2 → top 20% of modules
        num_modules: Optional[int] = None,  # absolute number (overrides top_p)
        threshold: Optional[float] = None   # kappa <= threshold (rarely used)
    ) -> List[str]:
        """
        Returns a list of module names ready for LoraConfig(target_modules=...).

        Priority order:
        1. num_modules (fixed budget)
        2. top_p (percentage)
        3. threshold (all modules below kappa)
        4. everything (fallback)

        Modules are sorted by ascending kappa (lowest = best for adaptation).
        """
        self._compute_kappas()
        if not self._condition_numbers:
            return []

        # Sort: lowest kappa first
        sorted_modules = sorted(
            self._condition_numbers.items(),
            key=lambda x: x[1]
        )

        if num_modules is not None:
            k = min(num_modules, len(sorted_modules))
            return [name for name, _ in sorted_modules[:k]]

        if top_p is not None:
            k = max(1, int(len(sorted_modules) * top_p))
            return [name for name, _ in sorted_modules[:k]]

        if threshold is not None:
            return [name for name, kappa in sorted_modules if kappa <= threshold]

        # fallback: all modules
        return [name for name, _ in sorted_modules]


# Optional convenience function (many people prefer this style)
def find_kappa_target_modules(
    model: nn.Module,
    top_p: float = 0.2,
    max_dim_size_to_analyze: int = 16384,
) -> List[str]:
    """One-liner version for quick use."""
    selector = KappaTuneSelector(model, max_dim_size_to_analyze)
    return selector.get_best_targets(top_p=top_p)
