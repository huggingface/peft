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

# See https://arxiv.org/abs/2506.16289 for details

import torch
import torch.nn as nn
from typing import List, Optional, Dict

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

class KappaTuneSelector:
    """
    Lightweight utility to compute per-module condition numbers (κ = σ_max / σ_min)
    and return the best LoRA target modules. Now supports bnb 4-bit models (paper reproduction).
    """
    def __init__(self, model: nn.Module, max_dim_size_to_analyze: int = 16384):
        self.model = model
        self.max_dim_size_to_analyze = max_dim_size_to_analyze
        self._condition_numbers: Optional[Dict[str, float]] = None

    def _compute_kappas(self) -> None:
        if self._condition_numbers is not None:
            return

        condition_numbers: Dict[str, float] = {}

        for module_name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            # Handle bnb 4-bit quantization (for QLoRA / paper example)
            weight = module.weight
            if bnb is not None and hasattr(weight, "quant_state"):
                try:
                    w = bnb.functional.dequantize_4bit(weight.data, weight.quant_state).float()
                except Exception:
                    w = weight.data.detach().float()
            else:
                w = weight.data.detach().float()

            # Skip huge matrices
            if any(dim > self.max_dim_size_to_analyze for dim in w.shape):
                continue

            try:
                S = torch.linalg.svdvals(w.view(w.size(0), -1))
                kappa = (S[0] / (S[-1] + 1e-8)).item()
                condition_numbers[module_name] = kappa
            except (torch.linalg.LinAlgError, RuntimeError):
                condition_numbers[module_name] = float("inf")

        self._condition_numbers = condition_numbers

    def get_best_targets(
        self,
        top_p: Optional[float] = None,
        num_modules: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[str]:
        
        """
        Return the best target modules according to one of three mutually-exclusive strategies.
        Args:
            top_p: Return the top best modules (e.g. 0.2 = paper default).
            num_modules: Return exactly this many best modules (fixed budget).
            threshold: Return every module with κ ≤ threshold (quality cutoff).
        Returns:
            List of module names (e.g. [model.layers.0.self_attn.q_proj, ...])
        Notes:
            - Precedence (checked in order): num_modules → top_p → threshold → all modules.
            - Modules are always sorted by ascending κ (lowest = best).
            - Recommended: top_p=0.2 for most models (Llama-3, Mistral, Qwen, etc.).
        """
        
        self._compute_kappas()
        if not self._condition_numbers:
            return []

        sorted_modules = sorted(self._condition_numbers.items(), key=lambda x: x[1])

        if num_modules is not None:
            k = min(num_modules, len(sorted_modules))
            return [name for name, _ in sorted_modules[:k]]
        if top_p is not None:
            k = max(1, int(len(sorted_modules) * top_p))
            return [name for name, _ in sorted_modules[:k]]
        if threshold is not None:
            return [name for name, kappa in sorted_modules if kappa <= threshold]

        return [name for name, _ in sorted_modules]


def find_kappa_target_modules(
    model: nn.Module, top_p: float = 0.2, max_dim_size_to_analyze: int = 16384
) -> List[str]:

    """
    One-liner convenience function (recommended for most users).
    Equivalent to:
        selector = KappaTuneSelector(model, max_dim_size_to_analyze)
        return selector.get_best_targets(top_p=top_p)
    Args:
        model: The base model.
        top_p: Fraction of best modules to return (paper default = 0.2).
        max_dim_size_to_analyze: See KappaTuneSelector.__init__.
    Returns:
        List of module names ready for LoraConfig(target_modules=...).
    """
    
    selector = KappaTuneSelector(model, max_dim_size_to_analyze)
    return selector.get_best_targets(top_p=top_p)
