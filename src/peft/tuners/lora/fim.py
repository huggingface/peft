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

"""FIM-guided adaptive LoRA rank allocation.

Uses the diagonal of the empirical Fisher Information Matrix (eFIM) to
measure per-layer loss sensitivity, then reallocates ranks so that
information-critical layers receive higher rank and less sensitive layers
receive lower rank, subject to a global budget constraint.

Reference: Optimal Brain Damage (LeCun et al., NeurIPS 1990).
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from typing import Optional

import torch
from torch import nn

from .layer import LoraLayer


__all__ = ["FimConfig", "initialize_lora_fim_ranks"]


class FimConfig:
    """Configuration for FIM-guided adaptive LoRA rank allocation.

    After wrapping a model with :func:`get_peft_model` using
    ``init_lora_weights='fim'``, call :func:`initialize_lora_fim_ranks`
    with a small calibration dataloader to redistribute ranks according
    to each layer's eFIM score.

    Args:
        fim_calibration_batches (`int`):
            Number of forward+backward calibration batches used to
            accumulate eFIM diagonal statistics. More batches give a
            better estimate at the cost of additional compute.
            Default: ``8``.
        r_min (`int`):
            Minimum rank assigned to any layer. Must be >= 1.
            Default: ``1``.
        r_max (`Optional[int]`):
            Maximum rank assigned to any layer. If ``None``, defaults to
            ``2 * r`` (where ``r`` is the base rank from
            :class:`~peft.LoraConfig`). Default: ``None``.
        adjust_scaling_factors (`bool`):
            When ``True``, rescales ``lora_alpha`` for each layer after
            rank reallocation so the effective scaling factor
            ``lora_alpha / r`` is preserved. Default: ``True``.

    Example::

        from peft import LoraConfig, get_peft_model
        from peft.tuners.lora.fim import FimConfig, initialize_lora_fim_ranks

        fim_cfg = FimConfig(fim_calibration_batches=8, r_min=1, r_max=32)
        config = LoraConfig(r=8, init_lora_weights="fim", fim_config=fim_cfg)
        model = get_peft_model(base_model, config)

        initialize_lora_fim_ranks(model, dataloader=calibration_loader)
    """

    def __init__(
        self,
        fim_calibration_batches: int = 8,
        r_min: int = 1,
        r_max: Optional[int] = None,
        adjust_scaling_factors: bool = True,
    ) -> None:
        if fim_calibration_batches < 1:
            raise ValueError("`fim_calibration_batches` must be >= 1.")
        if r_min < 1:
            raise ValueError("`r_min` must be >= 1.")
        if r_max is not None and r_max < r_min:
            raise ValueError("`r_max` must be >= `r_min`.")

        self.fim_calibration_batches = fim_calibration_batches
        self.r_min = r_min
        self.r_max = r_max
        self.adjust_scaling_factors = adjust_scaling_factors


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_lora_layers(model: nn.Module, adapter_name: str) -> dict[str, LoraLayer]:
    """Return a name→LoraLayer mapping for all active adapter layers."""
    layers = {}
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer) and adapter_name in module.lora_A:
            layers[name] = module
    return layers


def _accumulate_fim(
    model: nn.Module,
    dataloader,
    adapter_name: str,
    n_batches: int,
) -> dict[str, torch.Tensor]:
    """Run calibration forward+backward passes and accumulate eFIM diagonals.

    For each LoRA layer the eFIM diagonal is approximated as the mean
    squared gradient of the *lora_A* weight, averaged over all batches.

    Args:
        model: PeftModel wrapping a LoRA-adapted base model.
        dataloader: Iterable of model inputs (dicts passed as **kwargs).
        adapter_name: Name of the active LoRA adapter.
        n_batches: Number of batches to process.

    Returns:
        Mapping from layer name to eFIM diagonal tensor (same shape as
        lora_A weight).
    """
    lora_layers = _get_lora_layers(model, adapter_name)
    fim_accum: dict[str, torch.Tensor] = {}
    fim_steps: dict[str, int] = defaultdict(int)

    was_training = model.training
    model.train()

    try:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= n_batches:
                break

            if isinstance(batch, dict):
                inputs = {
                    k: v.to(next(model.parameters()).device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            else:
                inputs = batch

            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            loss.backward()

            with torch.no_grad():
                for name, layer in lora_layers.items():
                    w = layer.lora_A[adapter_name].weight
                    if w.grad is None:
                        continue
                    grad_sq = w.grad.detach() ** 2
                    if name not in fim_accum:
                        fim_accum[name] = torch.zeros_like(grad_sq)
                    fim_accum[name].add_(grad_sq)
                    fim_steps[name] += 1

            model.zero_grad()
    finally:
        model.train(was_training)

    # Normalise by number of accumulated steps
    return {name: fim_accum[name] / max(fim_steps[name], 1) for name in fim_accum}


def _compute_layer_importance(fim_diags: dict[str, torch.Tensor]) -> dict[str, float]:
    """Aggregate eFIM diagonal per layer into a scalar importance score.

    Uses the mean of the diagonal (i.e. mean squared gradient) as the
    importance score — layers with higher mean gradient variance are more
    sensitive to the loss.

    Args:
        fim_diags: Mapping from layer name to eFIM diagonal tensor.

    Returns:
        Mapping from layer name to scalar importance score.
    """
    return {name: fim.mean().item() for name, fim in fim_diags.items()}


def _allocate_ranks(
    importance: dict[str, float],
    base_r: int,
    r_min: int,
    r_max: int,
) -> dict[str, int]:
    """Allocate integer ranks proportional to layer importance scores.

    The budget constraint is: Σ rank_i = n_layers × base_r (i.e. the
    mean rank equals the original ``r``).  Ranks are proportional to
    normalised importance scores and clamped to [r_min, r_max].

    Args:
        importance: Layer-name → scalar importance score.
        base_r: Original LoRA rank (budget per layer).
        r_min: Minimum rank per layer.
        r_max: Maximum rank per layer.

    Returns:
        Mapping from layer name to allocated integer rank.
    """
    if not importance:
        return {}

    names = list(importance.keys())
    scores = [max(importance[n], 1e-10) for n in names]
    total_score = sum(scores)
    budget = base_r * len(names)

    # Proportional allocation (continuous)
    raw = [s / total_score * budget for s in scores]

    # Round to integers using largest-remainder method
    floors = [math.floor(x) for x in raw]
    remainders = [(raw[i] - floors[i], i) for i in range(len(raw))]
    remainder_budget = budget - sum(floors)
    remainders.sort(reverse=True)
    for j in range(int(remainder_budget)):
        floors[remainders[j][1]] += 1

    ranks = {names[i]: max(r_min, min(r_max, floors[i])) for i in range(len(names))}
    return ranks


def _resize_lora_layer(
    layer: LoraLayer,
    adapter_name: str,
    new_r: int,
    adjust_scaling: bool,
) -> None:
    """Resize lora_A and lora_B weight matrices to ``new_r``.

    Preserves the first ``min(old_r, new_r)`` rows/columns of the
    existing weights where possible, and initialises extra rows with the
    layer's current ``reset_lora_parameters`` strategy (zeros for B,
    kaiming uniform for A).

    Args:
        layer: The LoraLayer whose adapter weights will be resized.
        adapter_name: Name of the adapter to resize.
        new_r: Target rank.
        adjust_scaling: Whether to rescale ``scaling`` so that
            lora_alpha / new_r matches the original lora_alpha / old_r.
    """
    if adapter_name not in layer.lora_A:
        return

    lora_A = layer.lora_A[adapter_name]
    lora_B = layer.lora_B[adapter_name]
    old_r = lora_A.weight.shape[0]  # lora_A: (r, in_features)

    if old_r == new_r:
        return

    device = lora_A.weight.device
    dtype = lora_A.weight.dtype
    in_features = lora_A.weight.shape[1]
    out_features = lora_B.weight.shape[0]

    # Build new weight tensors
    new_A = torch.zeros(new_r, in_features, device=device, dtype=dtype)
    new_B = torch.zeros(out_features, new_r, device=device, dtype=dtype)

    copy_r = min(old_r, new_r)
    with torch.no_grad():
        new_A[:copy_r] = lora_A.weight[:copy_r]
        new_B[:, :copy_r] = lora_B.weight[:, :copy_r]
        # Initialise any extra rows of A with kaiming uniform (standard LoRA init)
        if new_r > old_r:
            nn.init.kaiming_uniform_(new_A[old_r:], a=math.sqrt(5))

    lora_A.weight = nn.Parameter(new_A)
    lora_B.weight = nn.Parameter(new_B)

    # Update stored rank
    layer.r[adapter_name] = new_r

    if adjust_scaling and adapter_name in layer.scaling:
        # Keep lora_alpha / r constant
        old_scaling = layer.scaling[adapter_name]
        layer.scaling[adapter_name] = old_scaling * old_r / new_r


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def initialize_lora_fim_ranks(
    model: nn.Module,
    dataloader=None,
    fim_scores: Optional[dict[str, torch.Tensor]] = None,
    adapter_name: str = "default",
    show_progress_bar: bool = True,
) -> nn.Module:
    """Reallocate LoRA ranks using FIM-guided layer importance scores.

    Runs a small calibration pass (or uses pre-computed ``fim_scores``)
    to estimate per-layer loss sensitivity via the eFIM diagonal, then
    redistributes ranks so that more sensitive layers receive higher rank
    and less sensitive layers receive lower rank, subject to a fixed
    total-rank budget.

    Must be called after :func:`~peft.get_peft_model` with
    ``init_lora_weights='fim'``.

    Args:
        model (`nn.Module`):
            A :class:`~peft.PeftModel` with ``init_lora_weights='fim'``.
        dataloader (optional):
            Iterable of batches (dicts) used for calibration. Each batch
            is passed as ``model(**batch)`` and must produce a ``.loss``
            attribute. Required if ``fim_scores`` is ``None``.
        fim_scores (`Optional[dict[str, torch.Tensor]]`):
            Pre-computed eFIM diagonal tensors (same shape as each
            layer's ``lora_A.weight``), keyed by layer name. If provided,
            ``dataloader`` is ignored.
        adapter_name (`str`):
            Name of the LoRA adapter to initialise. Default: ``"default"``.
        show_progress_bar (`bool`):
            Whether to show a tqdm progress bar during calibration.

    Returns:
        The model with reallocated LoRA ranks (in-place modification).

    Raises:
        ValueError: If ``model`` is not a PeftModel, or if
            ``init_lora_weights`` is not ``'fim'``, or if neither
            ``dataloader`` nor ``fim_scores`` is provided.
    """
    if not hasattr(model, "peft_config"):
        raise ValueError("`model` must be a PeftModel.")

    peft_config = model.peft_config[adapter_name]
    if peft_config.init_lora_weights != "fim":
        raise ValueError(
            "`initialize_lora_fim_ranks` can only be used with `init_lora_weights='fim'`; "
            f"got '{peft_config.init_lora_weights}'."
        )

    fim_cfg: FimConfig = peft_config.fim_config
    if fim_cfg is None:
        warnings.warn(
            "`fim_config` not set; using default FimConfig().",
            stacklevel=2,
        )
        fim_cfg = FimConfig()

    base_r = peft_config.r
    r_min = fim_cfg.r_min
    r_max = fim_cfg.r_max if fim_cfg.r_max is not None else 2 * base_r

    # ---- Step 1: accumulate eFIM if not provided -------------------------
    if fim_scores is None:
        if dataloader is None:
            raise ValueError("Either `dataloader` or `fim_scores` must be provided.")
        if show_progress_bar:
            try:
                from tqdm import tqdm

                dataloader = tqdm(dataloader, desc="FIM calibration", total=fim_cfg.fim_calibration_batches)
            except ImportError:
                pass
        fim_scores = _accumulate_fim(
            model=model,
            dataloader=dataloader,
            adapter_name=adapter_name,
            n_batches=fim_cfg.fim_calibration_batches,
        )

    if not fim_scores:
        warnings.warn(
            "No FIM scores were accumulated (no gradients found). "
            "This may indicate that no calibration batches produced gradients for LoRA layers. "
            "Ranks will remain unchanged.",
            stacklevel=2,
        )
        return model

    # ---- Step 2: compute per-layer importance scores ---------------------
    importance = _compute_layer_importance(fim_scores)

    # ---- Step 3: allocate integer ranks under budget ---------------------
    new_ranks = _allocate_ranks(importance, base_r=base_r, r_min=r_min, r_max=r_max)

    # ---- Step 4: resize lora_A / lora_B and update rank_pattern ----------
    lora_layers = _get_lora_layers(model, adapter_name)
    rank_pattern: dict[str, int] = {}

    for name, layer in lora_layers.items():
        new_r = new_ranks.get(name, base_r)
        _resize_lora_layer(
            layer=layer,
            adapter_name=adapter_name,
            new_r=new_r,
            adjust_scaling=fim_cfg.adjust_scaling_factors,
        )
        if new_r != base_r:
            rank_pattern[name] = new_r

    # Persist the rank_pattern in the config for serialisation
    peft_config.rank_pattern.update(rank_pattern)

    return model
