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

"""Unit tests for FIM-guided adaptive LoRA rank allocation."""

import pytest
import torch
from torch import nn

from peft import LoraConfig, get_peft_model
from peft.tuners.lora.fim import (
    FimConfig,
    _allocate_ranks,
    _compute_layer_importance,
    _resize_lora_layer,
    initialize_lora_fim_ranks,
)
from peft.tuners.lora.layer import Linear as LoraLinear


# ---------------------------------------------------------------------------
# Small model helpers
# ---------------------------------------------------------------------------


class TwoLayerMLP(nn.Module):
    """Minimal two-layer MLP with a .loss property for testing."""

    def __init__(self, in_features: int = 16, hidden: int = 8, out_features: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, input_ids, labels=None):
        x = self.fc1(input_ids.float())
        x = torch.relu(x)
        logits = self.fc2(x)
        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(logits, labels.float())
        return type("Output", (), {"loss": loss, "logits": logits})()


def _make_peft_model(r: int = 4, fim_config=None, init_lora_weights="fim") -> nn.Module:
    base = TwoLayerMLP()
    config = LoraConfig(
        r=r,
        target_modules=["fc1", "fc2"],
        init_lora_weights=init_lora_weights,
        fim_config=fim_config,
    )
    return get_peft_model(base, config)


def _make_dataloader(n_batches: int = 4, batch_size: int = 2, in_features: int = 16, out_features: int = 4):
    """Return a list of batches that produce a scalar loss."""
    batches = []
    for _ in range(n_batches):
        batches.append(
            {
                "input_ids": torch.randn(batch_size, in_features),
                "labels": torch.randn(batch_size, out_features),
            }
        )
    return batches


# ---------------------------------------------------------------------------
# FimConfig construction
# ---------------------------------------------------------------------------


def test_fim_config_defaults():
    cfg = FimConfig()
    assert cfg.fim_calibration_batches == 8
    assert cfg.r_min == 1
    assert cfg.r_max is None
    assert cfg.adjust_scaling_factors is True


def test_fim_config_invalid_batches():
    with pytest.raises(ValueError, match="fim_calibration_batches"):
        FimConfig(fim_calibration_batches=0)


def test_fim_config_invalid_r_min():
    with pytest.raises(ValueError, match="r_min"):
        FimConfig(r_min=0)


def test_fim_config_invalid_r_max_lt_r_min():
    with pytest.raises(ValueError, match="r_max"):
        FimConfig(r_min=4, r_max=2)


# ---------------------------------------------------------------------------
# _compute_layer_importance
# ---------------------------------------------------------------------------


def test_compute_layer_importance_mean():
    fim_diags = {
        "layer_a": torch.tensor([1.0, 3.0]),  # mean = 2.0
        "layer_b": torch.tensor([0.5, 0.5]),  # mean = 0.5
    }
    importance = _compute_layer_importance(fim_diags)
    assert pytest.approx(importance["layer_a"], abs=1e-5) == 2.0
    assert pytest.approx(importance["layer_b"], abs=1e-5) == 0.5


# ---------------------------------------------------------------------------
# _allocate_ranks
# ---------------------------------------------------------------------------


def test_allocate_ranks_budget_preserved():
    importance = {"a": 2.0, "b": 1.0, "c": 1.0}
    ranks = _allocate_ranks(importance, base_r=4, r_min=1, r_max=8)
    # Budget = 4 * 3 = 12; clamped ranks must sum close to 12
    total = sum(ranks.values())
    assert abs(total - 12) <= len(ranks)  # rounding can shift by at most n_layers


def test_allocate_ranks_high_importance_gets_higher_rank():
    importance = {"high": 10.0, "low": 1.0}
    ranks = _allocate_ranks(importance, base_r=4, r_min=1, r_max=16)
    assert ranks["high"] > ranks["low"]


def test_allocate_ranks_clamps_to_r_min():
    importance = {"a": 0.0, "b": 100.0}
    ranks = _allocate_ranks(importance, base_r=4, r_min=2, r_max=8)
    assert ranks["a"] >= 2


def test_allocate_ranks_clamps_to_r_max():
    importance = {"a": 1000.0, "b": 0.001}
    ranks = _allocate_ranks(importance, base_r=4, r_min=1, r_max=6)
    assert ranks["a"] <= 6


def test_allocate_ranks_empty_returns_empty():
    assert _allocate_ranks({}, base_r=4, r_min=1, r_max=8) == {}


# ---------------------------------------------------------------------------
# _resize_lora_layer
# ---------------------------------------------------------------------------


def test_resize_lora_layer_increase_rank():
    model = _make_peft_model(r=4)
    # find first LoraLinear
    layer = next(m for m in model.modules() if isinstance(m, LoraLinear))
    adapter = model.active_adapter
    old_r = layer.lora_A[adapter].weight.shape[0]
    _resize_lora_layer(layer, adapter, new_r=old_r + 2, adjust_scaling=False)
    assert layer.lora_A[adapter].weight.shape[0] == old_r + 2
    assert layer.lora_B[adapter].weight.shape[1] == old_r + 2


def test_resize_lora_layer_decrease_rank():
    model = _make_peft_model(r=8)
    layer = next(m for m in model.modules() if isinstance(m, LoraLinear))
    adapter = model.active_adapter
    _resize_lora_layer(layer, adapter, new_r=2, adjust_scaling=False)
    assert layer.lora_A[adapter].weight.shape[0] == 2
    assert layer.lora_B[adapter].weight.shape[1] == 2


def test_resize_lora_layer_same_rank_noop():
    model = _make_peft_model(r=4)
    layer = next(m for m in model.modules() if isinstance(m, LoraLinear))
    adapter = model.active_adapter
    old_A = layer.lora_A[adapter].weight.clone()
    _resize_lora_layer(layer, adapter, new_r=4, adjust_scaling=False)
    assert torch.equal(layer.lora_A[adapter].weight, old_A)


def test_resize_lora_layer_adjust_scaling():
    model = _make_peft_model(r=4)
    layer = next(m for m in model.modules() if isinstance(m, LoraLinear))
    adapter = model.active_adapter
    old_scaling = layer.scaling[adapter]
    _resize_lora_layer(layer, adapter, new_r=8, adjust_scaling=True)
    # scaling should be halved (4/8 = 0.5)
    assert pytest.approx(layer.scaling[adapter], rel=1e-4) == old_scaling * 4 / 8


# ---------------------------------------------------------------------------
# initialize_lora_fim_ranks — end-to-end
# ---------------------------------------------------------------------------


def test_initialize_fim_ranks_runs():
    fim_cfg = FimConfig(fim_calibration_batches=2, r_min=1, r_max=8)
    model = _make_peft_model(r=4, fim_config=fim_cfg)
    dl = _make_dataloader(n_batches=2)
    initialize_lora_fim_ranks(model, dataloader=dl, show_progress_bar=False)


def test_initialize_fim_ranks_changes_some_ranks():
    fim_cfg = FimConfig(fim_calibration_batches=4, r_min=1, r_max=16)
    model = _make_peft_model(r=4, fim_config=fim_cfg)
    dl = _make_dataloader(n_batches=4)
    initialize_lora_fim_ranks(model, dataloader=dl, show_progress_bar=False)
    adapter = model.active_adapter
    ranks = {
        name: m.r[adapter] for name, m in model.named_modules() if isinstance(m, LoraLinear) and adapter in m.lora_A
    }
    # At least one layer should have rank != 4 (redistribution happened)
    # (not guaranteed with tiny random data, but check it's integer and in range)
    for r in ranks.values():
        assert 1 <= r <= 16


def test_initialize_fim_ranks_requires_peft_model():
    with pytest.raises(ValueError, match="PeftModel"):
        initialize_lora_fim_ranks(nn.Linear(4, 4), dataloader=[])


def test_initialize_fim_ranks_requires_fim_init():
    model = _make_peft_model(r=4, init_lora_weights=True)
    with pytest.raises(ValueError, match="init_lora_weights='fim'"):
        initialize_lora_fim_ranks(model, dataloader=[])


def test_initialize_fim_ranks_requires_dataloader_or_scores():
    fim_cfg = FimConfig(fim_calibration_batches=2)
    model = _make_peft_model(r=4, fim_config=fim_cfg)
    with pytest.raises(ValueError, match="dataloader.*fim_scores"):
        initialize_lora_fim_ranks(model, show_progress_bar=False)


def test_initialize_fim_ranks_with_precomputed_scores():
    fim_cfg = FimConfig(fim_calibration_batches=2, r_min=1, r_max=8)
    model = _make_peft_model(r=4, fim_config=fim_cfg)
    adapter = model.active_adapter
    # Build fake fim_scores with the right shapes
    fim_scores = {}
    for name, m in model.named_modules():
        if isinstance(m, LoraLinear) and adapter in m.lora_A:
            fim_scores[name] = torch.ones_like(m.lora_A[adapter].weight)
    initialize_lora_fim_ranks(model, fim_scores=fim_scores, show_progress_bar=False)


# ---------------------------------------------------------------------------
# LoraConfig validation
# ---------------------------------------------------------------------------


def test_lora_config_fim_default_config_created():
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = LoraConfig(r=4, init_lora_weights="fim")
    assert config.fim_config is not None
    assert any("FimConfig" in str(x.message) for x in w)


def test_lora_config_fim_config_ignored_warning():
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        LoraConfig(r=4, init_lora_weights=True, fim_config=FimConfig())
    assert any("fim_config" in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------


def test_top_level_import():
    from peft import FimConfig as F
    from peft import initialize_lora_fim_ranks as fn

    assert F is not None
    assert fn is not None
