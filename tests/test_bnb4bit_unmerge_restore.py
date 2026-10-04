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

"""Exact restore of 4-bit base weights across merge/unmerge (#3878)."""

from __future__ import annotations

import warnings

import pytest
import torch

from peft import LoraConfig
from peft.import_utils import is_bnb_4bit_available


def _require_bnb_4bit():
    if not is_bnb_4bit_available():
        pytest.skip("bitsandbytes 4-bit is not available")


@pytest.fixture
def linear4bit_layer():
    _require_bnb_4bit()
    import bitsandbytes as bnb
    from peft.tuners.lora.bnb import Linear4bit

    torch.manual_seed(0)
    base = bnb.nn.Linear4bit(32, 32, bias=False, compute_dtype=torch.float32, quant_type="nf4")
    # Materialize quant_state on CPU (Params4bit quantizes on .to()).
    base = base.to("cpu")
    config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"], init_lora_weights=True)
    layer = Linear4bit(base, "default", config=config, r=4, lora_alpha=8)
    # Non-zero adapter so a lossy unmerge would drift.
    torch.nn.init.normal_(layer.lora_B["default"].weight, std=0.05)
    return layer


def _dequant(weight):
    from bitsandbytes.functional import dequantize_4bit

    return dequantize_4bit(weight.data, weight.quant_state)


def test_unmerge_restores_exact_params4bit_object(linear4bit_layer):
    layer = linear4bit_layer
    original = layer.base_layer.weight
    before = _dequant(original).clone()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        layer.merge()
        layer.unmerge()

    assert layer.base_layer.weight is original
    assert torch.equal(_dequant(layer.base_layer.weight), before)
    # Restoring from cache must not emit the lossy-unmerge rounding warning.
    assert not any("Unmerge lora module to 4-bit" in str(w.message) for w in caught)


def test_repeated_merge_unmerge_does_not_drift(linear4bit_layer):
    layer = linear4bit_layer
    original = layer.base_layer.weight
    before = _dequant(original).clone()

    for _ in range(20):
        layer.merge()
        layer.unmerge()

    assert layer.base_layer.weight is original
    assert torch.equal(_dequant(layer.base_layer.weight), before)
    assert layer._bnb4bit_weight_on_merge == {}


def test_multi_adapter_unmerge_restores_stack(linear4bit_layer):
    layer = linear4bit_layer
    config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"], init_lora_weights=True)
    layer.update_layer("second", r=4, lora_alpha=8, config=config)
    torch.nn.init.normal_(layer.lora_B["second"].weight, std=0.05)

    w0 = layer.base_layer.weight
    before = _dequant(w0).clone()

    layer.merge(adapter_names=["default"])
    w_after_first = layer.base_layer.weight
    assert w_after_first is not w0
    assert layer._bnb4bit_weight_on_merge["default"] is w0

    layer.merge(adapter_names=["second"])
    assert layer._bnb4bit_weight_on_merge["second"] is w_after_first

    layer.unmerge()  # pops second, then default
    assert layer.base_layer.weight is w0
    assert torch.equal(_dequant(layer.base_layer.weight), before)
    assert layer._bnb4bit_weight_on_merge == {}


def test_lossy_fallback_when_cache_missing(linear4bit_layer):
    """Without a stash (e.g. legacy merged state), unmerge keeps the old re-quant path."""
    layer = linear4bit_layer
    layer.merge()
    # Simulate a merged layer that never went through the new stash path.
    layer._bnb4bit_weight_on_merge.clear()
    merged_weight = layer.base_layer.weight

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        layer.unmerge()

    assert any("Unmerge lora module to 4-bit" in str(w.message) for w in caught)
    # Fallback replaces the Params4bit object rather than restoring the original handle.
    assert layer.base_layer.weight is not merged_weight
