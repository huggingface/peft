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

"""Tests for the KaSA (Knowledge-aware Singular-value Adaptation) LoRA variant.

KaSA changes vanilla LoRA in two ways (arXiv:2412.06071):
  1. a one-time, destructive SVD truncation of the frozen base weight (drop the r smallest singular components), and
  2. a learnable diagonal of singular values (lora_diag) inserted between the LoRA A and B factors.

These tests run on tiny random nn.Linear models on CPU; no downloads.
"""

import copy

import pytest
import torch
from torch import nn

from peft import KasaConfig, LoraConfig, PeftType, get_kasa_regularization_loss, get_peft_model
from peft.tuners.lora import KasaConfig as LoraKasaConfig
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.lora.variants import KasaLinearVariant
from peft.utils import get_peft_model_state_dict


class MLP(nn.Module):
    def __init__(self, in_features=16, hidden=12, out_features=10, bias=False):
        super().__init__()
        # in_features >= hidden >= out so min(in,out) - r stays positive at small r for both layers.
        self.lin0 = nn.Linear(in_features, hidden, bias=bias)
        self.lin1 = nn.Linear(hidden, out_features, bias=bias)

    def forward(self, x):
        return self.lin1(torch.relu(self.lin0(x)))


def _make_kasa_config(target_modules=("lin0", "lin1"), r=4, lora_alpha=8, beta=1e-4, gamma=1e-3, **kwargs):
    return LoraConfig(
        target_modules=list(target_modules),
        r=r,
        lora_alpha=lora_alpha,
        kasa_config=KasaConfig(beta=beta, gamma=gamma),
        **kwargs,
    )


# ----------------------------------------------------------------------------------------------------------------------
# Config wiring
# ----------------------------------------------------------------------------------------------------------------------


def test_kasa_config_object_triggers_variant():
    torch.manual_seed(0)
    model = get_peft_model(copy.deepcopy(MLP()), _make_kasa_config())
    found = 0
    for module in model.modules():
        if isinstance(module, LoraLinear):
            assert isinstance(module.lora_variant["default"], KasaLinearVariant)
            found += 1
    assert found == 2


def test_kasa_config_dict_round_trip():
    # A dict passed as kasa_config should be coerced to a KasaConfig in __post_init__.
    cfg = LoraConfig(target_modules=["lin0"], kasa_config={"beta": 0.5, "gamma": 0.25})
    assert isinstance(cfg.kasa_config, KasaConfig)
    assert cfg.kasa_config.beta == 0.5
    assert cfg.kasa_config.gamma == 0.25
    assert cfg.peft_type == PeftType.LORA


def test_kasa_config_alias_matches_lora_module_config():
    # The class re-exported from peft.tuners.lora must be the same as the top-level one.
    assert LoraKasaConfig is KasaConfig


def test_kasa_config_invalid_type_raises():
    with pytest.raises(TypeError, match="`kasa_config` must be a `KasaConfig`"):
        LoraConfig(target_modules=["lin0"], kasa_config=123)


def test_kasa_config_negative_coeffs_raise():
    with pytest.raises(ValueError, match="`beta` must be non-negative"):
        KasaConfig(beta=-1.0)
    with pytest.raises(ValueError, match="`gamma` must be non-negative"):
        KasaConfig(gamma=-1.0)


def test_kasa_rejects_too_large_rank():
    # r must be < min(in, out) for at least one base singular component to survive truncation.
    # lin1 is (out=10, in=12) so min=10; r=10 must raise.
    with pytest.raises(ValueError, match="KaSA requires `r`"):
        get_peft_model(copy.deepcopy(MLP()), _make_kasa_config(r=10))


# ----------------------------------------------------------------------------------------------------------------------
# Faithfulness: SVD truncation of the base weight + the new lora_diag parameter
# ----------------------------------------------------------------------------------------------------------------------


def test_kasa_lora_diag_shape_and_learnable():
    torch.manual_seed(0)
    r = 4
    model = get_peft_model(copy.deepcopy(MLP()), _make_kasa_config(r=r))
    for module in model.modules():
        if isinstance(module, LoraLinear):
            diag = module.lora_diag["default"]
            assert diag.shape == (r,)
            assert diag.requires_grad


def test_kasa_lora_b_is_zero_init():
    # The update must be zero at init (output == truncated base), which requires B == 0.
    torch.manual_seed(0)
    model = get_peft_model(copy.deepcopy(MLP()), _make_kasa_config())
    for module in model.modules():
        if isinstance(module, LoraLinear):
            assert torch.allclose(module.lora_B["default"].weight, torch.zeros_like(module.lora_B["default"].weight))


def test_kasa_truncates_base_weight_rank():
    """After init, the frozen base weight must have its rank reduced by exactly r (its r smallest singular values are
    dropped)."""
    torch.manual_seed(0)
    r = 3
    base = MLP(in_features=16, hidden=12, out_features=10)
    # Snapshot the original singular values of each targeted weight before adapting.
    orig_singulars = {}
    for name in ["lin0", "lin1"]:
        w = getattr(base, name).weight.detach().clone().float()
        orig_singulars[name] = torch.linalg.svdvals(w)

    model = get_peft_model(copy.deepcopy(base), _make_kasa_config(r=r))

    for name in ["lin0", "lin1"]:
        lora_layer = getattr(model.base_model.model, name)
        new_weight = lora_layer.get_base_layer().weight.detach().float()
        sv = torch.linalg.svdvals(new_weight)
        k = min(new_weight.shape)
        # Exactly r singular values should be (numerically) zero -> rank dropped by r.
        n_zero = int((sv < 1e-5).sum())
        assert n_zero == r, f"{name}: expected {r} zeroed singular values, got {n_zero} (sv={sv})"
        # The surviving (largest) singular values should match the original principal ones.
        kept_new = torch.sort(sv, descending=True).values[: k - r]
        kept_orig = torch.sort(orig_singulars[name], descending=True).values[: k - r]
        assert torch.allclose(kept_new, kept_orig, atol=1e-4)


def test_kasa_truncation_changes_base_forward():
    """Adding a KaSA adapter destructively edits the base weight, so the clean (adapter-disabled) forward differs from
    the original model. This documents the (intentional) departure from the usual "disable == base" contract."""
    torch.manual_seed(0)
    base = MLP()
    x = torch.randn(5, 16)
    with torch.no_grad():
        orig_out = base(x)

    model = get_peft_model(copy.deepcopy(base), _make_kasa_config(r=4))
    model.eval()
    with torch.no_grad():
        with model.disable_adapter():
            disabled_out = model(x)

    # Because B == 0 at init, the *active* adapter output equals the truncated base output...
    with torch.no_grad():
        active_out = model(x)
    assert torch.allclose(active_out, disabled_out, atol=1e-6)
    # ...but the truncated base is NOT the original weight, so the output differs from the original model.
    assert not torch.allclose(disabled_out, orig_out, atol=1e-4)


# ----------------------------------------------------------------------------------------------------------------------
# Merge / unmerge round-trip and forward consistency
# ----------------------------------------------------------------------------------------------------------------------


def _randomize_adapter(model):
    """Give the adapter a non-trivial value so merge/forward differences are observable (B and diag both non-zero)."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoraLinear):
                nn.init.normal_(module.lora_B["default"].weight, std=0.1)
                module.lora_diag["default"].copy_(torch.randn_like(module.lora_diag["default"]))


def test_kasa_merge_unmerge_round_trip():
    torch.manual_seed(0)
    model = get_peft_model(copy.deepcopy(MLP()), _make_kasa_config(r=4))
    _randomize_adapter(model)
    model.eval()

    x = torch.randn(7, 16)
    with torch.no_grad():
        out_unmerged = model(x)

    # Capture the truncated base weights (merge/unmerge must round-trip to THESE, not the original weights).
    truncated = {
        name: getattr(model.base_model.model, name).get_base_layer().weight.detach().clone()
        for name in ["lin0", "lin1"]
    }

    model.merge_adapter()
    with torch.no_grad():
        out_merged = model(x)
    assert torch.allclose(out_unmerged, out_merged, atol=1e-5)

    model.unmerge_adapter()
    for name in ["lin0", "lin1"]:
        restored = getattr(model.base_model.model, name).get_base_layer().weight.detach()
        assert torch.allclose(restored, truncated[name], atol=1e-5)

    with torch.no_grad():
        out_after = model(x)
    assert torch.allclose(out_unmerged, out_after, atol=1e-5)


def test_kasa_delta_weight_matches_formula():
    """ΔW = scaling * B @ diag(lora_diag) @ A, and merging adds exactly this to the (truncated) base."""
    torch.manual_seed(0)
    model = get_peft_model(copy.deepcopy(MLP()), _make_kasa_config(r=4, lora_alpha=8))
    _randomize_adapter(model)

    for name in ["lin0", "lin1"]:
        layer = getattr(model.base_model.model, name)
        A = layer.lora_A["default"].weight.detach()
        B = layer.lora_B["default"].weight.detach()
        diag = layer.lora_diag["default"].detach()
        scaling = layer.scaling["default"]
        expected = scaling * (B @ torch.diag(diag) @ A)

        before = layer.get_base_layer().weight.detach().clone()
        layer.merge(safe_merge=True)
        after = layer.get_base_layer().weight.detach()
        assert torch.allclose(after - before, expected, atol=1e-5)


# ----------------------------------------------------------------------------------------------------------------------
# Save / load
# ----------------------------------------------------------------------------------------------------------------------


def test_kasa_lora_diag_in_state_dict(tmp_path):
    torch.manual_seed(0)
    model = get_peft_model(copy.deepcopy(MLP()), _make_kasa_config(r=4))
    _randomize_adapter(model)

    sd = get_peft_model_state_dict(model)
    diag_keys = [k for k in sd if "lora_diag" in k]
    # One lora_diag entry per targeted linear layer.
    assert len(diag_keys) == 2

    model.eval()
    x = torch.randn(3, 16)
    with torch.no_grad():
        out_before = model(x)

    save_dir = tmp_path / "kasa_adapter"
    model.save_pretrained(save_dir)

    # Reload onto a fresh base model that carries the (already truncated) base weights. Because KaSA mutates the base
    # weight in-place at adapter-creation time, the user must persist that truncated base alongside the adapter; here we
    # emulate that by copying the truncated weights into a clean MLP before loading the adapter.
    from peft import PeftModel

    reloaded_base = MLP()
    with torch.no_grad():
        for name in ["lin0", "lin1"]:
            truncated_w = getattr(model.base_model.model, name).get_base_layer().weight.detach().clone()
            getattr(reloaded_base, name).weight.copy_(truncated_w)

    reloaded = PeftModel.from_pretrained(reloaded_base, save_dir)
    reloaded.eval()
    with torch.no_grad():
        out_after = reloaded(x)
    assert torch.allclose(out_before, out_after, atol=1e-5)


@pytest.mark.parametrize("low_cpu_mem_usage", [False, True])
def test_kasa_reload_onto_original_base_retruncates(tmp_path, low_cpu_mem_usage):
    """Reloading the adapter onto the *original* (un-truncated) base must reproduce the trained output, because the
    deterministic SVD truncation is re-applied at load time. With low_cpu_mem_usage=True the truncation is deferred to
    the first forward; this guards against it being silently skipped on that path."""
    from peft import PeftModel

    torch.manual_seed(0)
    base = MLP()
    original_state = copy.deepcopy(base.state_dict())
    model = get_peft_model(copy.deepcopy(base), _make_kasa_config(r=4))
    _randomize_adapter(model)
    model.eval()
    x = torch.randn(3, 16)
    with torch.no_grad():
        out_before = model(x)

    save_dir = tmp_path / "kasa_adapter"
    model.save_pretrained(save_dir)

    # Fresh base carrying the ORIGINAL (un-truncated) weights - the realistic reload scenario.
    fresh_base = MLP()
    fresh_base.load_state_dict(original_state)
    reloaded = PeftModel.from_pretrained(fresh_base, save_dir, low_cpu_mem_usage=low_cpu_mem_usage)
    reloaded.eval()
    with torch.no_grad():
        out_after = reloaded(x)
        # A second forward must be stable (truncation applied exactly once, no double-truncation).
        out_after2 = reloaded(x)
    assert torch.allclose(out_before, out_after, atol=1e-5)
    assert torch.allclose(out_after, out_after2, atol=1e-6)


# ----------------------------------------------------------------------------------------------------------------------
# Regularization helper (L2 singular-value penalty + L3 orthogonal regularization)
# ----------------------------------------------------------------------------------------------------------------------


def test_kasa_regularization_zero_when_no_kasa_layers():
    torch.manual_seed(0)
    model = get_peft_model(copy.deepcopy(MLP()), LoraConfig(target_modules=["lin0"], r=4))
    loss = get_kasa_regularization_loss(model)
    assert float(loss) == 0.0


def test_kasa_regularization_l2_matches_closed_form():
    """With gamma=0 (and orthonormal A/B so L3=0 anyway), the loss reduces to beta * sum(lora_diag**2)."""
    torch.manual_seed(0)
    beta = 0.3
    model = get_peft_model(copy.deepcopy(MLP()), _make_kasa_config(r=4, beta=beta, gamma=0.0))
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoraLinear):
                module.lora_diag["default"].copy_(torch.arange(1.0, 5.0))  # [1,2,3,4]

    expected_per_layer = beta * (1.0**2 + 2.0**2 + 3.0**2 + 4.0**2)  # = beta * 30
    expected = 2 * expected_per_layer  # two layers
    loss = get_kasa_regularization_loss(model, gamma=0.0)
    assert pytest.approx(loss.item(), rel=1e-5) == expected


def test_kasa_orthogonal_reg_zero_for_orthonormal_factors():
    """L3 = ||B^T B - I|| + ||A A^T - I|| must be ~0 when A and B have orthonormal rows/cols, and > 0 otherwise."""
    torch.manual_seed(0)
    # Use square-ish factors so A (r x in) can have orthonormal rows and B (out x r) orthonormal columns.
    model = get_peft_model(copy.deepcopy(MLP(in_features=16, hidden=12, out_features=12)), _make_kasa_config(r=4))

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoraLinear):
                A = module.lora_A["default"].weight  # (r, in)
                B = module.lora_B["default"].weight  # (out, r)
                # orthonormal rows of A
                qa, _ = torch.linalg.qr(A.T)  # (in, r) with orthonormal columns
                module.lora_A["default"].weight.copy_(qa[:, : A.shape[0]].T)
                # orthonormal columns of B
                qb, _ = torch.linalg.qr(B)  # (out, r) with orthonormal columns
                module.lora_B["default"].weight.copy_(qb)
                module.lora_diag["default"].zero_()  # kill L2 so we isolate L3

    loss_ortho = get_kasa_regularization_loss(model, beta=0.0, gamma=1.0)
    assert loss_ortho.item() < 1e-4

    # Now make B clearly non-orthonormal and confirm the penalty becomes strictly positive.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoraLinear):
                module.lora_B["default"].weight.mul_(3.0)
    loss_non_ortho = get_kasa_regularization_loss(model, beta=0.0, gamma=1.0)
    assert loss_non_ortho.item() > 1e-3


def test_kasa_regularization_has_gradients():
    """The regularization loss must be differentiable w.r.t. the KaSA parameters."""
    torch.manual_seed(0)
    model = get_peft_model(copy.deepcopy(MLP()), _make_kasa_config(r=4))
    _randomize_adapter(model)

    loss = get_kasa_regularization_loss(model)
    loss.backward()
    for module in model.modules():
        if isinstance(module, LoraLinear):
            assert module.lora_diag["default"].grad is not None
            assert module.lora_A["default"].weight.grad is not None
