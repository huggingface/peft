"""
Unit tests for AdaMSS ASA (Adaptive Subspace Allocation) functionality.

Tests cover:
- update_importance: EMA-based importance score accumulation
- reset_importance: clearing accumulated importance scores
- update_and_allocate: full ASA flow (accumulate → global mask → reset)
"""

import torch
from torch import nn

from peft import AdamssConfig, get_peft_model
from peft.tuners.adamss.layer import AdamssLayer


class SimpleMLP(nn.Module):
    """Minimal MLP for testing."""

    def __init__(self, in_features=20, hidden=40, out_features=5):
        super().__init__()
        self.lin0 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(hidden, out_features)

    def forward(self, x):
        return self.lin1(self.relu(self.lin0(x)))


def _make_asa_model(target_modules=("lin0", "lin1"), r=8, num_subspaces=4, subspace_rank=1, **extra):
    """Create a simple model with ASA enabled."""
    base = SimpleMLP()
    # Defaults that can be overridden via **extra
    config_kwargs = {
        "target_modules": list(target_modules),
        "r": r,
        "num_subspaces": num_subspaces,
        "subspace_rank": subspace_rank,
        "init_weights": None,
        "use_asa": True,
        "asa_target_subspaces": 2,
        "init_warmup": 0,
        "final_warmup": 100,
        "mask_interval": 10,
    }
    config_kwargs.update(extra)
    config = AdamssConfig(**config_kwargs)
    return get_peft_model(base, config)


def _run_train_step(model, optimizer, in_features=20):
    """Run one full training step (forward + backward + optimizer)."""
    x = torch.randn(4, in_features)
    out = model(x)
    loss = out.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss


def _get_adamss_layers(model):
    """Collect all AdamssLayer modules in the model."""
    return [m for m in model.modules() if isinstance(m, AdamssLayer)]


class TestAdamssAsa:
    # -- update_importance --------------------------------------------------

    def test_importance_populated_after_update(self):
        """update_importance should populate exp_avg_ipt_A/B and exp_avg_unc_A/B."""
        model = _make_asa_model()

        # Forward+backward (no optimizer step needed, we just need gradients)
        x = torch.randn(4, 20)
        model(x).sum().backward()

        layers = _get_adamss_layers(model)
        assert len(layers) > 0
        layer = layers[0]
        adapter = "default"

        # Before update: importance lists should be all None
        assert all(v is None for v in layer.exp_avg_ipt_A[adapter])
        assert all(v is None for v in layer.exp_avg_unc_A[adapter])

        # Update importance
        layer.update_importance(adapter, importance_beta=0.85, uncertainty_beta=0.85)

        # After update: at least some entries should be populated
        assert any(v is not None for v in layer.exp_avg_ipt_A[adapter]), "exp_avg_ipt_A should have entries"
        assert any(v is not None for v in layer.exp_avg_unc_A[adapter]), "exp_avg_unc_A should have entries"

        # At least some scores should be non-zero (B was seeded)
        has_nonzero = any(v.abs().sum() > 0 for v in layer.exp_avg_ipt_A[adapter] if v is not None)
        assert has_nonzero, "At least some importance scores should be non-zero"

    def test_importance_accumulates_across_steps(self):
        """Multiple training steps should produce changing (EMA-accumulated) scores."""
        model = _make_asa_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

        layers = _get_adamss_layers(model)
        layer = layers[0]
        adapter = "default"

        # Step 1: train so B becomes non-zero
        x = torch.randn(4, 20)
        model(x).sum().backward()
        optimizer.step()
        optimizer.zero_grad()

        # Step 2: now gradients for A should be non-zero
        model(x).sum().backward()
        layer.update_importance(adapter, 0.85, 0.85)
        optimizer.step()
        optimizer.zero_grad()

        # Find first populated entry
        first_idx = next(i for i, v in enumerate(layer.exp_avg_ipt_A[adapter]) if v is not None)
        score_after_2 = layer.exp_avg_ipt_A[adapter][first_idx].clone()

        # Step 3: another update should change scores via EMA
        model(x).sum().backward()
        layer.update_importance(adapter, 0.85, 0.85)
        optimizer.step()
        optimizer.zero_grad()

        score_after_3 = layer.exp_avg_ipt_A[adapter][first_idx].clone()

        assert not torch.allclose(score_after_2, score_after_3), (
            "Importance should change between steps due to EMA accumulation"
        )

    # -- reset_importance ---------------------------------------------------

    def test_reset_clears_scores(self):
        """reset_importance should clear all accumulated scores."""
        model = _make_asa_model()

        x = torch.randn(4, 20)
        model(x).sum().backward()

        layers = _get_adamss_layers(model)
        layer = layers[0]
        adapter = "default"

        # Populate importance
        layer.update_importance(adapter, 0.85, 0.85)
        assert any(v is not None for v in layer.exp_avg_ipt_A[adapter])

        # Reset
        layer.reset_importance(adapter)

        # After reset: all entries should be None
        assert all(v is None for v in layer.exp_avg_ipt_A[adapter]), "exp_avg_ipt_A should be all None after reset"
        assert all(v is None for v in layer.exp_avg_unc_A[adapter]), "exp_avg_unc_A should be all None after reset"

    # -- update_and_allocate ------------------------------------------------

    def test_importance_accumulated_every_step(self):
        """update_and_allocate should accumulate importance on non-mask-interval steps."""
        model = _make_asa_model(init_warmup=0, final_warmup=100, mask_interval=10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

        # Step 0: train to make B non-zero
        _run_train_step(model, optimizer)

        # Steps 1-2: in warmup, NOT a mask interval → should accumulate importance
        x = torch.randn(4, 20)
        model(x).sum().backward()
        optimizer.step()
        model.base_model.update_and_allocate(1)
        optimizer.zero_grad()

        layers = _get_adamss_layers(model)
        layer = layers[0]
        assert any(v is not None for v in layer.exp_avg_ipt_A["default"]), "Importance should be populated after step 1 (non-mask-interval)"

    def test_masking_reduces_active_params(self):
        """At mask intervals, some subspaces should be frozen."""
        model = _make_asa_model(
            init_warmup=1,
            final_warmup=100,
            mask_interval=5,
            asa_target_subspaces=2,
            num_subspaces=4,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

        # Count initially active params
        layers = _get_adamss_layers(model)
        initial_active = sum(1 for layer in layers for p in layer.adamss_A["default"] if p.requires_grad)

        # Train for several steps.  Step 0 warms up B (B=0 initially).
        # Steps 1-5 accumulate importance.  Step 5 hits mask_interval (5%5==0)
        # and triggers masking with meaningful scores.
        for step in range(6):
            x = torch.randn(4, 20)
            model(x).sum().backward()
            optimizer.step()
            model.base_model.update_and_allocate(step)
            optimizer.zero_grad()

        # After masking: should have fewer active params
        final_active = sum(1 for layer in layers for p in layer.adamss_A["default"] if p.requires_grad)
        final_frozen = sum(1 for layer in layers for p in layer.adamss_A["default"] if not p.requires_grad)

        assert final_frozen > 0, "Expected some subspace parameters to be frozen by ASA"
        assert final_active < initial_active, f"Active params should decrease: {initial_active} → {final_active}"

    def test_importance_reset_after_masking(self):
        """After a mask interval, importance should be reset for fresh accumulation."""
        model = _make_asa_model(init_warmup=1, final_warmup=100, mask_interval=5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

        # Run to step 5 which triggers masking (5 % 5 == 0) and then reset
        for step in range(6):
            x = torch.randn(4, 20)
            model(x).sum().backward()
            optimizer.step()
            model.base_model.update_and_allocate(step)
            optimizer.zero_grad()

        # After mask interval at step 5: importance should be cleared
        layers = _get_adamss_layers(model)
        for layer in layers:
            assert all(v is None for v in layer.exp_avg_ipt_A["default"]), "Importance should be reset after mask interval"

    def test_no_masking_outside_warmup(self):
        """update_and_allocate should be a no-op outside warmup range."""
        model = _make_asa_model(init_warmup=50, final_warmup=100, mask_interval=10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

        # Run step 10 (mask_interval hit but BEFORE init_warmup=50)
        _run_train_step(model, optimizer)
        model.base_model.update_and_allocate(10)

        # No importance should be accumulated (outside warmup)
        layers = _get_adamss_layers(model)
        for layer in layers:
            assert all(v is None for v in layer.exp_avg_ipt_A["default"]), "No importance accumulation should happen outside warmup"

    def test_asa_disabled_is_noop(self):
        """update_and_allocate should be a no-op when use_asa=False."""
        base = SimpleMLP()
        config = AdamssConfig(
            target_modules=["lin0"],
            r=8,
            num_subspaces=4,
            subspace_rank=1,
            use_asa=False,
        )
        model = get_peft_model(base, config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

        _run_train_step(model, optimizer)
        # Should not raise
        model.base_model.update_and_allocate(0)
        model.base_model.update_and_allocate(100)

        # All params still trainable
        layers = _get_adamss_layers(model)
        for layer in layers:
            for p in layer.adamss_A["default"]:
                assert p.requires_grad
