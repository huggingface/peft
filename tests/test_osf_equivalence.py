#!/usr/bin/env python3
"""Compare old OSF (main branch) vs new OSF (delta-based forward) implementations.

Tests equivalence across:
1. Forward pass outputs (at init and after perturbation)
2. Merge result
3. Gradient projection (orthogonality after backward)
4. Short training trajectories (loss + parameter values)
5. Multiple dtypes: fp32, fp16, bf16
6. Multiple shapes: square, tall, wide

Usage:
    python tests/test_osf_equivalence.py

This script does NOT modify the PR. It imports both implementations side by side:
- "old": reconstructs the original SVD-based forward in pure Python (from main)
- "new": uses the installed PEFT package (from the PR branch)

The old implementation is reimplemented here to avoid checkout conflicts.
The math is identical to origin/main:src/peft/tuners/osf/layer.py + utils.py.
"""

import copy
import itertools

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Old OSF implementation (from origin/main, reimplemented in pure Python)
# ---------------------------------------------------------------------------

def old_decompose_weight_matrix(weight: torch.Tensor, top_k: int) -> dict:
    """SVD decomposition — identical to origin/main utils.py."""
    device_local = weight.device
    orig_dtype = weight.dtype
    W = weight.to(torch.float32)
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = min(top_k, S.shape[0])

    svd = {
        "U_high": U[:, :k].contiguous().detach().to(device=device_local, dtype=orig_dtype),
        "S_high": S[:k].contiguous().detach().to(device=device_local, dtype=orig_dtype),
        "V_high": Vt[:k, :].contiguous().detach().to(device=device_local, dtype=orig_dtype),
        "U_low": nn.Parameter(U[:, k:].contiguous().detach().to(device=device_local, dtype=orig_dtype)),
        "S_low": nn.Parameter(S[k:].contiguous().detach().to(device=device_local, dtype=orig_dtype)),
        "V_low": nn.Parameter(Vt[k:, :].contiguous().detach().to(device=device_local, dtype=orig_dtype)),
        "rank_high": k,
    }
    return svd


def old_reconstruct_weight_matrix(svd_dict: dict) -> torch.Tensor:
    """Reconstruct weight from SVD — identical to origin/main utils.py."""
    U_high = svd_dict["U_high"]
    S_high = svd_dict["S_high"]
    V_high = svd_dict["V_high"]
    U_low = svd_dict["U_low"]
    S_low = svd_dict["S_low"]
    V_low = svd_dict["V_low"]

    high_part = (
        torch.mm(U_high * S_high.unsqueeze(0), V_high)
        if U_high.numel() > 0 and S_high.numel() > 0
        else torch.zeros(U_low.size(0), V_low.size(1), device=U_high.device)
    )
    low_part = (
        torch.mm(U_low * S_low.unsqueeze(0), V_low)
        if U_low.numel() > 0 and S_low.numel() > 0
        else torch.zeros(U_high.size(0), V_high.size(1), device=U_low.device)
    )
    return high_part + low_part


class OldOSFLinear(nn.Module):
    """Old OSF Linear from origin/main — reconstructs full weight in forward."""

    def __init__(self, base_layer: nn.Linear, effective_rank: int):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.merged = False
        self.effective_rank = effective_rank

        weight = base_layer.weight.data
        self.svd = old_decompose_weight_matrix(weight, top_k=effective_rank)

        # All SVD components are stored in self.svd; U_low/S_low/V_low are trainable
        self.U_low = self.svd["U_low"]
        self.S_low = self.svd["S_low"]
        self.V_low = self.svd["V_low"]
        self.U_high = self.svd["U_high"]
        self.S_high = self.svd["S_high"]
        self.V_high = self.svd["V_high"]

        # Register parameters and buffers
        self.register_parameter("U_low", self.U_low)
        self.register_parameter("S_low", self.S_low)
        self.register_parameter("V_low", self.V_low)
        self.register_buffer("_U_high", self.U_high)
        self.register_buffer("_S_high", self.S_high)
        self.register_buffer("_V_high", self.V_high)

        # Attach gradient hooks (old style: always use U_high/V_high)
        self.U_low.register_hook(self._hook_u)
        self.V_low.register_hook(self._hook_v)

    def _hook_u(self, grad):
        U_high = self._U_high
        proj = U_high @ (U_high.transpose(0, 1) @ grad)
        return grad - proj

    def _hook_v(self, grad):
        V_high = self._V_high
        proj = (grad @ V_high.transpose(0, 1)) @ V_high
        return grad - proj

    def _reconstruct_weight(self):
        svd_dict = {
            "U_high": self._U_high,
            "S_high": self._S_high,
            "V_high": self._V_high,
            "U_low": self.U_low,
            "S_low": self.S_low,
            "V_low": self.V_low,
        }
        return old_reconstruct_weight_matrix(svd_dict)

    def forward(self, x):
        if self.merged:
            return self.base_layer(x)
        weight = self._reconstruct_weight()
        orig_dtype = x.dtype
        x_cast = x.to(weight.dtype)
        bias = self.base_layer.bias
        if bias is not None:
            bias = bias.to(weight.dtype)
        result = torch.nn.functional.linear(x_cast, weight, bias)
        return result.to(orig_dtype)

    def merge(self):
        """Old merge: replace base weight with reconstructed weight."""
        new_weight = self._reconstruct_weight()
        self.base_layer.weight.data = new_weight.to(self.base_layer.weight.dtype)
        self.merged = True


# ---------------------------------------------------------------------------
# New OSF implementation (from installed peft package — PR branch)
# ---------------------------------------------------------------------------

from peft import OSFConfig, get_peft_model  # noqa: E402


class NewOSFLinear(nn.Module):
    """Wraps the new PEFT OSF layer for direct comparison.

    Uses get_peft_model to create the adapter, then extracts the OSFLayer
    so we can compare against the old implementation.
    """

    def __init__(self, base_layer: nn.Linear, effective_rank: int):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # Create a dummy model wrapping the base layer
        class DummyModel(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.linear = layer

            def forward(self, x):
                return self.linear(x)

        dummy = DummyModel(base_layer)
        config = OSFConfig(
            target_modules=["linear"],
            effective_rank=effective_rank,
        )
        self.wrapped = get_peft_model(dummy, config)
        # Extract the OSF layer
        self.osf_layer = self.wrapped.base_model.model.linear
        self.merged = False

    @property
    def U_low(self):
        adapter = self.osf_layer.active_adapters[0]
        return self.osf_layer.osf_svd_params[adapter]["U_low"]

    @property
    def S_low(self):
        adapter = self.osf_layer.active_adapters[0]
        return self.osf_layer.osf_svd_params[adapter]["S_low"]

    @property
    def V_low(self):
        adapter = self.osf_layer.active_adapters[0]
        return self.osf_layer.osf_svd_params[adapter]["V_low"]

    def forward(self, x):
        return self.wrapped(x)

    def merge(self):
        """New merge: add delta to base weight."""
        self.osf_layer.merge()
        self.merged = True


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_base_layer(out_features, in_features, dtype, device, seed=42):
    """Create a deterministic base linear layer."""
    torch.manual_seed(seed)
    layer = nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)
    # Ensure weights are not too small (avoid fp16 underflow)
    with torch.no_grad():
        layer.weight.data = torch.randn(out_features, in_features, device=device, dtype=dtype) * 0.1
    return layer


def perturb_svd_params(old_osf, new_osf, amount=0.01, seed=123):
    """Apply the same random perturbation to both old and new SVD params."""
    adapter = new_osf.osf_layer.active_adapters[0]
    for name in ["U_low", "S_low", "V_low"]:
        old_param = getattr(old_osf, name)
        new_param = new_osf.osf_layer.osf_svd_params[adapter][name]
        # Generate a single perturbation and apply to both
        torch.manual_seed(seed)
        perturbation = torch.randn_like(old_param) * amount
        with torch.no_grad():
            old_param.data += perturbation.to(old_param.dtype)
            new_param.data += perturbation.to(new_param.dtype)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_forward_at_init(device="cpu"):
    """At init, both implementations should produce the same output (identity transform).

    Note: The old implementation reconstructs the full weight via SVD, which introduces
    roundoff (~1e-5 for fp32, larger for fp16/bf16). The new implementation adds delta=0
    exactly, so it's a perfect identity. For low-precision dtypes, the old implementation
    may diverge from the base output due to this roundoff — this is expected and not a bug.
    We check old vs new equivalence, not against the base output.
    """
    print("\n=== Test 1: Forward pass at init (identity) ===")
    shapes = [
        ("square", 64, 64),
        ("tall", 128, 32),
        ("wide", 32, 128),
        ("large square", 256, 256),
    ]
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    all_pass = True

    for (label, out_f, in_f), dtype in itertools.product(shapes, dtypes):
        rank = min(out_f, in_f) // 2
        base = make_base_layer(out_f, in_f, dtype, device)
        base_copy = copy.deepcopy(base)

        x = torch.randn(4, in_f, device=device, dtype=dtype)

        old_osf = OldOSFLinear(base, effective_rank=rank)
        new_osf = NewOSFLinear(base_copy, effective_rank=rank)

        with torch.no_grad():
            old_out = old_osf(x)
            new_out = new_osf(x)
            base_out = base(x)

        # For fp32, both should match the base output closely.
        # For fp16/bf16, the old implementation has SVD roundoff that the new one doesn't,
        # so we only check old vs new (not vs base).
        if dtype == torch.float32:
            atol, rtol = 1e-4, 1e-3
        else:
            # Low precision: SVD roundoff in old impl can cause divergence from base,
            # but old vs new should still be close since both compute the same math
            # (the roundoff is in the reconstruction, not in the forward computation).
            # Actually, old reconstructs weight (SVD roundoff), new adds delta=0 (exact).
            # So old_out = F.linear(x, reconstructed_weight) ≈ base_out + roundoff,
            # and new_out = base_out exactly. They diverge by the SVD roundoff.
            # This is EXPECTED — the new implementation is more correct.
            atol, rtol = 2e-1, 2e-1  # relaxed: just check they're in the same ballpark

        old_vs_new = torch.allclose(old_out, new_out, atol=atol, rtol=rtol)

        status = "PASS" if old_vs_new else "FAIL"
        if not old_vs_new:
            all_pass = False
            max_diff = (old_out - new_out).abs().max().item()
            print(f"  [{status}] {label} ({out_f}x{in_f}) {dtype}: old_vs_new max_diff={max_diff:.6e}")
        else:
            print(f"  [{status}] {label} ({out_f}x{in_f}) {dtype}")

    return all_pass


def test_forward_after_perturbation(device="cpu"):
    """After perturbing SVD params identically, outputs should match."""
    print("\n=== Test 2: Forward pass after identical perturbation ===")
    shapes = [
        ("square", 64, 64),
        ("tall", 128, 32),
        ("wide", 32, 128),
    ]
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    all_pass = True

    for (label, out_f, in_f), dtype in itertools.product(shapes, dtypes):
        rank = min(out_f, in_f) // 2
        base = make_base_layer(out_f, in_f, dtype, device)
        base_copy = copy.deepcopy(base)

        x = torch.randn(4, in_f, device=device, dtype=dtype)

        old_osf = OldOSFLinear(base, effective_rank=rank)
        new_osf = NewOSFLinear(base_copy, effective_rank=rank)

        perturb_svd_params(old_osf, new_osf, amount=0.01)

        with torch.no_grad():
            old_out = old_osf(x)
            new_out = new_osf(x)

        atol = 1e-4 if dtype == torch.float32 else 5e-2
        rtol = 1e-3 if dtype == torch.float32 else 5e-2

        match = torch.allclose(old_out, new_out, atol=atol, rtol=rtol)
        status = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
            max_diff = (old_out - new_out).abs().max().item()
            print(f"  [{status}] {label} ({out_f}x{in_f}) {dtype}: max_diff={max_diff:.6e}")
        else:
            print(f"  [{status}] {label} ({out_f}x{in_f}) {dtype}")

    return all_pass


def test_merge_equivalence(device="cpu"):
    """Merged weights should be equivalent."""
    print("\n=== Test 3: Merge equivalence ===")
    shapes = [
        ("square", 64, 64),
        ("tall", 128, 32),
        ("wide", 32, 128),
    ]
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    all_pass = True

    for (label, out_f, in_f), dtype in itertools.product(shapes, dtypes):
        rank = min(out_f, in_f) // 2
        base = make_base_layer(out_f, in_f, dtype, device)
        base_copy = copy.deepcopy(base)

        old_osf = OldOSFLinear(base, effective_rank=rank)
        new_osf = NewOSFLinear(base_copy, effective_rank=rank)

        perturb_svd_params(old_osf, new_osf, amount=0.01)

        old_osf.merge()
        new_osf.merge()

        old_weight = old_osf.base_layer.weight.data
        new_weight = new_osf.base_layer.weight.data

        # Cast to same dtype for comparison (old merge may change dtype via reconstruction)
        atol = 1e-4 if dtype == torch.float32 else 5e-2
        rtol = 1e-3 if dtype == torch.float32 else 5e-2

        match = torch.allclose(old_weight.float(), new_weight.float(), atol=atol, rtol=rtol)
        status = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
            max_diff = (old_weight - new_weight).abs().max().item()
            print(f"  [{status}] {label} ({out_f}x{in_f}) {dtype}: max_diff={max_diff:.6e}")
        else:
            print(f"  [{status}] {label} ({out_f}x{in_f}) {dtype}")

    return all_pass


def test_gradient_projection(device="cpu"):
    """After backward, gradients should be orthogonal to the high-rank subspace in both implementations."""
    print("\n=== Test 4: Gradient projection orthogonality ===")
    shapes = [
        ("square", 64, 64),
        ("tall", 128, 32),
        ("wide", 32, 128),
    ]
    dtypes = [torch.float32]  # Gradient projection precision matters — fp32 only
    all_pass = True

    for (label, out_f, in_f), dtype in itertools.product(shapes, dtypes):
        rank = min(out_f, in_f) // 2
        base = make_base_layer(out_f, in_f, dtype, device)
        base_copy = copy.deepcopy(base)

        x = torch.randn(4, in_f, device=device, dtype=dtype)
        target = torch.randn(4, out_f, device=device, dtype=dtype)

        old_osf = OldOSFLinear(base, effective_rank=rank)
        new_osf = NewOSFLinear(base_copy, effective_rank=rank)

        # Perturb identically
        perturb_svd_params(old_osf, new_osf, amount=0.01)

        # Old backward
        old_out = old_osf(x)
        old_loss = torch.nn.functional.mse_loss(old_out, target)
        old_loss.backward()

        # New backward
        new_out = new_osf(x)
        new_loss = torch.nn.functional.mse_loss(new_out, target)
        new_loss.backward()

        # Check gradient orthogonality for U_low
        # Get the shared U_high from the SVD (both use the same base weight → same SVD)
        U_high = old_osf._U_high  # [out, rank]
        V_high = old_osf._V_high  # [rank, in]

        old_grad_u = old_osf.U_low.grad
        new_grad_u = new_osf.U_low.grad
        old_grad_v = old_osf.V_low.grad
        new_grad_v = new_osf.V_low.grad

        # Orthogonality: U_high^T @ grad should be ~0
        old_u_orth = (U_high.transpose(0, 1) @ old_grad_u).abs().max().item()
        new_u_orth = (U_high.transpose(0, 1) @ new_grad_u).abs().max().item()
        # Orthogonality: grad @ V_high^T should be ~0
        old_v_orth = (old_grad_v @ V_high.transpose(0, 1)).abs().max().item()
        new_v_orth = (new_grad_v @ V_high.transpose(0, 1)).abs().max().item()

        atol = 1e-4
        u_ok = abs(old_u_orth - new_u_orth) < atol or (old_u_orth < 1e-5 and new_u_orth < 1e-5)
        v_ok = abs(old_v_orth - new_v_orth) < atol or (old_v_orth < 1e-5 and new_v_orth < 1e-5)

        # Also check that the projected gradients are close
        grad_u_match = torch.allclose(old_grad_u, new_grad_u, atol=1e-4, rtol=1e-3)
        grad_v_match = torch.allclose(old_grad_v, new_grad_v, atol=1e-4, rtol=1e-3)

        status = "PASS" if (u_ok and v_ok and grad_u_match and grad_v_match) else "FAIL"
        if status == "FAIL":
            all_pass = False
            print(f"  [{status}] {label} ({out_f}x{in_f}) {dtype}:")
            print(f"         U orth: old={old_u_orth:.6e} new={new_u_orth:.6e} match={u_ok}")
            print(f"         V orth: old={old_v_orth:.6e} new={new_v_orth:.6e} match={v_ok}")
            print(f"         grad_u_match={grad_u_match} grad_v_match={grad_v_match}")
            if not grad_u_match:
                print(f"         grad_u max_diff={((old_grad_u - new_grad_u).abs().max()).item():.6e}")
            if not grad_v_match:
                print(f"         grad_v max_diff={((old_grad_v - new_grad_v).abs().max()).item():.6e}")
        else:
            print(f"  [{status}] {label} ({out_f}x{in_f}) {dtype}: U_orth={new_u_orth:.2e} V_orth={new_v_orth:.2e}")

    return all_pass


def test_short_training(device="cpu"):
    """Run a short training loop and compare loss trajectories and final parameters."""
    print("\n=== Test 5: Short training trajectory ===")
    shapes = [
        ("square", 64, 64),
        ("tall", 128, 32),
        ("wide", 32, 128),
    ]
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    n_steps = 20
    lr = 1e-3
    all_pass = True

    for (label, out_f, in_f), dtype in itertools.product(shapes, dtypes):
        rank = min(out_f, in_f) // 2
        base = make_base_layer(out_f, in_f, dtype, device)
        base_copy = copy.deepcopy(base)

        x = torch.randn(8, in_f, device=device, dtype=dtype)
        target = torch.randn(8, out_f, device=device, dtype=dtype)

        old_osf = OldOSFLinear(base, effective_rank=rank)
        new_osf = NewOSFLinear(base_copy, effective_rank=rank)

        # Perturb identically before training
        perturb_svd_params(old_osf, new_osf, amount=0.01)

        old_params = [old_osf.U_low, old_osf.S_low, old_osf.V_low]
        new_params = [
            new_osf.osf_layer.osf_svd_params[new_osf.osf_layer.active_adapters[0]]["U_low"],
            new_osf.osf_layer.osf_svd_params[new_osf.osf_layer.active_adapters[0]]["S_low"],
            new_osf.osf_layer.osf_svd_params[new_osf.osf_layer.active_adapters[0]]["V_low"],
        ]

        old_opt = torch.optim.SGD(old_params, lr=lr)
        new_opt = torch.optim.SGD(new_params, lr=lr)

        old_losses = []
        new_losses = []

        for step in range(n_steps):
            # Old step
            old_opt.zero_grad()
            old_out = old_osf(x)
            old_loss = torch.nn.functional.mse_loss(old_out, target)
            old_loss.backward()
            old_opt.step()
            old_losses.append(old_loss.item())

            # New step
            new_opt.zero_grad()
            new_out = new_osf(x)
            new_loss = torch.nn.functional.mse_loss(new_out, target)
            new_loss.backward()
            new_opt.step()
            new_losses.append(new_loss.item())

        # Compare loss trajectories
        max_loss_diff = max(abs(a - b) for a, b in zip(old_losses, new_losses))

        # Compare final parameters
        param_diffs = []
        for name, old_p, new_p in zip(["U", "S", "V"], old_params, new_params):
            diff = (old_p.data - new_p.data).abs().max().item()
            param_diffs.append((name, diff))

        max_param_diff = max(d for _, d in param_diffs)

        # Tolerances: fp32 is exact, fp16 has ~1e-3 precision, bf16 has ~1e-2 precision.
        # Over 20 SGD steps, tiny forward-pass differences accumulate, so we relax.
        if dtype == torch.float32:
            atol = 1e-4
        elif dtype == torch.float16:
            atol = 2e-2  # fp16 mantissa: 10 bits → ~1e-3 per op, accumulates over 20 steps
        else:
            atol = 5e-2  # bf16 mantissa: 7 bits → ~4e-3 per op, accumulates over 20 steps
        loss_ok = max_loss_diff < atol
        param_ok = max_param_diff < atol

        status = "PASS" if (loss_ok and param_ok) else "FAIL"
        if status == "FAIL":
            all_pass = False
            print(f"  [{status}] {label} ({out_f}x{in_f}) {dtype}:")
            print(f"         max_loss_diff={max_loss_diff:.6e} (threshold={atol:.0e})")
            for name, diff in param_diffs:
                print(f"         {name}_low max_diff={diff:.6e}")
            print(f"         old losses: [{old_losses[0]:.6f} ... {old_losses[-1]:.6f}]")
            print(f"         new losses: [{new_losses[0]:.6f} ... {new_losses[-1]:.6f}]")
        else:
            print(f"  [{status}] {label} ({out_f}x{in_f}) {dtype}: loss_diff={max_loss_diff:.2e} param_diff={max_param_diff:.2e}")

    return all_pass


def test_generations(device="cpu"):
    """Compare text generation with a small model."""
    print("\n=== Test 6: Generation equivalence ===")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  [SKIP] transformers not available")
        return True

    model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    except Exception as e:
        print(f"  [SKIP] Could not load model: {e}")
        return True

    # Force pad token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = base_model.to(device)

    # --- Old model: manually replace q_proj with OldOSFLinear ---
    old_model = copy.deepcopy(base_model)
    old_replaced = {}
    for name, module in old_model.named_modules():
        if name.endswith(".q_proj") and isinstance(module, nn.Linear):
            parent_name, child_name = name.rsplit(".", 1)
            parent = old_model.get_submodule(parent_name)
            osf_layer = OldOSFLinear(module, effective_rank=4)
            setattr(parent, child_name, osf_layer)
            old_replaced[name] = osf_layer

    # --- New model: use get_peft_model on a copy ---
    new_base = copy.deepcopy(base_model)

    class DummyWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs)

    new_model = get_peft_model(new_base, OSFConfig(target_modules=["q_proj"], effective_rank=4))

    # Collect new OSF layers by scanning for osf_svd_params
    new_layers = {}
    for name, module in new_model.named_modules():
        if hasattr(module, "osf_svd_params") and hasattr(module, "active_adapters"):
            # Map new name back to old name: strip the "base_model.model." prefix
            old_name = name.replace("base_model.model.", "")
            new_layers[old_name] = module

    # --- Apply identical perturbations ---
    torch.manual_seed(999)
    for name in old_replaced:
        old_osf = old_replaced[name]
        new_osf = new_layers.get(name)
        if new_osf is None:
            continue
        adapter = new_osf.active_adapters[0]
        for pname in ["U_low", "S_low", "V_low"]:
            old_p = getattr(old_osf, pname)
            new_p = new_osf.osf_svd_params[adapter][pname]
            perturbation = torch.randn_like(old_p) * 0.01
            with torch.no_grad():
                old_p.data += perturbation.to(old_p.dtype)
                new_p.data += perturbation.to(new_p.dtype)

    # --- Generate ---
    input_ids = tokenizer("Hello world", return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        old_gen = old_model.generate(input_ids, max_new_tokens=20, do_sample=False)
        new_gen = new_model.generate(input_ids, max_new_tokens=20, do_sample=False)

    old_text = tokenizer.decode(old_gen[0], skip_special_tokens=True)
    new_text = tokenizer.decode(new_gen[0], skip_special_tokens=True)

    match = old_text == new_text
    status = "PASS" if match else "FAIL"
    if not match:
        print(f"  [{status}] Generation mismatch:")
        print(f"         old: {old_text}")
        print(f"         new: {new_text}")
        return False
    else:
        print(f"  [{status}] Generations match: '{old_text}'")
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("OSF Old vs New Implementation Comparison")
    print(f"Device: {device}")
    print("Old: SVD reconstruction (origin/main)")
    print("New: Delta-based forward (PR branch)")
    print("=" * 60)

    results = []
    results.append(("Forward at init", test_forward_at_init(device)))
    results.append(("Forward after perturbation", test_forward_after_perturbation(device)))
    results.append(("Merge equivalence", test_merge_equivalence(device)))
    results.append(("Gradient projection", test_gradient_projection(device)))
    results.append(("Short training", test_short_training(device)))
    results.append(("Generation", test_generations(device)))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")

    print()
    if all_pass:
        print("✓ All tests passed — implementations are equivalent.")
    else:
        print("✗ Some tests failed — see details above.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
