# Copyright 2025-present the HuggingFace Inc. team.
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

import pytest
import torch
from torch import nn

from peft import LoraConfig, get_peft_model
from peft.optimizers import create_riemannian_optimizer
from peft.optimizers.riemannian import _collect_lora_pairs, _RiemannianPreconditioner

from .testing_utils import torch_device


class SimpleNet(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.embedding = nn.Embedding(100, 20)
        self.layer_norm = nn.LayerNorm(20)
        self.lin0 = nn.Linear(20, 20, bias=bias)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(20, 16, bias=bias)

    def forward(self, X):
        X = self.lin0(self.layer_norm(self.embedding(X)))
        X = self.relu(X)
        X = self.lin1(X)
        return X


def _make_lora_model(seed: int = 42, use_dora: bool = False):
    """Build a small deterministic LoRA-adapted SimpleNet."""
    torch.manual_seed(seed)
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["lin0", "lin1"],
        use_dora=use_dora,
    )
    return get_peft_model(SimpleNet(), config).to(torch_device)


def _forward_backward(model):
    """Run one forward + backward pass to populate gradients on `model`."""
    loss = nn.CrossEntropyLoss()
    x = torch.randint(100, (2, 4, 10)).to(torch_device)
    output = model(x).permute(0, 3, 1, 2)
    label = torch.randint(16, (2, 4, 10)).to(torch_device)
    loss(output, label).backward()


# ── factory: happy path ──────────────────────────────────────────────────────


def test_factory_creates_optimizer():
    model = _make_lora_model()
    optim = create_riemannian_optimizer(model, torch.optim.AdamW, lr=1e-3)
    assert isinstance(optim, torch.optim.Optimizer)
    # Only trainable params are in the optimizer (base weights are frozen by LoRA).
    n_optim_params = sum(len(g["params"]) for g in optim.param_groups)
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_optim_params == n_trainable


def test_step_runs_and_updates_lora_params():
    model = _make_lora_model()
    optim = create_riemannian_optimizer(model, torch.optim.AdamW, lr=1e-3)
    lora_pairs = _collect_lora_pairs(model)
    assert lora_pairs, "test fixture didn't produce any lora pairs"
    before = [p.detach().clone() for pair in lora_pairs for p in pair]

    # LoRA's standard init has lora_B == 0, so g_A vanishes on step 1 and
    # lora_A doesn't move. Take two steps so both lora_A and lora_B end up
    # with a non-zero gradient history and update.
    for _ in range(2):
        optim.zero_grad()
        _forward_backward(model)
        optim.step()

    after = [p.detach().clone() for pair in lora_pairs for p in pair]
    # Every LoRA weight should have moved after two preconditioned steps.
    for pre, post in zip(before, after):
        assert not torch.allclose(pre, post), "LoRA weight did not update after two steps"


# ── subclass permissiveness ──────────────────────────────────────────────────


@pytest.mark.parametrize("optimizer_cls", [torch.optim.AdamW, torch.optim.SGD])
def test_works_with_any_optimizer_subclass(optimizer_cls):
    model = _make_lora_model()
    optim = create_riemannian_optimizer(model, optimizer_cls, lr=1e-2)
    assert isinstance(optim, optimizer_cls)

    _forward_backward(model)
    optim.step()  # smoke check — no exception


# ── DoRA compatibility ──────────────────────────────────────────────────────


def test_dora_magnitude_vector_is_left_alone_by_preconditioner():
    """DoRA adds a per-column `lora_magnitude_vector` alongside lora_A / lora_B.

    The preconditioner is only defined for the low-rank product; the magnitude vector must be ignored by the
    pair-collection helper (still updated by the base optimizer, just not preconditioned).
    """
    model = _make_lora_model(use_dora=True)

    # Sanity: model has magnitude vectors.
    magnitude_params = [
        name for name, p in model.named_parameters() if "lora_magnitude_vector" in name and p.requires_grad
    ]
    assert magnitude_params, "DoRA fixture didn't produce magnitude vectors"

    pairs = _collect_lora_pairs(model)
    # No pair contains a magnitude vector.
    for a, b in pairs:
        assert a.ndim == 2 and b.ndim == 2, "pair collector returned non-2D params"

    # Full end-to-end: an optimization step should complete without error and
    # the magnitude vectors should also update (via the base optimizer path).
    optim = create_riemannian_optimizer(model, torch.optim.AdamW, lr=1e-3)
    magnitudes_before = {
        name: p.detach().clone() for name, p in model.named_parameters() if "lora_magnitude_vector" in name
    }
    _forward_backward(model)
    optim.step()
    for name, before in magnitudes_before.items():
        after = dict(model.named_parameters())[name]
        assert not torch.allclose(before, after), f"magnitude vector {name!r} did not update"


# ── error cases ──────────────────────────────────────────────────────────────


def test_raises_when_no_lora_parameters_on_model():
    model = SimpleNet()  # NOT wrapped in LoRA
    with pytest.raises(ValueError, match="lora_A/lora_B parameter pairs"):
        create_riemannian_optimizer(model, torch.optim.AdamW, lr=1e-3)


def test_raises_when_optimizer_cls_is_not_optimizer_subclass():
    model = _make_lora_model()
    with pytest.raises(TypeError, match="subclass of torch.optim.Optimizer"):
        create_riemannian_optimizer(model, dict, lr=1e-3)  # type: ignore[arg-type]


# ── preconditioner math ──────────────────────────────────────────────────────


def test_preconditioner_matches_paper_formula():
    """Verify the preconditioner produces `(B^T B + reg I)^-1 g_A` and
    `g_B (A A^T + reg I)^-1` up to floating-point tolerance."""
    torch.manual_seed(0)
    out_dim, r, in_dim = 8, 4, 6
    lora_a = nn.Parameter(torch.randn(r, in_dim))
    lora_b = nn.Parameter(torch.randn(out_dim, r))
    lora_a.grad = torch.randn_like(lora_a)
    lora_b.grad = torch.randn_like(lora_b)

    g_a_orig = lora_a.grad.clone()
    g_b_orig = lora_b.grad.clone()
    reg = 1e-6

    preconditioner = _RiemannianPreconditioner([(lora_a, lora_b)], reg=reg)
    preconditioner.step()

    a = lora_a.detach().to(torch.float32)
    b = lora_b.detach().to(torch.float32)
    eye_r = torch.eye(r)

    expected_g_a = torch.linalg.pinv(b.T @ b + reg * eye_r) @ g_a_orig.to(torch.float32)
    expected_g_b = g_b_orig.to(torch.float32) @ torch.linalg.pinv(a @ a.T + reg * eye_r)

    assert torch.allclose(lora_a.grad, expected_g_a.to(lora_a.dtype), atol=1e-5)
    assert torch.allclose(lora_b.grad, expected_g_b.to(lora_b.dtype), atol=1e-5)


def test_bf16_gradients_preconditioned_stably():
    """Even in bf16, the preconditioner must produce a finite result — internal
    compute promotes to float32 to avoid the numerical failure of small-r inverses in bf16."""
    torch.manual_seed(0)
    r = 4
    lora_a = nn.Parameter(torch.randn(r, 6, dtype=torch.bfloat16))
    lora_b = nn.Parameter(torch.randn(8, r, dtype=torch.bfloat16))
    lora_a.grad = torch.randn_like(lora_a)
    lora_b.grad = torch.randn_like(lora_b)

    preconditioner = _RiemannianPreconditioner([(lora_a, lora_b)], reg=1e-4)
    preconditioner.step()

    assert torch.isfinite(lora_a.grad).all(), "bf16 preconditioner produced non-finite g_A"
    assert torch.isfinite(lora_b.grad).all(), "bf16 preconditioner produced non-finite g_B"
    assert lora_a.grad.dtype == torch.bfloat16
    assert lora_b.grad.dtype == torch.bfloat16
