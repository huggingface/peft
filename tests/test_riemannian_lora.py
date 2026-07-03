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
    torch.manual_seed(seed)
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["lin0", "lin1"],
        use_dora=use_dora,
    )
    return get_peft_model(SimpleNet(), config).to(torch_device)


def _forward_backward(model):
    loss = nn.CrossEntropyLoss()
    x = torch.randint(100, (2, 4, 10)).to(torch_device)
    output = model(x).permute(0, 3, 1, 2)
    label = torch.randint(16, (2, 4, 10)).to(torch_device)
    loss(output, label).backward()


class TestRiemannianOptimizer:
    def test_factory_creates_optimizer(self):
        model = _make_lora_model()
        optim = create_riemannian_optimizer(model, torch.optim.AdamW, lr=1e-3)
        assert isinstance(optim, torch.optim.Optimizer)
        # Only trainable params are in the optimizer (base weights are frozen by LoRA).
        n_optim_params = sum(len(g["params"]) for g in optim.param_groups)
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert n_optim_params == n_trainable

    def test_step_runs_and_updates_lora_params(self):
        # LoRA's standard init has lora_B == 0, so g_A vanishes on step 1 and lora_A doesn't move — hence 2 steps.
        model = _make_lora_model()
        optim = create_riemannian_optimizer(model, torch.optim.AdamW, lr=1e-3)
        lora_pairs = _collect_lora_pairs(model)
        assert lora_pairs, "test fixture didn't produce any lora pairs"
        before = [p.detach().clone() for pair in lora_pairs for p in pair]

        for _ in range(2):
            optim.zero_grad()
            _forward_backward(model)
            optim.step()

        after = [p.detach().clone() for pair in lora_pairs for p in pair]
        for pre, post in zip(before, after):
            assert not torch.allclose(pre, post, atol=1e-6, rtol=1e-5), "LoRA weight did not update after two steps"

    @pytest.mark.parametrize("optimizer_cls", [torch.optim.AdamW, torch.optim.SGD])
    def test_works_with_any_optimizer_subclass(self, optimizer_cls):
        model = _make_lora_model()
        optim = create_riemannian_optimizer(model, optimizer_cls, lr=1e-2)
        assert isinstance(optim, optimizer_cls)

        _forward_backward(model)
        optim.step()  # smoke check — no exception

    def test_dora_magnitude_vector_is_left_alone_by_preconditioner(self):
        # DoRA adds a per-column lora_magnitude_vector alongside lora_A / lora_B; the pair collector must skip it
        # (still updated by the base optimizer, just not preconditioned).
        model = _make_lora_model(use_dora=True)

        magnitude_params = [
            name for name, p in model.named_parameters() if "lora_magnitude_vector" in name and p.requires_grad
        ]
        assert magnitude_params, "DoRA fixture didn't produce magnitude vectors"

        pairs = _collect_lora_pairs(model)
        for a, b in pairs:
            assert a.ndim == 2 and b.ndim == 2, "pair collector returned non-2D params"

        optim = create_riemannian_optimizer(model, torch.optim.AdamW, lr=1e-3)
        magnitudes_before = {
            name: p.detach().clone() for name, p in model.named_parameters() if "lora_magnitude_vector" in name
        }
        _forward_backward(model)
        optim.step()
        for name, before in magnitudes_before.items():
            after = dict(model.named_parameters())[name]
            assert not torch.allclose(before, after, atol=1e-6, rtol=1e-5), f"magnitude vector {name!r} did not update"

    def test_raises_when_no_lora_parameters_on_model(self):
        model = SimpleNet()
        with pytest.raises(ValueError, match="lora_A/lora_B parameter pairs"):
            create_riemannian_optimizer(model, torch.optim.AdamW, lr=1e-3)

    def test_raises_when_optimizer_cls_is_not_optimizer_subclass(self):
        model = _make_lora_model()
        with pytest.raises(TypeError, match="subclass of torch.optim.Optimizer"):
            create_riemannian_optimizer(model, dict, lr=1e-3)  # type: ignore[arg-type]

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_low_precision_gradients_preconditioned_stably(self, dtype):
        # bf16/fp16 factors on their own would overflow / NaN the small-r inverse; the preconditioner promotes to
        # float32 internally, so the resulting gradient must be finite and cast back to the input dtype.
        torch.manual_seed(0)
        r = 4
        lora_a = nn.Parameter(torch.randn(r, 6, dtype=dtype))
        lora_b = nn.Parameter(torch.randn(8, r, dtype=dtype))
        lora_a.grad = torch.randn_like(lora_a)
        lora_b.grad = torch.randn_like(lora_b)

        preconditioner = _RiemannianPreconditioner([(lora_a, lora_b)], reg=1e-4)
        preconditioner.step()

        assert torch.isfinite(lora_a.grad).all(), f"{dtype} preconditioner produced non-finite g_A"
        assert torch.isfinite(lora_b.grad).all(), f"{dtype} preconditioner produced non-finite g_B"
        assert lora_a.grad.dtype == dtype
        assert lora_b.grad.dtype == dtype
