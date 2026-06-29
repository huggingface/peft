"""Regression test for peft#2049: DoRA on a Linear with zero or low-precision-underflowing rows must not NaN.

Drop into ``peft/tests/test_dora_bf16_zero_rows.py``. Both tests fail in peft 0.19.1 / main
without the get_weight_norm fix, and pass after the fix.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model


class _TinyLinear(nn.Module):
    def __init__(self, in_features: int = 64, out_features: int = 32):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.proj(x)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_dora_does_not_nan_on_zero_rows(dtype):
    """A Linear whose base weight contains all-zero rows must not produce inf/NaN under DoRA.

    Real-world incidence: Llama-3.1-8B ``o_proj`` has 511 zero rows (issue #2049);
    Qwen3.5/3.6 MoE ``linear_attn.in_proj_qkv`` has 434 zero rows across the SSM layers
    (concentrated in layers 0/1/2).
    """
    torch.manual_seed(0)
    model = _TinyLinear().to(dtype)
    with torch.no_grad():
        model.proj.weight[:8].zero_()

    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["proj"],
        use_dora=True,
        bias="none",
    )
    peft_model = get_peft_model(model, config)
    peft_model.eval()

    x = torch.randn(2, 64, dtype=dtype)
    with torch.no_grad():
        out = peft_model(x)
    assert torch.isfinite(out).all(), (
        f"DoRA forward produced non-finite output on Linear with zero rows (dtype={dtype})"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dora_does_not_nan_on_low_precision_underflow_rows(dtype):
    """Rows whose entries are nonzero in fp32 but round to zero in bf16/fp16 must also not NaN."""
    torch.manual_seed(0)
    model = _TinyLinear()  # fp32 init
    with torch.no_grad():
        model.proj.weight[:8] = 1e-30  # far below the smallest representable bf16/fp16 normal
    model = model.to(dtype)

    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["proj"],
        use_dora=True,
        bias="none",
    )
    peft_model = get_peft_model(model, config)
    peft_model.eval()

    x = torch.randn(2, 64, dtype=dtype)
    with torch.no_grad():
        out = peft_model(x)
    assert torch.isfinite(out).all(), (
        f"DoRA forward produced non-finite output on Linear with bf16/fp16-underflow rows (dtype={dtype})"
    )
