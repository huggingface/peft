import torch
from torch import nn

from peft import AdamssConfig, get_peft_model
from peft.tuners.adamss.layer import AdamssLayer


def _model():
    base = nn.Sequential(nn.Linear(8, 8))
    config = AdamssConfig(
        target_modules=["0"],
        r=4,
        num_subspaces=2,
        subspace_rank=1,
        init_weights=None,
    )
    return get_peft_model(base, config)


def test_forward_follows_the_weight_dtype_after_cast():
    model = _model()
    layer = next(module for module in model.modules() if isinstance(module, AdamssLayer))
    x = torch.randn(2, 8)

    model(x)
    assert layer.dtype == torch.float32

    model.to(torch.float16)
    assert layer.dtype == torch.float16
    out = model(x)
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()

    model.to(torch.bfloat16)
    assert layer.dtype == torch.bfloat16
    out = model(x)
    assert out.dtype == x.dtype
    assert torch.isfinite(out.float()).all()

    model.to(torch.float32)
    assert layer.dtype == torch.float32
    out = model(torch.randn(2, 8))
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
