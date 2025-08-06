from tempfile import TemporaryDirectory

import pytest
import torch
from torch.testing import assert_close

from peft import OSFConfig, get_peft_model
from peft.utils.osf_utils import (
    decompose_weight_matrix,
    reconstruct_weight_matrix,
)


def test_osf_roundtrip():
    w = torch.randn(10, 8)
    svd = decompose_weight_matrix(w, top_k=4)
    w_rec = reconstruct_weight_matrix(svd)
    assert_close(w_rec, w, atol=1e-5, rtol=1e-5)


class DummyConfig(dict):
    pass


class DummyModel(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.linear = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.linear(x)


def test_osf_get_peft_model_preserves_output():
    torch.manual_seed(0)
    model = DummyModel(DummyConfig())
    x = torch.randn(2, 8)
    y_ref = model(x)
    wrapped = get_peft_model(model, OSFConfig())
    y = wrapped(x)
    assert_close(y, y_ref, atol=1e-5, rtol=1e-5)


def test_osf_gradient_projection_hook():
    torch.manual_seed(0)
    model = DummyModel(DummyConfig())
    cfg = OSFConfig(target_svd_config={"linear.weight": 2})
    wrapped = get_peft_model(model, cfg)
    x = torch.randn(3, 8)
    wrapped(x).sum().backward()
    inner = wrapped.base_model.model
    safe_name = next(iter(inner.svd_params))
    module_svd = inner.svd_params[safe_name]
    U_high = getattr(inner, f"{safe_name}_U_high")
    V_high = getattr(inner, f"{safe_name}_V_high")
    assert_close(
        U_high.T @ module_svd.U_low.grad, torch.zeros_like(U_high.T @ module_svd.U_low.grad), atol=1e-6, rtol=1e-6
    )
    assert_close(
        module_svd.V_low.grad @ V_high.T,
        torch.zeros_like(module_svd.V_low.grad @ V_high.T),
        atol=1e-6,
        rtol=1e-6,
    )


def test_osf_config_roundtrip():
    cfg = OSFConfig(target_svd_config={"linear.weight": 2})
    with TemporaryDirectory() as tmp:
        cfg.save_pretrained(tmp)
        loaded = OSFConfig.from_pretrained(tmp)
    assert cfg.target_svd_config == loaded.target_svd_config


def test_osf_merge_unmerge_unsupported():
    model = DummyModel(DummyConfig())
    cfg = OSFConfig(target_svd_config={"linear.weight": 2})
    wrapped = get_peft_model(model, cfg)
    with pytest.raises(NotImplementedError):
        wrapped.merge_adapter()
    with pytest.raises(NotImplementedError):
        wrapped.unmerge_adapter()
    with pytest.raises(NotImplementedError):
        wrapped.merge_and_unload()