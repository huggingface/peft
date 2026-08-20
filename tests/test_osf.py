import pytest
import torch
from torch.testing import assert_close

from peft import OSFConfig, get_peft_model
from peft.tuners.osf.layer import OSFLayer
from peft.tuners.osf.utils import (
    decompose_weight_matrix,
)


def test_osf_roundtrip():
    w = torch.randn(10, 8)
    svd = decompose_weight_matrix(w, top_k=4)
    high_part = torch.mm(svd["U_high"] * svd["S_high"].unsqueeze(0), svd["V_high"])
    low_part = torch.mm(svd["U_low"] * svd["S_low"].unsqueeze(0), svd["V_low"])
    w_rec = high_part + low_part
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


def test_osf_gradient_projection_hook():
    torch.manual_seed(0)
    model = DummyModel(DummyConfig())
    # Specify target module explicitly for DummyModel
    # DummyModel.linear is nn.Linear(8, 4): weight shape is [4, 8], so out=4 < in=8.
    # U is square (recoverable from U_low_init), V is not square (V_high is stored).
    cfg = OSFConfig(target_modules=["linear"], effective_rank=2)
    wrapped = get_peft_model(model, cfg)
    x = torch.randn(3, 8)
    wrapped(x).sum().backward()
    # Access the injected OSF layer
    osf_linear = wrapped.base_model.model.linear
    adapter = wrapped.base_model.active_adapters[0]
    svd_params = osf_linear.osf_svd_params[adapter]

    # Check orthogonality of gradients after projection.
    # For the U factor (square case), projection uses U_low_init instead of U_high.
    # For the V factor (non-square case), projection uses the stored V_high.
    # In both cases, the projected gradient must be orthogonal to the high-rank subspace.
    # We verify by checking that the gradient is orthogonal to the original full SVD basis.

    # Reconstruct the full SVD to get the original high-rank subspace for verification
    base_weight = osf_linear.get_base_layer().weight.data
    svd_full = decompose_weight_matrix(base_weight, top_k=2)
    U_high_full = svd_full["U_high"]
    V_high_full = svd_full["V_high"]

    # U_low gradient should be orthogonal to U_high subspace
    proj_u = U_high_full.T @ svd_params["U_low"].grad
    assert_close(proj_u, torch.zeros_like(proj_u), atol=1e-5, rtol=1e-5)

    # V_low gradient should be orthogonal to V_high subspace
    proj_v = svd_params["V_low"].grad @ V_high_full.T
    assert_close(proj_v, torch.zeros_like(proj_v), atol=1e-5, rtol=1e-5)


def test_osf_merge_and_unload_and_unmerge_behavior():
    model = DummyModel(DummyConfig())
    cfg = OSFConfig(target_modules=["linear"], effective_rank=2)
    wrapped = get_peft_model(model, cfg)

    # merge_adapter should work via BaseTuner and OSFLayer.merge
    osf_linear = wrapped.base_model.model.linear
    assert isinstance(osf_linear, OSFLayer)
    wrapped.merge_adapter()
    assert osf_linear.merged, "OSF layer should be marked as merged after merge_adapter()"

    # unmerge_adapter is not supported for OSF
    with pytest.raises(NotImplementedError):
        wrapped.unmerge_adapter()

    # merge_and_unload should return the base model (no OSF wrappers)
    merged_model = wrapped.merge_and_unload()
    assert isinstance(merged_model.linear, torch.nn.Linear)
