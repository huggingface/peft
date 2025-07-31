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


class DummyConfig:
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
