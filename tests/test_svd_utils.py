import torch
from torch.testing import assert_close

from peft.utils.svd_utils import (
    decompose_weight_matrix,
    reconstruct_weight_matrix,
)


def test_svd_roundtrip():
    w = torch.randn(10, 8)
    svd = decompose_weight_matrix(w, top_k=4)
    w_rec = reconstruct_weight_matrix(svd)
    assert_close(w_rec, w, atol=1e-5, rtol=1e-5)
