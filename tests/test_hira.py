import pytest
import torch
import torch.nn as nn

from peft.tuners.hira import Linear  # Assuming your HiRA implementation is under peft.tuners.hira
from peft.tuners.hira.layer import Conv1d as HiraConv1d
from peft.tuners.hira.layer import Conv2d as HiraConv2d


def test_hira_linear_merge_unmerge_basic():
    """
    Basic test for HiRA Linear layer: ensures merge and unmerge preserve outputs.
    """
    # Setup
    input_dim, output_dim, rank = 10, 5, 2
    adapter_name = "test_adapter"
    batch_size = 4

    # Base layer
    base_linear = nn.Linear(input_dim, output_dim)

    # Wrap base layer with HiRA
    hira_linear = Linear(
        base_layer=base_linear,
        adapter_name=adapter_name,
        r=rank,
        hira_alpha=rank,
        hira_dropout=0.0,
        init_hira_weights=True,
    )

    # Dummy input
    x = torch.randn(batch_size, input_dim)

    # Forward pass without merging
    output_before_merge = hira_linear(x)

    # Merge adapter weights
    hira_linear.merge()
    output_after_merge = hira_linear(x)

    # Assert merge preserves output
    assert torch.allclose(output_before_merge, output_after_merge, atol=1e-5), (
        "Merged HiRA Linear output doesn't match original"
    )

    # Unmerge adapter weights
    hira_linear.unmerge()
    output_after_unmerge = hira_linear(x)

    # Assert unmerge restores original output
    assert torch.allclose(output_before_merge, output_after_unmerge, atol=1e-5), (
        "Unmerged HiRA Linear output doesn't match original"
    )


@pytest.mark.parametrize(
    "batch_size,in_ch,out_ch,length,rank",
    [
        (2, 4, 6, 10, 3),
        (3, 2, 5, 8, 2),
    ],
)
def test_hira_conv1d_merge_unmerge(batch_size, in_ch, out_ch, length, rank):
    base_conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
    hira_conv = HiraConv1d(
        base_layer=base_conv,
        adapter_name="test_adapter",
        r=rank,
        hira_dropout=0.0,
        init_hira_weights=True,
    )
    x = torch.randn(batch_size, in_ch, length)

    # Before merge
    y0 = hira_conv(x)

    # Merge into W and test
    hira_conv.merge()
    y1 = hira_conv(x)
    assert torch.allclose(y0, y1, atol=1e-5), "Merged Conv1d HiRA output doesn't match original"

    # Unmerge and test
    hira_conv.unmerge()
    y2 = hira_conv(x)
    assert torch.allclose(y0, y2, atol=1e-5), "Unmerged Conv1d HiRA output doesn't match original"


@pytest.mark.parametrize(
    "batch_size,in_ch,out_ch,H,W,rank",
    [
        (2, 3, 5, 8, 8, 2),
        (1, 1, 4, 10, 10, 1),
    ],
)
def test_hira_conv2d_merge_unmerge(batch_size, in_ch, out_ch, H, W, rank):
    base_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
    hira_conv = HiraConv2d(
        base_layer=base_conv,
        adapter_name="test_adapter",
        r=rank,
        hira_dropout=0.0,
        init_hira_weights=True,
    )
    x = torch.randn(batch_size, in_ch, H, W)

    # Before merge
    y0 = hira_conv(x)

    # Merge into W and test
    hira_conv.merge()
    y1 = hira_conv(x)
    assert torch.allclose(y0, y1, atol=1e-5), "Merged Conv2d HiRA output doesn't match original"

    # Unmerge and test
    hira_conv.unmerge()
    y2 = hira_conv(x)
    assert torch.allclose(y0, y2, atol=1e-5), "Unmerged Conv2d HiRA output doesn't match original"


def test_manual_hira_linear_equivalence():
    import torch.nn.functional as F

    torch.manual_seed(42)
    batch_size, input_dim, output_dim, rank = 3, 8, 6, 2
    adapter_name = "manual_test"

    # create base linear and HiRA wrapper
    base = nn.Linear(input_dim, output_dim, bias=False)
    # init W0 to something deterministic
    nn.init.uniform_(base.weight, -0.5, 0.5)

    hira = Linear(
        base_layer=base,
        adapter_name=adapter_name,
        r=rank,
        hira_dropout=0.0,
        init_hira_weights=True,
    )
    # force A, B to known values
    with torch.no_grad():
        hira.hira_A[adapter_name].copy_(torch.randn(rank, input_dim))
        hira.hira_B[adapter_name].copy_(torch.randn(output_dim, rank))

    x = torch.randn(batch_size, input_dim)

    # HiRA forward (without merging)
    y_hira = hira(x)

    # manual forward
    W0 = base.weight.data  # (out, in)
    A = hira.hira_A[adapter_name]  # (r, in)
    B = hira.hira_B[adapter_name]  # (out, r)
    BA = B @ A  # (out, in)
    effW = W0 * BA  # element-wise
    # base output
    y0 = F.linear(x, W0)  # (batch, out)
    # delta output
    y_delta = F.linear(x, effW)
    y_manual = y0 + y_delta

    assert torch.allclose(y_hira, y_manual, atol=1e-6), (
        f"HiRA forward mismatch: max diff = {(y_hira - y_manual).abs().max()}"
    )
