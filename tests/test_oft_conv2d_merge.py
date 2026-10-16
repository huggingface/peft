# Copyright 2026-present the HuggingFace Inc. team.
import copy

import pytest
import torch
from torch import nn

from peft import OFTConfig, get_peft_model


@pytest.mark.parametrize("kernel_size", [1, 3])
def test_oft_conv2d_unmerged_matches_merged(kernel_size):
    """Unmerged OFT Conv2d forward must match merge (regression for #3982).

    The old unfold→rotate→fold path only agrees with weight-side merge for 1x1 kernels;
    overlapping patches for k>1 made fold sum and diverged.
    """
    torch.manual_seed(0)
    base = nn.Sequential(nn.Conv2d(4, 8, kernel_size, padding=kernel_size // 2))
    model = get_peft_model(
        base,
        OFTConfig(target_modules=["0"], r=0, oft_block_size=4, init_weights=False),
    )
    x = torch.randn(1, 4, 8, 8)
    with torch.no_grad():
        y_unmerged = model(x)
        y_merged = copy.deepcopy(model).merge_and_unload()(x)
    max_diff = (y_unmerged - y_merged).abs().max().item()
    assert max_diff < 1e-5, f"kernel {kernel_size}x{kernel_size}: max |unmerged-merged|={max_diff}"


def test_oft_conv2d_unmerged_is_trainable():
    """Weight-side unmerged forward must still backprop into OFT parameters."""
    torch.manual_seed(0)
    base = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1))
    model = get_peft_model(
        base,
        OFTConfig(target_modules=["0"], r=0, oft_block_size=4, init_weights=False),
    )
    x = torch.randn(2, 4, 8, 8)
    loss = model(x).sum()
    loss.backward()
    oft_layer = model.base_model.model[0]
    grads = [p.grad for p in oft_layer.oft_R["default"].parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
