# Copyright 2023-present the HuggingFace Inc. team.
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

"""
Regression tests for AdaLora SVDConv2d fan_in_fan_out / transpose fix.

Verifies that SVDConv2d.get_delta_weight() correctly applies transpose() using
the fan_in_fan_out attribute, consistent with SVDLinear.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

peft_src_path = str(Path(__file__).parent.parent / "src")
if peft_src_path not in sys.path:
    sys.path.insert(0, peft_src_path)

from peft.tuners.adalora.config import AdaLoraConfig
from peft.tuners.adalora.layer import SVDConv2d, SVDLinear


def _make_config(fan_in_fan_out: bool) -> AdaLoraConfig:
    return AdaLoraConfig(
        init_r=4,
        target_r=8,
        lora_alpha=8,
        tinit=0,
        tfinal=0,
        total_step=1,
        deltaT=1,
        fan_in_fan_out=fan_in_fan_out,
        init_lora_weights=False,
    )


def _build_svd_conv2d(fan_in_fan_out: bool, seed: int = 0) -> SVDConv2d:
    torch.manual_seed(seed)
    base = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3))
    config = _make_config(fan_in_fan_out)
    layer = SVDConv2d(base, adapter_name="default", config=config, r=4, lora_alpha=8)
    return layer


def _copy_params(src: nn.Module, dst: nn.Module) -> None:
    for (n1, p1), (n2, p2) in zip(src.named_parameters(), dst.named_parameters()):
        assert n1 == n2
        p2.data.copy_(p1.data)


class TestAdaLoraConv2dTranspose:
    """Regression tests for the SVDConv2d fan_in_fan_out transpose fix."""

    def test_svdconv2d_get_delta_weight_with_fan_in_fan_out_true(self):
        """When fan_in_fan_out=True, get_delta_weight must apply transpose and return
        a tensor with the correct Conv2d weight shape."""
        layer = _build_svd_conv2d(fan_in_fan_out=True)
        assert layer.fan_in_fan_out is True

        delta = layer.get_delta_weight("default")
        base_weight = layer.get_base_layer().weight

        assert delta.shape == base_weight.shape, (
            f"delta shape {delta.shape} must match base weight shape {base_weight.shape}"
        )
        assert torch.isfinite(delta).all(), "delta weight must be finite"

    def test_svdconv2d_get_delta_weight_with_fan_in_fan_out_false(self):
        """When fan_in_fan_out=False (default), get_delta_weight must still return a
        correctly shaped tensor — no regression from the fix."""
        layer = _build_svd_conv2d(fan_in_fan_out=False)
        assert layer.fan_in_fan_out is False

        delta = layer.get_delta_weight("default")
        base_weight = layer.get_base_layer().weight

        assert delta.shape == base_weight.shape, (
            f"delta shape {delta.shape} must match base weight shape {base_weight.shape}"
        )
        assert torch.isfinite(delta).all(), "delta weight must be finite"

    def test_svdconv2d_fan_in_fan_out_affects_output(self):
        """Changing fan_in_fan_out must change the output of get_delta_weight, proving
        the transpose call is effective."""
        layer_true = _build_svd_conv2d(fan_in_fan_out=True, seed=42)
        layer_false = _build_svd_conv2d(fan_in_fan_out=False, seed=42)
        _copy_params(layer_true, layer_false)

        delta_true = layer_true.get_delta_weight("default")
        delta_false = layer_false.get_delta_weight("default")

        assert not torch.allclose(delta_true, delta_false), (
            "fan_in_fan_out=True and False must produce different delta weights"
        )

    def test_svdconv2d_merge_unmerge_reversibility(self):
        """After merge followed by unmerge, the base weight must be restored within
        numerical precision."""
        layer = _build_svd_conv2d(fan_in_fan_out=True)
        base_layer = layer.get_base_layer()
        orig_weight = base_layer.weight.data.clone()

        layer.merge()
        assert layer.merged, "layer should be merged"
        merged_weight = base_layer.weight.data.clone()
        # Weight should have changed after merge
        assert not torch.allclose(orig_weight, merged_weight), (
            "merge should modify the base weight"
        )

        layer.unmerge()
        assert not layer.merged, "layer should be unmerged"
        restored_weight = base_layer.weight.data

        assert torch.allclose(orig_weight, restored_weight, atol=1e-5, rtol=1e-5), (
            "unmerge must restore the original base weight"
        )

    def test_svdconv2d_consistency_with_svdlinear(self):
        """Both SVDConv2d and SVDLinear must respect fan_in_fan_out in get_delta_weight.
        Verify via source inspection that both call transpose with fan_in_fan_out."""
        import inspect

        for cls in (SVDConv2d, SVDLinear):
            source = inspect.getsource(cls.get_delta_weight)
            assert "transpose" in source, f"{cls.__name__}.get_delta_weight must call transpose()"
            assert "fan_in_fan_out" in source, (
                f"{cls.__name__}.get_delta_weight must reference fan_in_fan_out"
            )

    def test_svdconv2d_safe_merge_unmerge_reversibility(self):
        """safe_merge=True followed by unmerge must also restore weights."""
        layer = _build_svd_conv2d(fan_in_fan_out=True)
        base_layer = layer.get_base_layer()
        orig_weight = base_layer.weight.data.clone()

        layer.merge(safe_merge=True)
        layer.unmerge()

        assert torch.allclose(orig_weight, base_layer.weight.data, atol=1e-5, rtol=1e-5), (
            "safe_merge + unmerge must restore original weights"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
