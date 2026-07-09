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

import torch
from torch import nn

from peft import DeftConfig, get_peft_model

from .testing_utils import require_torch_gpu


class MLP(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.lin1 = nn.Linear(20, 2, bias=bias)
        self.sm = nn.LogSoftmax(dim=-1)
        self.dtype = torch.float

    def forward(self, X):
        X = X.to(self.dtype)
        X = self.lin0(X)
        X = self.relu(X)
        X = self.drop(X)
        X = self.lin1(X)
        X = self.sm(X)
        return X


class TestDeftPaRa:
    """Dedicated tests for the DEFT PaRa mode (`para=True`): pure subspace removal, no injection.

    PaRa is not an identity at init (it removes a sub-space of W), so it does not fit the shared custom-model test
    cases that assume an identity-at-init adapter.
    """

    def test_para_removal_only_and_exact_merge(self):
        torch.manual_seed(0)
        model = MLP()
        model.eval()  # disable dropout so the forward is deterministic
        x = torch.rand(5, 10)
        base_out = model(x).detach().clone()
        w0 = model.lin0.weight.detach().clone()

        config = DeftConfig(target_modules=["lin0"], decomposition_method="qr", para=True)
        peft_model = get_peft_model(model, config)
        peft_model.eval()
        layer = peft_model.base_model.model.lin0

        # PaRa creates no injection matrix R; P is the only trainable matrix.
        assert "default" in layer.deft_P
        assert "default" not in layer.deft_R

        # PaRa is not an identity at init: removing a sub-space of W changes the output.
        para_out = peft_model(x)
        assert not torch.allclose(base_out, para_out, atol=1e-4)

        # merge then forward equals the unmerged forward; unmerge restores the original weight exactly.
        peft_model.merge_adapter(safe_merge=True)
        merged_out = peft_model(x)
        assert torch.allclose(para_out, merged_out, atol=1e-4)
        peft_model.unmerge_adapter()
        assert torch.allclose(layer.base_layer.weight, w0, atol=1e-5)


class TestDeftMerge:
    """DEFT caches only the small base-weight-dependent factor (`right.T @ W`) at merge, not the full delta."""

    def test_merge_caches_small_factor_and_unmerges_exactly(self):
        # Caching the full out x in delta per merged adapter is expensive with many adapters; DEFT instead caches only
        # right.T @ W (r x in_features) and recomputes the exact delta at unmerge (see review feedback).
        torch.manual_seed(0)
        model = MLP()
        model.eval()  # disable dropout for a deterministic forward
        x = torch.rand(5, 10)

        config = DeftConfig(target_modules=["lin0"], decomposition_method="relu", r=4)
        peft_model = get_peft_model(model, config)
        peft_model.eval()
        layer = peft_model.base_model.model.lin0

        # make the injection non-trivial so the merge delta is non-zero (identity-init alone gives delta == 0)
        with torch.no_grad():
            layer.deft_R["default"].normal_(std=0.1)

        r = layer.deft_r["default"]
        out_features, in_features = layer.base_layer.weight.shape
        unmerged_out = peft_model(x).detach().clone()
        w0 = layer.base_layer.weight.detach().clone()

        peft_model.merge_adapter(safe_merge=True)
        # the cache is the small r x in_features factor, strictly smaller than the full out x in delta
        factor = layer._cached_merge_factor["default"]
        assert factor.shape == (r, in_features)
        assert factor.numel() < out_features * in_features

        # merge is correct, and unmerge restores the original weight exactly from the cached factor
        assert torch.allclose(peft_model(x), unmerged_out, atol=1e-4)
        peft_model.unmerge_adapter()
        assert torch.allclose(layer.base_layer.weight, w0, atol=1e-5)

    def test_merge_caches_factor_in_float32_for_bf16_base_weight(self):
        # Regression test: the cached factor must stay in float32 regardless of the base layer's dtype.
        # `_merge_factor` always returns float32 (see its docstring); previously it was downcast to the
        # base layer's dtype before caching (e.g. bf16), so `unmerge` recomputed the delta from a
        # *rounded* factor instead of the exact one `merge` used -- the two deltas then differed, breaking
        # the "unmerge recomputes the exact delta" guarantee this cache exists for. That bug is invisible
        # with a float32 base layer (downcasting float32 -> float32 loses nothing), which is why the fp32
        # test above (`test_merge_caches_small_factor_and_unmerges_exactly`) never caught it.
        torch.manual_seed(0)
        model = MLP()
        model.dtype = torch.bfloat16
        model = model.to(torch.bfloat16)
        model.eval()
        x = torch.rand(5, 10)

        config = DeftConfig(target_modules=["lin0"], decomposition_method="relu", r=4)
        peft_model = get_peft_model(model, config)
        peft_model.eval()
        layer = peft_model.base_model.model.lin0

        with torch.no_grad():
            layer.deft_R["default"].normal_(std=0.1)

        # Capture the delta merge() is about to apply, from the pristine (not-yet-cached) factor.
        factor_before_merge = layer._merge_factor("default")
        delta_at_merge = layer._delta_from_factor("default", factor_before_merge)

        peft_model.merge_adapter()

        cached_factor = layer._cached_merge_factor["default"]
        assert cached_factor.dtype == torch.float32, (
            f"cached merge factor must stay float32, got {cached_factor.dtype} "
            f"(base layer weight dtype is {layer.base_layer.weight.dtype})"
        )
        assert torch.equal(cached_factor, factor_before_merge)

        # This mirrors exactly what unmerge() does with the cached factor.
        delta_at_unmerge = layer._delta_from_factor("default", cached_factor.to(torch.float32))
        assert torch.equal(delta_at_merge, delta_at_unmerge), (
            "delta recomputed at unmerge time must be bit-identical to the delta merge() applied"
        )

    @require_torch_gpu
    def test_cached_merge_factor_follows_model_to_device(self):
        # Regression test: `_cached_merge_factor` was a plain dict, so `model.to(device)` never moved its
        # tensors (unlike deft_P/deft_R, which are ParameterDicts and are moved automatically). Merging on
        # one device, moving the model, then unmerging used to raise a device-mismatch RuntimeError.
        torch.manual_seed(0)
        model = MLP()
        model.eval()
        config = DeftConfig(target_modules=["lin0"], decomposition_method="relu", r=4)
        peft_model = get_peft_model(model, config)
        layer = peft_model.base_model.model.lin0

        peft_model.merge_adapter()
        assert layer._cached_merge_factor["default"].device.type == "cpu"

        peft_model = peft_model.to("cuda")
        assert layer._cached_merge_factor["default"].device.type == "cuda"

        peft_model.unmerge_adapter()  # must not raise
