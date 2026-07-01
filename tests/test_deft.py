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
