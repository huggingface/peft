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

# This test file is for tests specific to PEANuT, since PEANuT's delta is conditioned on the base weight.

import torch
from torch import nn

from peft import PeanutConfig, get_peft_model


class MLP(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.lin1 = nn.Linear(20, 20, bias=bias)  # lin1 and lin2 have same shape
        self.lin2 = nn.Linear(20, 20, bias=bias)
        self.lin3 = nn.Linear(20, 2, bias=bias)
        self.sm = nn.LogSoftmax(dim=-1)

    def forward(self, X):
        X = self.lin0(X)
        X = self.relu(X)
        X = self.lin1(X)
        X = self.relu(X)
        X = self.lin2(X)
        X = self.relu(X)
        X = self.lin3(X)
        X = self.sm(X)
        return X


class TestPeanut:
    def _build_two_adapter_model(self):
        # PEANuT's delta is `Tweaker(W)`, so it is explicitly conditioned on the base weight. Building this the
        # exact same way every time (fresh model, same seed, same call order) means adapters "A" and "B" always
        # end up with identical random parameters across independently-built models, so the different
        # merge-call orderings below are directly comparable.
        torch.manual_seed(0)
        model = MLP()
        config_a = PeanutConfig(target_modules=["lin1"], r=4, depth=1, act_fn="gelu", init_weights=False)
        peft_model = get_peft_model(model, config_a, adapter_name="A")
        config_b = PeanutConfig(target_modules=["lin1"], r=4, depth=1, act_fn="gelu", init_weights=False)
        peft_model.add_adapter("B", config_b)
        return peft_model

    def test_sequential_merge_calls_match_combined_merge_regardless_of_order(self):
        # Regression test: separate, sequential merge() calls used to silently diverge from a single combined
        # merge() call, because each new call recomputed its delta against the *current* (possibly
        # already-merged) base weight instead of the layer's original, pristine one.
        combined_model = self._build_two_adapter_model()
        combined_layer = combined_model.base_model.model.lin1
        pristine_weight = combined_layer.base_layer.weight.data.clone()
        combined_layer.merge(adapter_names=["A", "B"])
        ground_truth = combined_layer.base_layer.weight.data.clone()

        # sanity check: merging must actually change the weight, else the comparisons below are vacuous
        assert not torch.allclose(ground_truth, pristine_weight, atol=1e-6)

        model_ab = self._build_two_adapter_model()
        layer_ab = model_ab.base_model.model.lin1
        layer_ab.merge(adapter_names=["A"])
        layer_ab.merge(adapter_names=["B"])
        weight_ab = layer_ab.base_layer.weight.data.clone()

        model_ba = self._build_two_adapter_model()
        layer_ba = model_ba.base_model.model.lin1
        layer_ba.merge(adapter_names=["B"])
        layer_ba.merge(adapter_names=["A"])
        weight_ba = layer_ba.base_layer.weight.data.clone()

        assert torch.allclose(weight_ab, ground_truth, atol=1e-4, rtol=1e-4)
        assert torch.allclose(weight_ba, ground_truth, atol=1e-4, rtol=1e-4)
        # the clearest proof this was a real bug and not just numerical noise: merging in the other order used
        # to produce a *different* wrong answer
        assert torch.allclose(weight_ab, weight_ba, atol=1e-4, rtol=1e-4)

    def test_unmerge_after_sequential_merge_calls_restores_pristine_weight(self):
        # The delta *values* cached at merge time and subtracted by unmerge() were never wrong -- only the
        # computation of a *new* delta during a later, separate merge() call was. So after fully unmerging,
        # the base weight must come back exactly to the original, regardless of how many separate merge()
        # calls were used to get there.
        model = self._build_two_adapter_model()
        layer = model.base_model.model.lin1
        pristine_weight = layer.base_layer.weight.data.clone()

        layer.merge(adapter_names=["A"])
        layer.merge(adapter_names=["B"])
        assert not torch.allclose(layer.base_layer.weight.data, pristine_weight, atol=1e-6)

        layer.unmerge()
        assert torch.allclose(layer.base_layer.weight.data, pristine_weight, atol=1e-5, rtol=1e-5)
