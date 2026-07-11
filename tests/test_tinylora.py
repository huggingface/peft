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

from peft import TinyLoraConfig, get_peft_model


class MLP(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.lin0 = nn.Linear(10, 10, bias=bias)
        self.lin1 = nn.Linear(10, 10, bias=bias)
        self.lin2 = nn.Linear(10, 10, bias=bias)
        self.lin3 = nn.Linear(10, 10, bias=bias)
        self.relu = nn.ReLU()
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


class TestTinyLoraMultipleAdapters:
    """Regression tests for the model-level `_target_key_to_idx` cache used to compute `weight_tying` groups."""

    def get_mlp(self):
        torch.manual_seed(0)
        return MLP()

    def test_second_adapter_overlapping_target_modules_matches_fresh_control(self):
        """Adding a 2nd adapter whose target_modules overlap with (but differ from) a prior adapter's must not reuse
        the first adapter's stale layer-index mapping to compute the `weight_tying` groups.

        This is a regression test for a bug in `TinyLoraModel._create_and_replace` where the model-level mapping
        from module key to layer index (used to derive the number of `weight_tying` groups) was only rebuilt when
        the current module's key was missing from the existing mapping, instead of whenever a new adapter's
        injection cycle starts. Because `PeftModel.add_adapter` calls `inject_adapter` directly (bypassing
        `_pre_injection_hook`), a second adapter whose target_modules overlapped with the first adapter's kept
        reusing the first adapter's mapping (wrong layer index and module count) for any overlapping module, which
        silently corrupted the `weight_tying` group assignment with no error or warning.
        """
        # First adapter targets all 4 linear layers. This seeds the model-level layer-index cache that must not
        # leak into the second adapter's group computation below.
        mlp = self.get_mlp()
        config_default = TinyLoraConfig(target_modules=["lin0", "lin1", "lin2", "lin3"], r=2, u=4, weight_tying=0.0)
        model = get_peft_model(mlp, config_default)

        # Second adapter targets a different (strict subset) but overlapping set of modules, with partial
        # weight_tying, so lin1 and lin2 are expected to share a single trainable v (1 group, not 2).
        config_b = TinyLoraConfig(target_modules=["lin1", "lin2"], r=2, u=4, weight_tying=1 / 3)
        model.add_adapter("b", config_b)

        # Control: a freshly built model with ONLY the second adapter's config, so nothing precedes it.
        mlp_control = self.get_mlp()
        model_control = get_peft_model(mlp_control, config_b, adapter_name="b")

        b_groups = model.tinylora_v["b"]
        control_groups = model_control.tinylora_v["b"]

        # Both should resolve to exactly 1 group (full tying between the 2 targeted modules) with u=4 parameters.
        assert len(control_groups) == 1
        assert sum(p.numel() for p in control_groups.values()) == 4
        assert len(b_groups) == len(control_groups)
        assert sum(p.numel() for p in b_groups.values()) == sum(p.numel() for p in control_groups.values())

        # lin1 and lin2 must share the exact same trainable vector (full tying), matching the control model.
        lin1_v = model.base_model.model.lin1._tinylora_v_ref["b"]
        lin2_v = model.base_model.model.lin2._tinylora_v_ref["b"]
        assert lin1_v.data_ptr() == lin2_v.data_ptr()

        control_lin1_v = model_control.base_model.model.lin1._tinylora_v_ref["b"]
        control_lin2_v = model_control.base_model.model.lin2._tinylora_v_ref["b"]
        assert control_lin1_v.data_ptr() == control_lin2_v.data_ptr()

    def test_second_adapter_overlapping_target_modules_after_delete_and_readd(self):
        """Deleting an adapter and re-adding one with the same name but different target_modules must also rebuild
        the layer-index mapping instead of reusing the stale one left behind by the deleted adapter.
        """
        mlp = self.get_mlp()
        config_default = TinyLoraConfig(target_modules=["lin0", "lin1", "lin2", "lin3"], r=2, u=4, weight_tying=0.0)
        model = get_peft_model(mlp, config_default)

        config_b1 = TinyLoraConfig(target_modules=["lin0", "lin1", "lin2", "lin3"], r=2, u=4, weight_tying=0.0)
        model.add_adapter("b", config_b1)
        model.delete_adapter("b")

        config_b2 = TinyLoraConfig(target_modules=["lin1", "lin2"], r=2, u=4, weight_tying=1 / 3)
        model.add_adapter("b", config_b2)

        mlp_control = self.get_mlp()
        model_control = get_peft_model(mlp_control, config_b2, adapter_name="b")

        assert len(model.tinylora_v["b"]) == len(model_control.tinylora_v["b"]) == 1
