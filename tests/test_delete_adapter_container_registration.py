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

# Regression tests for `delete_adapter` fully removing per-adapter state.
#
# Several tuners kept per-adapter state in containers (dicts / BufferDicts) that were not listed in the layer's
# `adapter_layer_names` / `other_param_names`. `BaseTunerLayer.delete_adapter` only iterates those two tuples, so the
# state of a deleted adapter was left behind. Each parametrization below deletes one of two adapters and asserts that
# none of the previously-unregistered containers still holds the deleted adapter. These fail before the containers are
# registered and pass afterwards.

import pytest
import torch
from torch import nn

from peft import FourierFTConfig, PsoftConfig, UniLoraConfig, get_peft_model


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = nn.Linear(16, 16)
        self.lin1 = nn.Linear(16, 16)

    def forward(self, x):
        return self.lin1(self.lin0(x))


# tuner name -> (config factory, per-adapter containers that used to leak on delete)
CASES = {
    "unilora": (
        lambda: UniLoraConfig(target_modules=["lin0", "lin1"], r=4, theta_d_length=100),
        ["r", "unilora_indices_A", "unilora_indices_B"],
    ),
    "fourierft": (
        lambda: FourierFTConfig(target_modules=["lin0", "lin1"], n_frequency=50),
        ["indices"],
    ),
    "psoft": (
        lambda: PsoftConfig(target_modules=["lin0", "lin1"], r=4),
        ["random_seed", "_psoft_A_cache", "_psoft_B_cache"],
    ),
}


@pytest.mark.parametrize("tuner", list(CASES))
def test_delete_adapter_removes_previously_unregistered_containers(tuner):
    make_config, containers = CASES[tuner]
    model = get_peft_model(MLP(), make_config(), adapter_name="a1")
    model.add_adapter("a2", make_config())

    # Exercise the second adapter so any lazily-populated per-adapter buffers exist before deletion.
    model.base_model.set_adapter("a2")
    model(torch.randn(2, 16))
    model.base_model.set_adapter("a1")

    model.delete_adapter("a2")

    for module in model.modules():
        for name in containers:
            container = getattr(module, name, None)
            if container is not None and hasattr(container, "keys"):
                assert "a2" not in container, f"deleted adapter still present in `{name}`"
                assert "a1" in container, f"remaining adapter missing from `{name}`"
