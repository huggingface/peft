# Copyright 2025-present the HuggingFace Inc. team.
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
import torch.nn as nn

from peft.tuners.hira import Linear


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
        init_weights=True,
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
