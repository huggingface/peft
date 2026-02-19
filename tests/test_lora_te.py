# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
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

import copy

import pytest
import torch
from torch import nn

from peft import LoraConfig, get_peft_model
from peft.import_utils import is_te_available

from .testing_utils import require_torch_gpu


pytestmark = pytest.mark.skipif(not is_te_available(), reason="transformer_engine is not available")


if is_te_available():
    import transformer_engine.pytorch as te

    from peft.tuners.lora.te import TELoRA


class SmallTEModel(nn.Module):
    def __init__(self, hidden_size: int = 10, ffn_hidden_size: int = 16, bias: bool = True):
        super().__init__()
        self.linear = te.Linear(hidden_size, hidden_size, bias=bias)
        self.ln_linear = te.LayerNormLinear(hidden_size, hidden_size)
        self.ln_mlp = te.LayerNormMLP(hidden_size, ffn_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.ln_linear(x)
        x = self.ln_mlp(x)
        return x


@require_torch_gpu
def test_te_lora_wraps_te_linear_and_keeps_forward_working():
    model = SmallTEModel()
    cfg = LoraConfig(target_modules=["linear", "ln_linear", "ln_mlp"], r=2, lora_alpha=8)

    lora_model = get_peft_model(model, cfg)
    wrapped_linear = lora_model.base_model.model.linear
    wrapped_ln_linear = lora_model.base_model.model.ln_linear
    wrapped_ln_mlp = lora_model.base_model.model.ln_mlp

    assert isinstance(wrapped_linear, TELoRA)
    assert isinstance(wrapped_ln_linear, TELoRA)
    assert isinstance(wrapped_ln_mlp, TELoRA)
    assert "default" in wrapped_linear.lora_A and "default" in wrapped_linear.lora_B
    assert wrapped_linear.get_base_layer().weight.requires_grad is False
    assert wrapped_linear.lora_A["default"].weight.requires_grad is True
    assert wrapped_linear.lora_B["default"].weight.requires_grad is True

    x = torch.randn(2, 10, dtype=torch.float32, device=wrapped_linear.get_base_layer().weight.device)
    out = lora_model(x)
    assert out.shape == (2, 10)


@require_torch_gpu
def test_te_lora_forward_matches_base_before_backward():
    model = SmallTEModel()
    dummy_model = copy.deepcopy(model)
    cfg = LoraConfig(target_modules=["linear", "ln_linear", "ln_mlp"], r=4, lora_alpha=8)
    lora_model = get_peft_model(model, cfg)

    x = torch.ones(
        2, 10, dtype=torch.float32, device=lora_model.base_model.model.linear.get_base_layer().weight.device
    )
    lora_result = lora_model(x)
    dummy_result = dummy_model(x.to(next(dummy_model.parameters()).device))

    assert torch.allclose(lora_result, dummy_result)


@require_torch_gpu
def test_te_lora_backward():
    model = SmallTEModel()
    cfg = LoraConfig(target_modules=["linear", "ln_linear", "ln_mlp"], r=4, lora_alpha=8)
    lora_model = get_peft_model(model, cfg)

    optimizer = torch.optim.AdamW(lora_model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    device = lora_model.base_model.model.linear.get_base_layer().weight.device
    x = torch.randn(4, 10, requires_grad=True, device=device)
    labels = torch.randint(10, (4,), device=device)

    output = lora_model(x)
    loss = loss_fn(output, labels)
    loss.backward()
    optimizer.step()
