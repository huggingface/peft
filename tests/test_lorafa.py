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

from __future__ import annotations

import math

import torch
from torch import nn

from peft import LoraConfig, get_peft_model
from peft.optimizers import create_lorafa_optimizer

from .testing_utils import torch_device


class SimpleNet(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.embedding = nn.Embedding(100, 20)
        self.layer_norm = nn.LayerNorm(20)
        self.lin0 = nn.Linear(20, 20, bias=bias)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(20, 16, bias=bias)

    def forward(self, X):
        X = self.lin0(self.layer_norm(self.embedding(X)))
        X = self.relu(X)
        X = self.lin1(X)
        return X


def test_lorafa_init_default():
    """
    Test if the optimizer is correctly created
    """
    lora_rank = 16
    lora_alpha = 32
    lr = 7e-5

    model = SimpleNet()
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["lin0", "lin1"],
        bias="none",
    )
    model = get_peft_model(model, config)
    optimizer = create_lorafa_optimizer(model=model, r=lora_rank, lora_alpha=lora_alpha, lr=lr)

    assert math.isclose(optimizer.param_groups[0]["scaling_factor"], lora_alpha / lora_rank, rel_tol=1e-9, abs_tol=0.0)

    all_A_fixed = True
    all_B_trainable = True

    assert optimizer is not None

    for name, param in model.named_parameters():
        if "lora_A" in name:
            all_A_fixed &= not param.requires_grad
        elif "lora_B" in name:
            all_B_trainable &= param.requires_grad

    assert all_A_fixed and all_B_trainable


def test_lorafa_init_rslora():
    """
    Test if the optimizer is correctly created when use_rslora = True
    """
    lora_rank = 16
    lora_alpha = 32
    lr = 7e-5

    model = SimpleNet()
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["lin0", "lin1"],
        bias="none",
    )
    model = get_peft_model(model, config)
    optimizer = create_lorafa_optimizer(
        model=model, r=lora_rank, lora_alpha=lora_alpha, lr=lr, use_rslora=True
    )
    assert math.isclose(
        optimizer.param_groups[0]["scaling_factor"], lora_alpha / math.sqrt(lora_rank), rel_tol=1e-9, abs_tol=0.0
    )


def test_LoraFAOptimizer_step():
    """
    Test if the optimizer's step function runs without any exception
    """
    lora_rank = 16
    lora_alpha = 32
    lr = 7e-5

    model = SimpleNet()
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["lin0", "lin1"],
        bias="none",
    )
    model = get_peft_model(model, config).to(torch_device)
    optimizer = create_lorafa_optimizer(model=model, r=16, lora_alpha=32, lr=7e-5)
    loss = torch.nn.CrossEntropyLoss()
    x = torch.randint(100, (2, 4, 10)).to(torch_device)
    output = model(x).permute(0, 3, 1, 2)
    label = torch.randint(16, (2, 4, 10)).to(torch_device)
    loss_value = loss(output, label)
    loss_value.backward()
    optimizer.step()
