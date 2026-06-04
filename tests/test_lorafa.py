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


def _run_lorafa_weight_decay_step(config: LoraConfig, lr: float, weight_decay: float):
    seed = 42

    torch.manual_seed(seed)
    model_no_wd = get_peft_model(SimpleNet(), config).to(torch_device)
    torch.manual_seed(seed)
    model_wd = get_peft_model(SimpleNet(), config).to(torch_device)

    # Shared setup invariant: both models must start from the same parameters.
    for (name_no_wd, param_no_wd), (name_wd, param_wd) in zip(
        model_no_wd.named_parameters(), model_wd.named_parameters()
    ):
        assert name_no_wd == name_wd
        assert torch.equal(param_no_wd, param_wd)

    optimizer_no_wd = create_lorafa_optimizer(
        model=model_no_wd,
        r=config.r,
        lora_alpha=config.lora_alpha,
        lr=lr,
        weight_decay=0.0,
    )
    optimizer_wd = create_lorafa_optimizer(
        model=model_wd,
        r=config.r,
        lora_alpha=config.lora_alpha,
        lr=lr,
        weight_decay=weight_decay,
    )
    loss = torch.nn.CrossEntropyLoss()

    # Save initial lora_A weights. Only from one model since both models params are identical
    initial_lora_A_weights = {
        name: param.clone() for name, param in model_no_wd.named_parameters() if "lora_A" in name
    }

    # Generate random input and label using different seeds
    torch.manual_seed(seed + 1)
    x = torch.randint(100, (2, 4, 10)).to(torch_device)
    output_no_wd = model_no_wd(x).permute(0, 3, 1, 2)
    output_wd = model_wd(x).permute(0, 3, 1, 2)
    torch.manual_seed(seed + 2)
    label = torch.randint(16, (2, 4, 10)).to(torch_device)

    # Calculate both losses and perform backward passes
    loss_value_no_wd = loss(output_no_wd, label)
    loss_value_no_wd.backward()
    loss_value_wd = loss(output_wd, label)
    loss_value_wd.backward()

    non_lora_trainable_names = [
        name
        for name, param in model_no_wd.named_parameters()
        if "lora" not in name and param.requires_grad and param.grad is not None
    ]

    # Perform both optimizer steps
    optimizer_no_wd.step()
    optimizer_wd.step()

    return (
        dict(model_no_wd.named_parameters()),
        dict(model_wd.named_parameters()),
        initial_lora_A_weights,
        non_lora_trainable_names,
    )


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
    optimizer = create_lorafa_optimizer(model=model, r=lora_rank, lora_alpha=lora_alpha, lr=lr, use_rslora=True)
    assert math.isclose(
        optimizer.param_groups[0]["scaling_factor"], lora_alpha / math.sqrt(lora_rank), rel_tol=1e-9, abs_tol=0.0
    )


def test_LoraFAOptimizer_step():
    """
    Test if the optimizer's step function runs without any exception and checks specific conditions on lora_A and
    lora_B weights.
    """
    lora_rank = 16
    lora_alpha = 32
    lr = 7e-5
    num_steps = 5

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

    # Save initial weights of lora_A
    initial_lora_A_weights = {name: param.clone() for name, param in model.named_parameters() if "lora_A" in name}
    # Ensure lora_B is initialized to zero
    for name, param in model.named_parameters():
        if "lora_B" in name:
            assert torch.all(param == 0), f"lora_B weights not initialized to zero for {name}"

    for _ in range(num_steps):  # Run the optimizer step multiple times
        # Generate random input and label for each step
        x = torch.randint(100, (2, 4, 10)).to(torch_device)
        output = model(x).permute(0, 3, 1, 2)
        label = torch.randint(16, (2, 4, 10)).to(torch_device)

        # Calculate loss and perform backward pass
        loss_value = loss(output, label)
        loss_value.backward()

        # Perform optimizer step
        optimizer.step()

        # Zero the gradients after each step to prevent accumulation
        optimizer.zero_grad()

    # Check if lora_A weights have not changed
    for name, param in model.named_parameters():
        if "lora_A" in name:
            assert torch.equal(param, initial_lora_A_weights[name]), f"lora_A weights changed for {name}"

    # Check if lora_B weights are non-zero
    for name, param in model.named_parameters():
        if "lora_B" in name:
            assert torch.any(param != 0), f"lora_B weights are still zero for {name}"


def test_lorafa_weight_decay_decoupled_update_lora_b():
    """
    Test that one optimizer step applies decoupled weight decay to LoRA B weights.
    """
    lora_rank = 16
    lora_alpha = 32
    # Stronger lr and weight_decay to make the decay effect more pronounced for testing
    lr = 1e-2
    weight_decay = 1.0
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["lin0", "lin1"],
        bias="none",
    )
    params_no_wd, params_wd, initial_lora_A_weights, _ = _run_lorafa_weight_decay_step(config, lr, weight_decay)

    # Compute the scaling factor for the expected relation of with and without weight decay
    scale = 1.0 - lr * weight_decay

    # Check if lora_A weights have not changed
    for name, param in params_no_wd.items():
        if "lora_A" in name:
            assert torch.equal(param, initial_lora_A_weights[name]), f"lora_A weights changed for {name}"

    # Check if lora_B weights are non-zero and if they follow the expected relation
    for name, param_no_wd in params_no_wd.items():
        if "lora_B" in name:
            assert torch.any(param_no_wd != 0), f"lora_B weights are still zero for {name}"
            assert torch.allclose(params_wd[name], param_no_wd * scale, rtol=1e-5, atol=1e-6), (
                f"lora_B weights for {name} do not match decoupled weight decay scaling"
            )


def test_lorafa_weight_decay_decoupled_update_non_lora_params():
    """
    Test that one optimizer step applies decoupled weight decay to non-LoRA trainable parameters.
    """
    lora_rank = 16
    lora_alpha = 32
    # Stronger lr and weight_decay to make the decay effect more pronounced for testing
    lr = 1e-2
    weight_decay = 1.0
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["lin0", "lin1"],
        bias="all",  # Include bias to check non-LoRA trainable parameters
    )
    params_no_wd, params_wd, _, non_lora_trainable_names = _run_lorafa_weight_decay_step(config, lr, weight_decay)

    # Compute the scaling factor for the expected relation of with and without weight decay
    scale = 1.0 - lr * weight_decay

    # Sanity check: non-LoRA trainable parameters
    assert non_lora_trainable_names, "Expected at least one non-LoRA trainable parameter with gradients"

    # Check if all non-LoRA params also follow the expected relation
    for name in non_lora_trainable_names:
        assert torch.allclose(
            params_wd[name],
            params_no_wd[name] * scale,
            rtol=1e-5,
            atol=1e-6,
        ), f"{name} does not match decoupled weight decay scaling"
