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
import warnings

import pytest
import torch
from torch import nn

import peft.optimizers.lorafa as lorafa_module
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


class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 8)
        self.lin = nn.Linear(8, 4)

    def forward(self, X):
        return self.lin(self.embedding(X))


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


@pytest.mark.parametrize("use_rslora", [False, True])
def test_lorafa_init(use_rslora):
    """
    Test if the optimizer is correctly created for both standard LoRA and rsLoRA configs.
    """
    lora_rank = 16
    lora_alpha = 32
    lr = 7e-5

    model = SimpleNet()
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["lin0", "lin1"],
        use_rslora=use_rslora,
        bias="none",
    )
    model = get_peft_model(model, config)
    optimizer = create_lorafa_optimizer(model=model, r=lora_rank, lora_alpha=lora_alpha, lr=lr)

    expected_scaling = lora_alpha / math.sqrt(lora_rank) if use_rslora else lora_alpha / lora_rank
    scaling_factors = [factor for factor in optimizer.param_groups[0]["scaling_factors"] if factor is not None]
    assert scaling_factors
    assert all(math.isclose(factor, expected_scaling, rel_tol=1e-9, abs_tol=0.0) for factor in scaling_factors)

    all_A_fixed = True
    all_B_trainable = True

    assert optimizer is not None

    for name, param in model.named_parameters():
        if "lora_A" in name:
            all_A_fixed &= not param.requires_grad
        elif "lora_B" in name:
            all_B_trainable &= param.requires_grad

    assert all_A_fixed and all_B_trainable


@pytest.mark.parametrize("use_rslora", [False, True])
def test_lorafa_rslora_flag_mismatch_raises(use_rslora):
    """
    Test if passing an explicit use_rslora flag that disagrees with the active adapter config raises.
    """
    lora_rank = 16
    lora_alpha = 32
    lr = 7e-5

    model = SimpleNet()
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["lin0", "lin1"],
        use_rslora=not use_rslora,
        bias="none",
    )
    model = get_peft_model(model, config)

    with pytest.raises(ValueError, match="was passed to create_lorafa_optimizer"):
        create_lorafa_optimizer(model=model, r=lora_rank, lora_alpha=lora_alpha, lr=lr, use_rslora=use_rslora)


def test_lorafa_init_embedding_target_module():
    """
    Test if embedding-targeted LoRA adapters resolve scaling_factors and the optimizer step works
    """
    lora_rank = 16
    lora_alpha = 32
    lr = 7e-5

    model = EmbeddingNet()
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["embedding"],
        bias="none",
    )
    model = get_peft_model(model, config).to(torch_device)
    optimizer = create_lorafa_optimizer(model=model, r=lora_rank, lora_alpha=lora_alpha, lr=lr)

    lora_scaling_factors = [
        scaling_factor
        for name, scaling_factor in zip(
            optimizer.param_groups[0]["names"], optimizer.param_groups[0]["scaling_factors"]
        )
        if "lora" in name
    ]
    assert lora_scaling_factors
    assert all(scaling_factor is not None for scaling_factor in lora_scaling_factors)
    assert all(
        math.isclose(scaling_factor, lora_alpha / lora_rank, rel_tol=1e-9, abs_tol=0.0)
        for scaling_factor in lora_scaling_factors
    )

    # Run a single optimizer step to ensure it works without crashing
    x = torch.randint(100, (2, 3)).to(torch_device)
    output = model(x)
    output.sum().backward()

    optimizer.step()


# TODO remove after 2026-11-01
def test_lorafa_scaling_factor_deprecation_warning():
    """
    Test that using the legacy scaling_factor emits a FutureWarning
    """
    param = nn.Parameter(torch.ones(2, 2))
    optimizer = lorafa_module.LoraFAOptimizer(
        [
            {
                "params": [param],
                "lr": 1e-3,
                "names": ["dummy.weight"],
                "scaling_factor": 1.0,
                "betas": (0.9, 0.999),
                "eps": 1e-6,
                "weight_decay": 0.0,
                "correct_bias": True,
            }
        ]
    )
    param.grad = torch.ones_like(param)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        optimizer.step()

    assert any(
        isinstance(warning.message, FutureWarning) and "`scaling_factor` is deprecated" in str(warning.message)
        for warning in caught_warnings
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


def test_lorafa_respects_layer_specific_scaling_patterns(monkeypatch):
    """
    Test if the optimizer uses each layer's own scaling when rank_pattern changes the effective rank
    """

    monkeypatch.setattr(lorafa_module, "is_bf16_available", lambda: False)

    seed = 123
    torch.manual_seed(seed)

    lora_rank = 16
    lora_alpha = 32
    lr = 7e-5
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["lin0", "lin1"],
        rank_pattern={
            "lin0": 8
        },  # lin0 will have effective rank 8, lin1 will have effective rank 16, so their scaling will differ
        bias="none",
    )
    model = get_peft_model(SimpleNet(), config).to(torch_device)
    optimizer = create_lorafa_optimizer(model=model, r=lora_rank, lora_alpha=lora_alpha, lr=lr)

    lin0 = model.base_model.model.lin0
    lin1 = model.base_model.model.lin1

    # Manually set gradients for lin0 and lin1's lora_B weights to a known value (e.g., all 0.5) to test the expected exp_avg_B update
    lin0_grad = torch.full_like(lin0.lora_B.default.weight, 0.5)
    lin1_grad = torch.full_like(lin1.lora_B.default.weight, 0.5)
    lin0.lora_B.default.weight.grad = lin0_grad
    lin1.lora_B.default.weight.grad = lin1_grad

    optimizer.step()

    # func to compute the expected exp_avg_B after one step, given the layer and its gradient
    # Replicates math from lorarfa_module.LoraFAOptimizer.step()
    def expected_exp_avg_B(layer, grad):
        # scaling factor for the layer
        scale = layer.scaling["default"]
        # A
        a_weight = layer.lora_A.default.weight

        # projection
        delta = 1e-8

        # computing the inverse matrix
        aa_T = a_weight @ a_weight.T
        aa_T_inv = torch.linalg.pinv(aa_T + delta * torch.eye(a_weight.shape[0]).to(a_weight.device))

        projected_grad_B = (1.0 / scale**2) * (grad @ aa_T_inv)

        # Get beta1 from the only one optimizer's param group created in lorarfa_module.create_lorafa_optimizer()
        beta1, _ = optimizer.param_groups[0]["betas"]

        # Since expected_exp_avg_B starts at zero, the first step's expected value is just the projected_grad scaled by (1 - beta1)
        expect_exp_avg_B = projected_grad_B * (1.0 - beta1)
        return expect_exp_avg_B

    # Retrieve the updated exp_avg_B from the optimizer's state for both lin0 and lin1
    lin0_state = optimizer.state["base_model.model.lin0.lora"]
    lin1_state = optimizer.state["base_model.model.lin1.lora"]

    # Check that the exp_avg_B values in the optimizer's state match the expected values based on the manually set gradients and the layer-specific scaling
    assert torch.allclose(lin1_state["exp_avg_B"], expected_exp_avg_B(lin1, lin1_grad), rtol=1e-5, atol=1e-6), (
        "lin1 exp_avg_B does not match the expected layer-specific scaling"
    )
    assert torch.allclose(lin0_state["exp_avg_B"], expected_exp_avg_B(lin0, lin0_grad), rtol=1e-5, atol=1e-6), (
        "lin0 exp_avg_B does not match the expected layer-specific scaling"
    )
