# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from peft import LoraConfig, PeftType, VeloraConfig, get_peft_model
from peft.tuners.lora import VeloraConfig as LoraVeloraConfig
from peft.tuners.lora.velora import (
    VeloraFunction,
    _compress_activations,
    _normalize_projection,
    _reconstruct_activations,
    _reshape_to_grouped_subtokens,
)
from peft.utils import get_peft_model_state_dict


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = nn.Linear(16, 32)
        self.lin1 = nn.Linear(32, 16)

    def forward(self, x):
        return self.lin1(torch.relu(self.lin0(x)))


class SingleLinear(nn.Module):
    def __init__(self, in_features=128, out_features=64, bias=False):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.lin(x)


def _saved_tensor_bytes(loss_factory) -> int:
    saved_bytes = 0

    def pack_hook(tensor):
        nonlocal saved_bytes
        saved_bytes += tensor.numel() * tensor.element_size()
        return tensor

    def unpack_hook(tensor):
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        loss = loss_factory()
    loss.backward()
    return saved_bytes


def _make_velora_lora_config(
    *,
    target_modules,
    r,
    velora_scale=1.0,
    init_type="batch_average_once",
    num_groups=32,
    lora_alpha=None,
    init_lora_weights=True,
    velora_config_cls=VeloraConfig,
):
    kwargs = {
        "target_modules": target_modules,
        "r": r,
        "velora_config": velora_config_cls(
            scale=velora_scale,
            init_type=init_type,
            num_groups=num_groups,
        ),
    }
    if lora_alpha is not None:
        kwargs["lora_alpha"] = lora_alpha
    if init_lora_weights is not True:
        kwargs["init_lora_weights"] = init_lora_weights
    return LoraConfig(**kwargs)


def test_velora_config_alias_matches_lora_module_config():
    torch.manual_seed(0)
    lora_model = get_peft_model(
        copy.deepcopy(MLP()),
        _make_velora_lora_config(
            target_modules=["lin0"],
            r=4,
            lora_alpha=8,
            velora_scale=1.0,
            init_type="random",
            num_groups=8,
            velora_config_cls=VeloraConfig,
        ),
    )

    torch.manual_seed(0)
    alias_model = get_peft_model(
        copy.deepcopy(MLP()),
        _make_velora_lora_config(
            target_modules=["lin0"],
            r=4,
            lora_alpha=8,
            velora_scale=1.0,
            init_type="random",
            num_groups=8,
            velora_config_cls=LoraVeloraConfig,
        ),
    )

    lora_config = lora_model.peft_config["default"]
    alias_config = alias_model.peft_config["default"]

    assert lora_config.peft_type == PeftType.LORA
    assert alias_config.peft_type == PeftType.LORA
    assert lora_config.velora_config is not None
    assert alias_config.velora_config is not None
    assert lora_config.velora_config.num_groups == alias_config.velora_config.num_groups == 8
    assert lora_config.velora_config.init_type == alias_config.velora_config.init_type == "random"
    assert lora_config.velora_config.scale == alias_config.velora_config.scale == 1.0

    lora_state = get_peft_model_state_dict(lora_model)
    alias_state = get_peft_model_state_dict(alias_model)

    assert lora_state.keys() == alias_state.keys()
    for key in lora_state:
        assert torch.equal(lora_state[key], alias_state[key]), f"Mismatch for {key}"


def test_velora_supports_non_divisible_groups():
    model = MLP()
    config = _make_velora_lora_config(target_modules=["lin0"], r=4, velora_scale=1.0, num_groups=7)

    model = get_peft_model(model, config)
    layer = model.base_model.model.lin0

    assert layer.lora_velora_embed["default"].shape == (3,)

    x = torch.randn(2, 16)
    model.train()
    output = model(x)
    output.sum().backward()

    assert output.shape == (2, 16)
    assert layer.lora_A["default"].weight.grad is not None


def test_velora_grouping_pads_remainder_features():
    x = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    grouped = _reshape_to_grouped_subtokens(x, num_groups=3)

    expected = torch.tensor(
        [
            [[0, 1], [2, 3], [4, 0]],
            [[5, 6], [7, 8], [9, 0]],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(grouped, expected)

    embed = torch.tensor([1.0, 0.0])
    compressed = _compress_activations(x, embed, num_groups=3)
    reconstructed = _reconstruct_activations(compressed, embed, in_features=5, velora_scale=1.0)

    assert compressed.shape == (2, 3)
    assert reconstructed.shape == x.shape


def test_reshape_to_grouped_subtokens_pads_non_divisible_input_dim_and_velora_autograd():
    x = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    grouped = _reshape_to_grouped_subtokens(x, num_groups=2)

    expected = torch.tensor(
        [
            [[0, 1], [2, 0]],
            [[3, 4], [5, 0]],
            [[6, 7], [8, 0]],
            [[9, 10], [11, 0]],
        ],
        dtype=torch.float32,
    )

    assert grouped.shape == (4, 2, 2)
    assert torch.equal(grouped, expected)

    x = x.detach().requires_grad_(True)
    weight = (torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10).requires_grad_(True)
    bias = (torch.arange(5, dtype=torch.float32) / 10).requires_grad_(True)
    embed = _normalize_projection(torch.tensor([1.0, 2.0]))

    output = VeloraFunction.apply(x, weight, bias, embed, 2, 0.5)
    assert torch.allclose(output, F.linear(x, weight, bias))

    output.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert weight.grad is not None
    assert weight.grad.shape == weight.shape
    assert bias.grad is not None
    assert bias.grad.shape == bias.shape


def _expected_batch_average_embed(x: torch.Tensor, num_groups: int, target: torch.Tensor) -> torch.Tensor:
    subtokens = _reshape_to_grouped_subtokens(x, num_groups)
    embed = _normalize_projection(subtokens.reshape(-1, subtokens.shape[-1]).mean(dim=0))
    return embed.to(target)


@pytest.mark.parametrize(
    "init_type, updates_every_forward",
    [
        ("batch_average_once", False),
        ("batch_average", True),
    ],
)
def test_velora_batch_average_update_policy(init_type, updates_every_forward):
    torch.manual_seed(0)
    model = get_peft_model(
        MLP(),
        _make_velora_lora_config(
            target_modules=["lin0"],
            r=4,
            velora_scale=1.0,
            init_type=init_type,
            num_groups=8,
        ),
    )
    layer = model.base_model.model.lin0

    x0 = torch.randn(2, 16)
    model.train()
    _ = model(x0)

    expected0 = _expected_batch_average_embed(x0, num_groups=8, target=layer.lora_velora_embed["default"])
    assert layer.lora_velora_initialized["default"] is True
    assert torch.allclose(layer.lora_velora_embed["default"], expected0, atol=1e-6, rtol=1e-5)

    stored_embed = layer.lora_velora_embed["default"].clone()
    x1 = torch.randn(2, 16) + 5
    _ = model(x1)

    if updates_every_forward:
        expected1 = _expected_batch_average_embed(x1, num_groups=8, target=layer.lora_velora_embed["default"])
        assert torch.allclose(layer.lora_velora_embed["default"], expected1, atol=1e-6, rtol=1e-5)
        assert not torch.allclose(layer.lora_velora_embed["default"], stored_embed, atol=1e-6, rtol=1e-5)
    else:
        assert torch.allclose(layer.lora_velora_embed["default"], stored_embed, atol=1e-6, rtol=1e-5)


def test_velora_reduces_saved_activation_memory_vs_vanilla_lora():
    torch.manual_seed(0)
    base_model = SingleLinear()
    lora_model = get_peft_model(
        copy.deepcopy(base_model),
        LoraConfig(target_modules=["lin"], r=8, lora_alpha=8, init_lora_weights=False),
    )
    velora_model = get_peft_model(
        copy.deepcopy(base_model),
        _make_velora_lora_config(
            target_modules=["lin"],
            r=8,
            lora_alpha=8,
            init_lora_weights=False,
            velora_scale=1.0,
            init_type="random",
            num_groups=32,
        ),
    )

    target = torch.randn(8, 4, 64)
    lora_model.train()
    velora_model.train()

    def make_loss(model, x):
        output = model(x)
        return (output - target).pow(2).mean()

    x_lora = torch.randn(8, 4, 128, requires_grad=True)
    x_velora = x_lora.detach().clone().requires_grad_(True)

    lora_saved_bytes = _saved_tensor_bytes(lambda: make_loss(lora_model, x_lora))
    velora_saved_bytes = _saved_tensor_bytes(lambda: make_loss(velora_model, x_velora))
    assert velora_saved_bytes < lora_saved_bytes


def test_velora_backward_matches_manual_reconstruction():
    torch.manual_seed(0)
    model = get_peft_model(
        SingleLinear(in_features=16, out_features=10, bias=False),
        _make_velora_lora_config(
            target_modules=["lin"],
            r=4,
            lora_alpha=2,
            velora_scale=0.5,
            init_type="random",
            num_groups=4,
        ),
    )
    layer = model.base_model.model.lin

    embed = _normalize_projection(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    layer.lora_velora_embed["default"] = embed.to(layer.lora_velora_embed["default"])
    layer.lora_velora_initialized["default"] = True
    with torch.no_grad():
        layer.lora_A["default"].weight.copy_(torch.arange(64, dtype=torch.float32).reshape(4, 16) / 100)
        layer.lora_B["default"].weight.copy_(torch.arange(40, dtype=torch.float32).reshape(10, 4) / 50)

    x = torch.randn(2, 3, 16, requires_grad=True)
    grad_output = torch.randn(2, 3, 10)

    model.train()
    output = model(x)
    output.backward(grad_output)

    # Compress the input activations as per equation (1) in the original paper
    compressed = _compress_activations(x.detach(), embed.to(x.dtype), num_groups=4)

    # Reconstruct the input activations during the backwards pass as per equation (2) in the original paper
    reconstructed = _reconstruct_activations(compressed, embed.to(x.dtype), in_features=16, velora_scale=0.5)

    # Original forward pass using original input activation
    grad_output_2d = grad_output.reshape(-1, 10)
    scaling = layer.scaling["default"]
    lora_A_weight = layer.lora_A["default"].weight.detach()
    lora_B_weight = layer.lora_B["default"].weight.detach()
    after_A = F.linear(x.detach(), lora_A_weight).reshape(-1, lora_A_weight.shape[0])

    # VeLoRA approximates the LoRA A gradient with the reconstructed input X_hat
    # instead of the original input X:
    # dL/dW_A = (scaling * dL/dY @ W_B)^T @ X_hat
    expected_grad_lora_A = ((grad_output_2d * scaling) @ lora_B_weight).transpose(0, 1) @ reconstructed
    expected_grad_lora_B = grad_output_2d.transpose(0, 1) @ after_A * scaling

    assert layer.base_layer.weight.grad is None
    assert torch.allclose(layer.lora_A["default"].weight.grad, expected_grad_lora_A, atol=1e-6, rtol=1e-5)
    assert torch.allclose(layer.lora_B["default"].weight.grad, expected_grad_lora_B, atol=1e-6, rtol=1e-5)
