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

"""Cortex-specific initialization, routing, structural merge, and checkpoint behavior."""

import argparse
import gc
import glob
import json
import os
import runpy
import subprocess
import sys
import warnings
import weakref
from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache, partial, wraps
from pathlib import Path
from types import MethodType

import pytest
import torch
from accelerate import cpu_offload
from accelerate.hooks import AlignDevicesHook, ModelHook, add_hook_to_module, remove_hook_from_module
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, PretrainedConfig

from peft import (
    CortexConfig,
    CortexFfnSpec,
    CortexMoeSpec,
    PeftConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
)
from peft.tuners.cortex import merge as merge_utils
from peft.tuners.cortex import model as model_utils
from peft.tuners.cortex import runtime as runtime_utils
from peft.tuners.cortex.layer import CortexExperts, CortexLayer, CortexMLP, ExpertLinear
from peft.tuners.cortex.runtime import check_moe_forward_hook
from peft.tuners.cortex.structure import describe_mlp, is_moe, resolve_target_branches, shared_names
from peft.utils.other import ModulesToSaveWrapper

from .testing_utils import ARCHITECTURES, BlockModel, NativeFFNModel, activate_branches, configured_branch
from .testing_utils import create_tiny_causal_lm as tiny_model
from .testing_utils import deterministic_device as device_runtime


DEVICE_CASES = [
    ("gemma3_text", "routed"),
    ("qwen3", "routed"),
    ("qwen3_moe", "shared"),
    ("qwen3_moe", "routed"),
    ("qwen3_moe", ["shared", "routed"]),
    ("gpt_oss", "shared"),
    ("gpt_oss", "routed"),
    ("gpt_oss", ["shared", "routed"]),
]


def reference_branch(layer, kind, value, expert=None, prefix=""):
    """Compute the branch directly from tensors, independently of Cortex's forward and activation dispatch."""

    def project(name, tensor):
        projection = getattr(layer, f"cortex_{prefix}{name}")["default"]
        weight, bias = projection.weight, projection.bias
        if expert is not None:
            weight, bias = weight[expert], bias[expert]
        return F.linear(tensor, weight, bias)

    gate, up = project("gate", value), project("up", value)
    if kind == "gpt_oss":
        gate, up = gate.clamp(max=7.0), up.clamp(-7.0, 7.0)
        hidden = gate * torch.sigmoid(gate * 1.702) * (up + 1)
    elif kind == "gemma3_text":
        hidden = F.gelu(gate, approximate="tanh") * up
    else:
        hidden = F.silu(gate) * up
    return project("down", hidden)


class CortexDeviceChecks:
    @pytest.mark.parametrize("kind,mode", DEVICE_CASES)
    @pytest.mark.parametrize("init", ["random", "sample"])
    def test_device_initialization_placement_and_two_training_steps(self, kind, mode, init, device, dtype):
        base = tiny_model(kind).to(device=device, dtype=dtype).eval()
        tokens = torch.tensor([[1, 4, 7, 3, 2], [1, 5, 8, 4, 2]], device=device)
        with torch.no_grad():
            expected = base(tokens, use_cache=False).logits
        frozen = [(parameter, parameter.detach().clone()) for parameter in base.parameters()]
        model = get_peft_model(
            base,
            CortexConfig(k=3, moe_target=mode, init_weights=init, adapter_bias=True, task_type="CAUSAL_LM"),
            autocast_adapter_dtype=False,
        )
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        assert trainable and all(
            parameter.device == tokens.device and parameter.dtype == dtype for parameter in trainable
        )
        for name, parameter in model.named_parameters():
            if "cortex_" in name and "down" in name:
                assert torch.count_nonzero(parameter) == 0
        torch.testing.assert_close(model(tokens, use_cache=False).logits, expected, rtol=0, atol=0)
        optimizer = torch.optim.AdamW(trainable, lr=0.03)
        model.train()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            result = model(tokens, labels=tokens, use_cache=False)
            assert torch.isfinite(result.loss) and torch.isfinite(result.logits).all()
            result.loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                    if step == 0 and "down" not in name:
                        assert torch.count_nonzero(parameter.grad) == 0
            downs = [
                parameter for name, parameter in model.named_parameters() if "down" in name and parameter.requires_grad
            ]
            assert any(parameter.grad.abs().sum() > 0 for parameter in downs)
            optimizer.step()
            assert all(torch.isfinite(parameter).all() for parameter in trainable)
        for parameter, before in frozen:
            assert not parameter.requires_grad and parameter.grad is None
            torch.testing.assert_close(parameter, before, rtol=0, atol=0)
        model.eval()
        with model.disable_adapter():
            torch.testing.assert_close(model(tokens, use_cache=False).logits, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("kind,mode", DEVICE_CASES)
    def test_device_fp32_checkpoint_and_native_export_contract(self, kind, mode, device, tmp_path):
        base = tiny_model(kind).to(device).eval()
        clean = deepcopy(base)
        model = get_peft_model(
            base,
            CortexConfig(k=3, moe_target=mode, init_weights="sample", adapter_bias=kind == "gpt_oss"),
        ).eval()
        activate_branches(model)
        tokens = torch.tensor([[1, 4, 7, 3, 2], [1, 5, 8, 4, 2]], device=device)
        with torch.no_grad():
            expected = model(tokens, use_cache=False).logits
        model.save_pretrained(tmp_path / "adapter", save_embedding_layers=False)
        loaded = PeftModel.from_pretrained(clean, tmp_path / "adapter", adapter_name="renamed").eval()
        assert all(parameter.device == tokens.device for parameter in loaded.parameters())
        torch.testing.assert_close(loaded(tokens, use_cache=False).logits, expected, rtol=0, atol=0)
        if kind == "qwen3_moe" and mode in ("shared", ["shared", "routed"]) and not base.config.norm_topk_prob:
            # Preserve the native routing policy. The default unnormalized public
            # branch can be checkpointed but cannot be folded into routed experts.
            with pytest.raises(ValueError, match="routing weights.*constant"):
                model.merge_adapter(safe_merge=True)
            assert not any(module.merged for module in model.modules() if isinstance(module, CortexLayer))
            torch.testing.assert_close(model(tokens, use_cache=False).logits, expected, rtol=0, atol=0)
            return
        model.merge_adapter(safe_merge=True)
        torch.testing.assert_close(model(tokens, use_cache=False).logits, expected, rtol=2e-4, atol=3e-5)
        model.unmerge_adapter()
        torch.testing.assert_close(model(tokens, use_cache=False).logits, expected, rtol=0, atol=0)
        native = model.merge_and_unload(safe_merge=True)
        assert not any(isinstance(module, CortexLayer) for module in native.modules())
        native.save_pretrained(tmp_path / "native")
        restored, info = AutoModelForCausalLM.from_pretrained(
            tmp_path / "native",
            output_loading_info=True,
            attn_implementation="eager",
            **({"experts_implementation": "eager"} if kind in {"qwen3_moe", "gpt_oss"} else {}),
        )
        assert not any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
        restored.to(device).eval()
        torch.testing.assert_close(restored(tokens, use_cache=False).logits, expected, rtol=2e-4, atol=3e-5)

    @pytest.mark.parametrize("kind,mode", DEVICE_CASES)
    def test_device_bf16_base_fp32_adapter_migration_and_training(self, kind, mode, device):
        if device.type == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
            pytest.skip("This CUDA GPU does not support native BF16.")
        # Exercise migration after injection as well as injection into a CUDA base.
        model = get_peft_model(
            tiny_model(kind).to(torch.bfloat16), CortexConfig(k=3, moe_target=mode, task_type="CAUSAL_LM")
        ).to(device)
        tokens = torch.tensor([[1, 4, 7, 3, 2], [1, 5, 8, 4, 2]], device=device)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        assert trainable and all(
            parameter.dtype == torch.float32 and parameter.device == tokens.device for parameter in trainable
        )
        assert all(
            parameter.dtype == torch.bfloat16 for parameter in model.parameters() if not parameter.requires_grad
        )
        result = model(tokens, labels=tokens, use_cache=False)
        assert torch.isfinite(result.loss)
        result.loss.backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable)
        assert any(parameter.grad.abs().sum() > 0 for parameter in trainable)
        optimizer = torch.optim.AdamW(trainable, lr=0.01)
        optimizer.step()
        assert all(torch.isfinite(parameter).all() for parameter in trainable)

    @pytest.mark.parametrize("kind,mode", DEVICE_CASES)
    def test_device_fp32_nonzero_formula_and_input_adapter_routing_gradients(self, kind, mode, device):
        block = tiny_model(kind).model.layers[0].mlp.to(device)
        dense = kind in {"gemma3_text", "qwen3"}
        base = block if dense else block.experts
        config = CortexConfig(k=3, adapter_bias=True)
        if dense:
            layer = CortexMLP(base, "default", config, describe_mlp(base))
        else:
            role = "routed_public" if isinstance(mode, list) else {"shared": "public", "routed": "routed"}[mode]
            layer = CortexExperts(base, "default", config, describe_mlp(base), role=role, public_scale=1.0)
        with torch.no_grad():
            for name, parameter in layer.named_parameters():
                if "cortex_" in name:
                    parameter.normal_(std=0.2)
        inputs = (torch.randn(5, 16, device=device) * 20).requires_grad_()
        indices = torch.tensor([[0, 2], [1, 2], [2, 0], [1, 0], [2, 1]], device=device)
        weights = torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.6, 0.4], [0.1, 0.9]], device=device)
        weights.requires_grad_()

        branch = partial(reference_branch, layer, kind)

        if dense:
            actual = layer(inputs)
            expected = base(inputs) + branch(inputs)
            parameters = [inputs]
        else:
            actual = layer(inputs, indices, weights)
            expected = base(inputs, indices, weights)
            if mode in ("routed", ["shared", "routed"]):
                expected = expected + torch.stack(
                    [sum(weights[t, j] * branch(inputs[t], int(indices[t, j])) for j in range(2)) for t in range(5)]
                )
            if mode in ("shared", ["shared", "routed"]):
                expected = expected + branch(inputs, prefix="shared_" if mode == ["shared", "routed"] else "")
            parameters = [inputs, weights]
        parameters += [parameter for name, parameter in layer.named_parameters() if "cortex_" in name]
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
        actual_gradients = torch.autograd.grad(actual.square().mean(), parameters, retain_graph=True)
        expected_gradients = torch.autograd.grad(expected.square().mean(), parameters)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-4, atol=2e-4)

    @pytest.mark.parametrize("kind", ["deepseek_v3", "qwen3_next", "nemotron_h"])
    def test_device_native_shared_and_latent_moe_training(self, kind, device):
        base = BlockModel(kind).to(device).eval()
        inputs = torch.randn(2, 5, 16, device=device)
        expected = base(inputs).detach()
        model = get_peft_model(base, CortexConfig(k=3, moe_target=["shared", "routed"]))
        torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
        model(inputs).square().mean().backward()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        assert trainable and all(parameter.device == inputs.device for parameter in trainable)
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable)
        assert any(parameter.grad.abs().sum() > 0 for parameter in trainable)

    @pytest.mark.parametrize("kind", ["qwen3_moe", "gpt_oss"])
    def test_device_packed_adamw_inactive_expert_decay(self, kind, device):
        base = tiny_model(kind).to(device)
        model = get_peft_model(base, CortexConfig(k=3, moe_target="routed", adapter_bias=True))
        layer = model.get_base_model().model.layers[0].mlp.experts
        with torch.no_grad():
            for name, parameter in layer.named_parameters():
                if "cortex_" in name:
                    parameter.normal_(std=0.2)
        parameters = [parameter for name, parameter in layer.named_parameters() if "cortex_" in name]
        optimizer = torch.optim.AdamW(parameters, lr=0.1, weight_decay=0.2)
        before = [parameter[2].detach().clone() for parameter in parameters]
        inputs = torch.ones(4, 16, device=device)
        indices = torch.tensor([[0, 1]] * 4, device=device)
        weights = torch.tensor([[0.7, 0.3]] * 4, device=device)
        layer(inputs, indices, weights).square().mean().backward()
        assert any(parameter.grad[:2].abs().sum() > 0 for parameter in parameters)
        assert all(torch.count_nonzero(parameter.grad[2]) == 0 for parameter in parameters)
        optimizer.step()
        for parameter, original in zip(parameters, before):
            torch.testing.assert_close(parameter[2], original * 0.98, rtol=1e-6, atol=1e-7)
            assert optimizer.state[parameter]["step"] == 1


PUBLIC_ROUTING_CASES = [
    (kind, mode) for kind in ("olmoe", "qwen3_moe", "gpt_oss") for mode in ("shared", ["shared", "routed"])
]


def public_config(mode, weighting="routing", **kwargs):
    return CortexConfig(k=3, moe_target=mode, public_weighting=weighting, **kwargs)


class CortexPublicRoutingChecks:
    @pytest.mark.parametrize("kind,mode", PUBLIC_ROUTING_CASES)
    def test_public_routing_matches_selected_expert_formula_and_gradients(self, kind, mode, device):
        block = tiny_model(kind).model.layers[0].mlp.to(device)
        layer = CortexExperts(
            block.experts,
            "default",
            public_config(mode, adapter_bias=True),
            describe_mlp(block.experts),
            role="public" if mode == "shared" else "routed_public",
        )
        with torch.no_grad():
            for name, parameter in layer.named_parameters():
                if "cortex_" in name:
                    parameter.normal_(std=0.2)
        x = (torch.randn(5, 16, device=device) * 20).requires_grad_()
        indices = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2], [1, 0]], device=device)
        # Different routing masses, including zero, rule out an accidental fixed
        # scale or normalization. Nonzero down bias must be weighted as well.
        weights = torch.tensor([[0.2, 0.3], [0.05, 0.15], [0.7, 0.3], [0.1, 0.6], [0.0, 0.0]], device=device)
        weights.requires_grad_()

        branch = partial(reference_branch, layer, kind)

        actual = layer(x, indices, weights)
        prefix = "shared_" if mode == ["shared", "routed"] else ""
        increments = []
        for token in range(len(x)):
            selected = []
            for slot in range(indices.shape[1]):
                delta = branch(x[token], prefix=prefix)
                if mode == ["shared", "routed"]:
                    delta = delta + branch(x[token], int(indices[token, slot]))
                selected.append(weights[token, slot] * delta)
            increments.append(sum(selected))
        expected = block.experts(x, indices, weights) + torch.stack(increments)
        torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-5)
        parameters = [x, weights] + [p for name, p in layer.named_parameters() if "cortex_" in name]
        actual_gradients = torch.autograd.grad(actual.square().mean(), parameters, retain_graph=True)
        expected_gradients = torch.autograd.grad(expected.square().mean(), parameters)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            assert torch.isfinite(actual_gradient).all()
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("kind,mode", PUBLIC_ROUTING_CASES)
    @pytest.mark.parametrize("init", ["random", "sample"])
    def test_public_routing_training_checkpoint_merge_and_native_reload(
        self, kind, mode, init, device, dtype, tmp_path
    ):
        base = tiny_model(kind).to(device=device, dtype=dtype).eval()
        clean = deepcopy(base)
        frozen = [(p, p.detach().clone()) for p in base.parameters()]
        tokens = torch.tensor([[1, 4, 7, 3, 2], [1, 5, 8, 4, 2]], device=device)
        with torch.no_grad():
            baseline = base(tokens, use_cache=False).logits
        model = get_peft_model(
            base,
            public_config(mode, init_weights=init, init_noise_std=0, adapter_bias=kind == "gpt_oss"),
            autocast_adapter_dtype=False,
        )
        torch.testing.assert_close(model(tokens, use_cache=False).logits, baseline, rtol=0, atol=0)
        trainable = [p for p in model.parameters() if p.requires_grad]
        before = [p.detach().clone() for p in trainable]
        assert all(p.device == tokens.device and p.dtype == dtype for p in trainable)
        optimizer = torch.optim.AdamW(trainable, lr=0.03)
        model.train()
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            loss = model(tokens, labels=tokens, use_cache=False).loss
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
            optimizer.step()
        assert any(not torch.equal(p, original) for p, original in zip(trainable, before))
        for parameter, original in frozen:
            assert parameter.grad is None
            torch.testing.assert_close(parameter, original, rtol=0, atol=0)
        model.eval()
        with torch.no_grad():
            expected = model(tokens, use_cache=False).logits.detach().clone()
        assert not torch.equal(expected, baseline)
        model.save_pretrained(tmp_path / "adapter", save_embedding_layers=False)
        saved_config = json.loads((tmp_path / "adapter/adapter_config.json").read_text())
        assert saved_config["public_weighting"] == "routing"
        loaded = PeftModel.from_pretrained(
            clean,
            tmp_path / "adapter",
            adapter_name="renamed",
            low_cpu_mem_usage=True,
            autocast_adapter_dtype=False,
        ).eval()
        assert loaded.peft_config["renamed"].public_weighting == "routing"
        assert all(p.device == tokens.device for p in loaded.parameters())
        torch.testing.assert_close(loaded(tokens, use_cache=False).logits, expected, rtol=0, atol=0)
        tolerances = {"rtol": 2e-4, "atol": 3e-5} if dtype == torch.float32 else {"rtol": 0.02, "atol": 0.002}
        model.merge_adapter(safe_merge=True)
        torch.testing.assert_close(model(tokens, use_cache=False).logits, expected, **tolerances)
        model.unmerge_adapter()
        torch.testing.assert_close(model(tokens, use_cache=False).logits, expected, rtol=0, atol=0)
        native = model.merge_and_unload(safe_merge=True)
        assert not any(isinstance(m, CortexLayer) for m in native.modules())
        if kind != "gpt_oss":
            assert native.config.norm_topk_prob is False
        torch.testing.assert_close(native(tokens, use_cache=False).logits, expected, **tolerances)
        native.save_pretrained(tmp_path / "native")
        restored, info = AutoModelForCausalLM.from_pretrained(
            tmp_path / "native",
            dtype=dtype,
            attn_implementation="eager",
            experts_implementation="eager",
            output_loading_info=True,
        )
        assert not any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
        restored.to(device).eval()
        torch.testing.assert_close(restored(tokens, use_cache=False).logits, expected, **tolerances)


def routed_branch(kind):
    current, layer = configured_branch(kind, "routed", torch.float32)
    parameters = [p for name, p in current.named_parameters() if "cortex_" in name]
    with torch.no_grad():
        for parameter in parameters:
            parameter.normal_(std=0.2)
    assert all(parameter.requires_grad for parameter in parameters)
    return layer, parameters


def routed_loss(layer, active):
    inputs = torch.ones(4, 8)
    indices = torch.tensor([active] * 4)
    weights = torch.tensor([[0.7, 0.3]] * 4)
    return layer(inputs, indices, weights).square().mean()


def cortex_config(mode="shared", **kwargs):
    return CortexConfig(k=3, moe_target=mode, task_type="CAUSAL_LM", **kwargs)


def task_config(kind, mode, init, width):
    return CortexConfig(
        k=width,
        moe_target=mode,
        init_weights=init,
        init_noise_std=0,
        adapter_bias=kind == "gpt_oss",
        task_type="CAUSAL_LM",
    )


def activate_task(model, current, previous, cumulative):
    if cumulative:
        model.base_model.set_adapter([*previous, current])
    else:
        model.set_adapter(current)
    # Activation enables gradients on active adapters. Freeze previous tasks
    # afterwards, and discard gradients left by the previous training stage.
    model.set_requires_grad(previous, requires_grad=False)
    model.set_requires_grad(current, requires_grad=True)
    model.zero_grad(set_to_none=True)


def logits(model, tokens):
    model.eval()
    with torch.no_grad():
        return model(tokens, use_cache=False).logits.detach().clone()


def train_task(model, tokens, current):
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable and all("cortex_" in name and f".{current}." in name for name in trainable)
    previous = {name: parameter.detach().clone() for name, parameter in trainable.items()}
    frozen = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters() if not parameter.requires_grad
    }
    # A fresh optimizer excludes previous adapters, including their old AdamW
    # momentum and decay. Frozen parameters must remain exactly unchanged.
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=0.03, weight_decay=0.1)
    model.train()
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        loss = model(tokens, labels=tokens, use_cache=False).loss
        assert torch.isfinite(loss)
        loss.backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable.values()
        )
        assert any(parameter.grad.abs().sum() > 0 for parameter in trainable.values())
        optimizer.step()
        for name, parameter in model.named_parameters():
            if name in frozen:
                assert not parameter.requires_grad and parameter.grad is None
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
    assert any(not torch.equal(parameter, previous[name]) for name, parameter in trainable.items())


def bypassing_kernel(self, hidden_states):
    raise AssertionError("The bypassing parent kernel must not execute.")


def _protected_moe(protected):
    torch.manual_seed(123)
    base = BlockModel("qwen3_next").eval()
    clean = deepcopy(base)
    retained = "shared_expert" if protected == "experts" else "experts"
    role = "shared" if retained == "shared_expert" else "routed"
    model = get_peft_model(
        base,
        CortexConfig(
            k=2,
            target_modules=[f"mlp.{retained}"],
            modules_to_save=[f"mlp.{protected}"],
            moe_target=role,
        ),
    ).eval()
    wrapper = base.get_submodule(f"mlp.{protected}")
    assert isinstance(wrapper, ModulesToSaveWrapper)
    return clean, model, wrapper, retained, role


def _module_name_boundary_model(nested=False):
    class CustomFFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(4, 12, bias=False)
            self.up_proj = nn.Linear(4, 12, bias=False)
            self.down_proj = nn.Linear(12, 4, bias=False)
            self.act_fn = nn.SiLU()

        def forward(self, inputs):
            return self.down_proj(self.act_fn(self.gate_proj(inputs)) * self.up_proj(inputs))

    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            if nested:
                self.nested = nn.ModuleDict({"foo_mlp": nn.Sequential(CustomFFN())})
            else:
                self.foo_mlp = CustomFFN()
            self.other_ffn = CustomFFN()
            self.mlp = nn.Linear(4, 4)

        def forward(self, inputs):
            branch = self.nested["foo_mlp"] if nested else self.foo_mlp
            return branch(inputs) + self.other_ffn(inputs) + self.mlp(inputs)

    return CustomModel().eval(), "nested.foo_mlp.0" if nested else "foo_mlp"


def assert_native_reload(native, tokens, expected, path, **tolerances):
    native.save_pretrained(path)
    kwargs = {"experts_implementation": "eager"} if any(is_moe(module) for module in native.modules()) else {}
    restored, info = AutoModelForCausalLM.from_pretrained(
        path, output_loading_info=True, attn_implementation="eager", **kwargs
    )
    assert not any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    restored.eval()
    torch.testing.assert_close(restored(tokens, use_cache=False).logits, expected, **tolerances)


class RenamedFfn(nn.Module):
    def __init__(self, config, width_field="ffn_channels"):
        super().__init__()
        self.config = config
        self.channels = getattr(config, width_field)
        self.w1 = nn.Linear(4, self.channels)
        self.w3 = nn.Linear(4, self.channels)
        self.w2 = nn.Linear(self.channels, 4)
        self.activation = nn.SiLU()

    def forward(self, inputs):
        return self.w2(self.activation(self.w1(inputs)) * self.w3(inputs))


class RenamedFusedFfn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.channels = config.ffn_channels
        self.w13 = nn.Linear(4, 2 * self.channels)
        self.w2 = nn.Linear(self.channels, 4)
        self.activation = nn.SiLU()

    def forward(self, inputs):
        gate, up = self.w13(inputs).chunk(2, dim=-1)
        return self.w2(self.activation(gate) * up)


class RenamedExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.channels = config.expert_channels
        self.num_experts = 3
        self.is_transposed = False
        self.is_concatenated = True
        self.input_weight = nn.Parameter(torch.randn(3, 2 * self.channels, 4) * 0.1)
        self.output_weight = nn.Parameter(torch.randn(3, 4, self.channels) * 0.1)
        self.input_weight_bias = nn.Parameter(torch.randn(3, 2 * self.channels) * 0.1)
        self.output_weight_bias = nn.Parameter(torch.randn(3, 4) * 0.1)
        self.activation = nn.SiLU()

    def forward(self, hidden_states, selected_ids, selected_weights):
        output = torch.zeros_like(hidden_states)
        for expert in range(self.num_experts):
            token, slot = torch.where(selected_ids == expert)
            gate, up = F.linear(
                hidden_states[token],
                self.input_weight[expert],
                self.input_weight_bias[expert],
            ).chunk(2, dim=-1)
            current = F.linear(
                self.activation(gate) * up,
                self.output_weight[expert],
                self.output_weight_bias[expert],
            )
            output.index_add_(0, token, current * selected_weights[token, slot, None])
        return output


class RenamedMoe(nn.Module):
    def __init__(self, config, shared):
        super().__init__()
        self.bank = RenamedExperts(config)
        self.selector = nn.Linear(4, 3, bias=False)
        if shared:
            field = "shared_channels" if hasattr(config, "shared_channels") else "ffn_channels"
            self.common = RenamedFfn(config, field)

    def forward(self, hidden_states):
        shape = hidden_states.shape
        inputs = hidden_states.reshape(-1, shape[-1])
        weights, indices = self.selector(inputs).softmax(-1).topk(2, dim=-1)
        output = self.bank(inputs, selected_ids=indices, selected_weights=weights)
        if hasattr(self, "common"):
            output = output + self.common(inputs)
        return output.reshape(shape)


def renamed_ffn_specs():
    return {
        "RenamedFfn": CortexFfnSpec(
            gate_proj="w1",
            up_proj="w3",
            down_proj="w2",
            act_fn="activation",
            width_field="ffn_channels",
            width_attributes=("channels",),
        ),
        "RenamedFusedFfn": CortexFfnSpec(
            gate_up_proj="w13",
            down_proj="w2",
            act_fn="activation",
            width_field="ffn_channels",
            width_attributes=("channels",),
        ),
        "RenamedExperts": CortexFfnSpec(
            packed=True,
            gate_up_proj="input_weight",
            down_proj="output_weight",
            act_fn="activation",
            width_field="expert_channels",
            width_attributes=("channels",),
        ),
    }


@pytest.mark.usefixtures("deterministic_cortex_cpu")
class TestCortexStructureSpecs:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("mlp.0", ("dense", None, ())),
            ("mlp.1", ("moe", "mlp.1.bank", ("mlp.1.common",))),
            ("mlp.1.bank", ("routed", "mlp.1.bank", ())),
            ("mlp.1.common", ("shared", None, ("mlp.1.common",))),
        ],
    )
    def test_target_branch_paths_follow_custom_specs(self, path, expected):
        config = PretrainedConfig(ffn_channels=8, expert_channels=6)
        base = NativeFFNModel(nn.Sequential(RenamedFfn(config), RenamedMoe(config, True)), config)
        adapter = CortexConfig(
            k=2,
            moe_target=["shared", "routed"],
            ffn_specs=renamed_ffn_specs(),
            moe_specs={
                "RenamedMoe": CortexMoeSpec(
                    experts="bank",
                    shared_modules=("common",),
                    routing_indices="selected_ids",
                    routing_weights="selected_weights",
                )
            },
        )
        assert resolve_target_branches(base, path, adapter.ffn_specs, adapter.moe_specs) == expected
        get_peft_model(base, adapter)
        # Wrapping a physical leaf does not change its native branch classification.
        assert resolve_target_branches(base, path, adapter.ffn_specs, adapter.moe_specs) == expected

    @pytest.mark.parametrize("shared_modules", [None, ()])
    def test_target_branch_paths_preserve_unknown_shared_ownership(self, shared_modules):
        config = PretrainedConfig(expert_channels=6)
        base = NativeFFNModel(RenamedMoe(config, False), config)
        specs = {"RenamedMoe": CortexMoeSpec(experts="bank", shared_modules=shared_modules)}
        assert resolve_target_branches(base, "mlp", renamed_ffn_specs(), specs) == (
            "moe",
            "mlp.bank",
            shared_modules,
        )

    @pytest.mark.parametrize("init", ["random", "sample"])
    @pytest.mark.parametrize("qualified", [False, True])
    @pytest.mark.parametrize("ffn_class", [RenamedFfn, RenamedFusedFfn])
    def test_renamed_dense_checkpoint_merge_and_export_padding(self, init, qualified, ffn_class, tmp_path):
        config = PretrainedConfig(ffn_channels=8)
        base = NativeFFNModel(nn.Sequential(ffn_class(config), ffn_class(config)), config).eval()
        clean = deepcopy(base)
        specs = renamed_ffn_specs()
        if qualified:
            specs[f"{ffn_class.__module__}.{ffn_class.__qualname__}"] = specs.pop(ffn_class.__name__)
        model = get_peft_model(
            base,
            CortexConfig(k=3, target_modules=["mlp.0"], ffn_specs=specs, init_weights=init),
        ).eval()
        activate_branches(model)
        inputs = torch.randn(2, 3, 4)
        expected = model(inputs).detach()
        assert not torch.allclose(expected, clean(inputs))
        model.save_pretrained(tmp_path, save_embedding_layers=False)
        loaded = PeftModel.from_pretrained(clean, tmp_path).eval()
        torch.testing.assert_close(loaded(inputs), expected)

        loaded.merge_adapter(safe_merge=True)
        assert clean.mlp[0].base_layer.channels == 11
        torch.testing.assert_close(loaded(inputs), expected)
        loaded.unmerge_adapter()
        assert clean.mlp[0].base_layer.channels == 8
        torch.testing.assert_close(loaded(inputs), expected)

        native = loaded.merge_and_unload(safe_merge=True)
        assert native.config.ffn_channels == 11
        assert [block.channels for block in native.mlp] == [11, 11]
        assert [block.w2.in_features for block in native.mlp] == [11, 11]
        torch.testing.assert_close(native(inputs), expected)
        # Both FFNs use the exported config field, including the untargeted sibling.
        restored_config = PretrainedConfig.from_dict(native.config.to_dict())
        restored = NativeFFNModel(
            nn.Sequential(ffn_class(restored_config), ffn_class(restored_config)),
            restored_config,
        ).eval()
        restored.load_state_dict(native.state_dict())
        torch.testing.assert_close(restored(inputs), expected)

    @pytest.mark.parametrize("shared", [False, True])
    @pytest.mark.parametrize("mode", ["shared", "routed", ["shared", "routed"]])
    def test_renamed_moe_branches_and_routing_roundtrip(self, shared, mode, tmp_path):
        config = PretrainedConfig(ffn_channels=8, expert_channels=6)
        base = NativeFFNModel(RenamedMoe(config, shared), config).eval()
        clean = deepcopy(base)
        adapter = CortexConfig(
            k=2,
            init_weights="sample",
            init_noise_std=0,
            moe_target=mode,
            public_weighting="routing",
            ffn_specs=renamed_ffn_specs(),
            moe_specs={
                "RenamedMoe": CortexMoeSpec(
                    experts="bank",
                    shared_modules=("common",) if shared else (),
                    routing_indices="selected_ids",
                    routing_weights="selected_weights",
                )
            },
        )
        model = get_peft_model(base, adapter).eval()
        expected_roles = {}
        if shared:
            if mode != "shared":
                expected_roles["mlp.bank"] = "routed"
            if mode != "routed":
                expected_roles["mlp.common"] = "shared"
        else:
            expected_roles["mlp.bank"] = {"shared": "public", "routed": "routed"}.get(str(mode), "routed_public")
        assert adapter.target_roles == expected_roles
        activate_branches(model)
        inputs = torch.randn(2, 3, 4)
        expected = model(inputs).detach()
        assert not torch.allclose(expected, clean(inputs))
        model(inputs).square().mean().backward()
        assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
        model.save_pretrained(tmp_path, save_embedding_layers=False)
        loaded = PeftModel.from_pretrained(clean, tmp_path).eval()
        torch.testing.assert_close(loaded(inputs), expected)
        loaded.merge_adapter(safe_merge=True)
        torch.testing.assert_close(loaded(inputs), expected)
        loaded.unmerge_adapter()
        torch.testing.assert_close(loaded(inputs), expected)
        native = loaded.merge_and_unload(safe_merge=True)
        restored_config = PretrainedConfig.from_dict(native.config.to_dict())
        restored = NativeFFNModel(RenamedMoe(restored_config, shared), restored_config).eval()
        restored.load_state_dict(native.state_dict())
        torch.testing.assert_close(restored(inputs), expected)

    def test_renamed_shared_ffn_uses_its_own_width_field(self):
        config = PretrainedConfig(ffn_channels=8, expert_channels=6, shared_channels=5)
        base = NativeFFNModel(nn.Sequential(RenamedFfn(config), RenamedMoe(config, shared=True)), config).eval()
        specs = renamed_ffn_specs()
        specs["RenamedFfn"] = CortexFfnSpec(
            gate_proj="w1",
            up_proj="w3",
            down_proj="w2",
            act_fn="activation",
            width_field="ffn_channels",
            shared_width_field="shared_channels",
            width_attributes=("channels",),
        )
        model = get_peft_model(
            base,
            CortexConfig(
                k=2,
                init_weights=False,
                target_modules=["mlp.1"],
                moe_target="shared",
                ffn_specs=specs,
                moe_specs={
                    "RenamedMoe": CortexMoeSpec(
                        experts="bank",
                        shared_modules=("common",),
                        routing_indices="selected_ids",
                        routing_weights="selected_weights",
                    )
                },
            ),
        ).eval()
        inputs = torch.randn(2, 3, 4)
        expected = model(inputs).detach()
        native = model.merge_and_unload(safe_merge=True)
        assert native.config.shared_channels == 7
        assert native.config.ffn_channels == 8
        assert native.config.expert_channels == 6
        restored_config = PretrainedConfig.from_dict(native.config.to_dict())
        restored = NativeFFNModel(
            nn.Sequential(RenamedFfn(restored_config), RenamedMoe(restored_config, shared=True)),
            restored_config,
        ).eval()
        restored.load_state_dict(native.state_dict())
        torch.testing.assert_close(restored(inputs), expected)

    @pytest.mark.parametrize("conflict", [False, True])
    def test_specs_across_adapters_resolve_class_aliases_before_injection(self, conflict, tmp_path):
        config = PretrainedConfig(ffn_channels=8, alternative_width=8)
        base = NativeFFNModel(nn.Sequential(RenamedFfn(config), RenamedFfn(config)), config).eval()
        clean = deepcopy(base)
        first = renamed_ffn_specs()["RenamedFfn"]
        model = get_peft_model(
            base,
            CortexConfig(
                k=3,
                init_weights=False,
                target_modules=["mlp.0"],
                ffn_specs={"RenamedFfn": first},
            ),
        ).eval()
        second = CortexFfnSpec(
            gate_proj="w1",
            up_proj="w3",
            down_proj="w2",
            act_fn="activation",
            width_field="alternative_width" if conflict else "ffn_channels",
            width_attributes=("channels",),
        )
        other = CortexConfig(
            k=2,
            init_weights=False,
            target_modules=["mlp.1"],
            ffn_specs={f"{RenamedFfn.__module__}.{RenamedFfn.__qualname__}": second},
        )
        inputs = torch.randn(2, 3, 4)
        before = model(inputs).detach()
        if conflict:
            with pytest.raises(ValueError, match="Conflicting Cortex"):
                model.add_adapter("second", other)
            assert not isinstance(base.mlp[1], CortexLayer)
            torch.testing.assert_close(model(inputs), before)
            return

        model.add_adapter("second", other)
        model.base_model.set_adapter(["default", "second"])
        expected = model(inputs).detach()
        assert not torch.allclose(expected, before)
        model.save_pretrained(tmp_path, save_embedding_layers=False)
        loaded = PeftModel.from_pretrained(clean, tmp_path).eval()
        loaded.load_adapter(tmp_path / "second", adapter_name="second")
        loaded.base_model.set_adapter(["default", "second"])
        torch.testing.assert_close(loaded(inputs), expected)
        native = loaded.merge_and_unload(adapter_names=["default", "second"], safe_merge=True)
        assert native.config.ffn_channels == 11
        torch.testing.assert_close(native(inputs), expected)

    @pytest.mark.parametrize("complete", [False, True])
    def test_each_adapter_keeps_the_specs_needed_for_independent_export(self, complete, tmp_path):
        class OtherFfn(RenamedFfn):
            pass

        config = PretrainedConfig(ffn_channels=8)
        base = NativeFFNModel(nn.Sequential(RenamedFfn(config), OtherFfn(config)), config).eval()
        clean = deepcopy(base)
        spec = renamed_ffn_specs()["RenamedFfn"]
        catalog = {"RenamedFfn": spec, "OtherFfn": spec}
        model = get_peft_model(
            base,
            CortexConfig(
                k=3,
                target_modules=["mlp.0"],
                init_weights=False,
                ffn_specs=catalog if complete else {"RenamedFfn": spec},
            ),
        ).eval()
        second = CortexConfig(
            k=2,
            target_modules=["mlp.1"],
            init_weights=False,
            ffn_specs=catalog if complete else {"OtherFfn": spec},
        )
        inputs = torch.randn(2, 3, 4)
        if not complete:
            before = model(inputs).detach()
            with pytest.raises(ValueError, match="Conflicting Cortex ffn_specs"):
                model.add_adapter("second", second)
            assert not isinstance(base.mlp[1], CortexLayer)
            torch.testing.assert_close(model(inputs), before)
            return

        model.add_adapter("second", second)
        model.set_adapter("second")
        expected = model(inputs).detach()
        model.save_pretrained(tmp_path, selected_adapters=["second"], save_embedding_layers=False)
        standalone = PeftModel.from_pretrained(clean, tmp_path / "second", adapter_name="second").eval()
        torch.testing.assert_close(standalone(inputs), expected)

        model.delete_adapter("default")
        for current in (model, standalone):
            native = current.merge_and_unload(safe_merge=True)
            assert native.config.ffn_channels == 10
            assert [block.channels for block in native.mlp] == [10, 10]
            torch.testing.assert_close(native(inputs), expected)
            restored_config = PretrainedConfig.from_dict(native.config.to_dict())
            restored = NativeFFNModel(
                nn.Sequential(RenamedFfn(restored_config), OtherFfn(restored_config)),
                restored_config,
            ).eval()
            restored.load_state_dict(native.state_dict())
            torch.testing.assert_close(restored(inputs), expected)

    @pytest.mark.parametrize("change", ["second_adapter", "config_mutation"])
    def test_native_moe_ownership_cannot_change_after_injection(self, change):
        source = BlockModel("qwen3_next")
        block_class = type(source.mlp).__name__
        base = NativeFFNModel(nn.Sequential(source.mlp, deepcopy(source.mlp)), source.config).eval()
        shared_width = source.config.shared_expert_intermediate_size
        expert_width = source.config.moe_intermediate_size
        model = get_peft_model(
            base,
            CortexConfig(k=2, init_weights=False, target_modules=["mlp.0"], moe_target="shared"),
        ).eval()
        inputs = torch.randn(2, 3, source.config.hidden_size)
        expected = model(inputs).detach()
        changed_specs = {block_class: CortexMoeSpec(shared_modules=())}

        if change == "second_adapter":
            other = CortexConfig(
                k=3,
                target_modules=["mlp.1"],
                moe_target="routed",
                moe_specs=changed_specs,
            )
            with pytest.raises(ValueError, match="Conflicting.*moe_specs|MoE"):
                model.add_adapter("second", other)
            assert not any(isinstance(module, CortexLayer) for module in base.mlp[1].modules())
        else:
            model.peft_config["default"].moe_specs = changed_specs
            with pytest.raises(ValueError, match="MoE"):
                model.merge_and_unload(safe_merge=True)
            model.peft_config["default"].moe_specs = {}

        assert model.peft_config["default"].target_roles == {"mlp.0.shared_expert": "shared"}
        assert not base.mlp[0].shared_expert.merged
        torch.testing.assert_close(model(inputs), expected)
        native = model.merge_and_unload(safe_merge=True)
        assert native.config.shared_expert_intermediate_size == shared_width + 2
        assert native.config.moe_intermediate_size == expert_width
        assert [block.shared_expert.down_proj.in_features for block in native.mlp] == [shared_width + 2] * 2
        torch.testing.assert_close(native(inputs), expected)

    @pytest.mark.parametrize("failure", ["missing_projection", "inconsistent_dimensions", "activation"])
    def test_invalid_ffn_binding_is_rejected_before_injection(self, failure):
        config = PretrainedConfig(ffn_channels=8)
        base = NativeFFNModel(nn.Sequential(RenamedFfn(config), RenamedFfn(config)), config)
        if failure == "missing_projection":
            del base.mlp[1].w2
        elif failure == "inconsistent_dimensions":
            base.mlp[1].w2 = nn.Linear(9, 4)
        else:
            base.mlp[1].activation = nn.Softmax(dim=-1)
        with pytest.raises(ValueError):
            get_peft_model(base, CortexConfig(k=2, ffn_specs=renamed_ffn_specs()))
        assert not any(isinstance(module, CortexLayer) for module in base.modules())

    @pytest.mark.parametrize("failure", ["missing_experts", "missing_shared", "unknown_shared"])
    def test_invalid_moe_boundaries_are_rejected_before_injection(self, failure):
        config = PretrainedConfig(ffn_channels=8, expert_channels=6)
        base = NativeFFNModel(RenamedMoe(config, shared=False), config)
        shared = {"missing_experts": (), "missing_shared": ("missing",), "unknown_shared": None}[failure]
        spec = CortexMoeSpec(
            experts="missing" if failure == "missing_experts" else "bank",
            shared_modules=shared,
            routing_indices="selected_ids",
            routing_weights="selected_weights",
        )
        with pytest.raises(ValueError):
            get_peft_model(
                base,
                CortexConfig(
                    k=2,
                    ffn_specs=renamed_ffn_specs(),
                    moe_specs={"RenamedMoe": spec},
                ),
            )
        assert not any(isinstance(module, CortexLayer) for module in base.modules())

    def test_moe_spec_does_not_prove_constant_routing_mass(self):
        config = PretrainedConfig(ffn_channels=8, expert_channels=6)
        base = NativeFFNModel(RenamedMoe(config, shared=False), config).eval()
        model = get_peft_model(
            base,
            CortexConfig(
                k=2,
                moe_target="shared",
                init_weights=False,
                ffn_specs=renamed_ffn_specs(),
                moe_specs={
                    "RenamedMoe": CortexMoeSpec(
                        experts="bank",
                        shared_modules=(),
                        routing_indices="selected_ids",
                        routing_weights="selected_weights",
                    )
                },
            ),
        ).eval()
        inputs = torch.randn(2, 3, 4)
        expected = model(inputs).detach()
        with pytest.raises(ValueError, match="routing weights.*constant"):
            model.merge_and_unload(safe_merge=True)
        assert not any(module.merged for module in model.modules() if isinstance(module, CortexLayer))
        torch.testing.assert_close(model(inputs), expected)


class TestCortexNativeModels:
    @pytest.fixture(autouse=True)
    def deterministic_cpu(self):
        previous_threads = torch.get_num_threads()
        torch.manual_seed(123)
        torch.set_num_threads(1)
        yield
        torch.set_num_threads(previous_threads)

    @pytest.mark.parametrize(
        "kind,path,expected",
        [
            ("qwen3_next", "mlp", ("moe", "mlp.experts", ("mlp.shared_expert",))),
            ("qwen3_next", "mlp.experts", ("routed", "mlp.experts", ())),
            ("qwen3_next", "mlp.shared_expert", ("shared", None, ("mlp.shared_expert",))),
            ("mistral", "model.layers.0.mlp", ("dense", None, ())),
        ],
    )
    def test_target_branch_paths_use_registered_and_inferred_layouts(self, kind, path, expected):
        base = BlockModel(kind) if kind == "qwen3_next" else tiny_model(kind)
        assert resolve_target_branches(base, path) == expected

    def test_olmoe_public_merge_rejected_without_modifying_model(self):
        base = tiny_model("olmoe").eval()
        assert base.config.norm_topk_prob is False
        model = get_peft_model(base, CortexConfig(k=3, moe_target="shared")).eval()
        activate_branches(model)
        tokens = torch.randint(3, 32, (1, 5))
        before = model(tokens).logits.detach()
        state = deepcopy(model.state_dict())
        with pytest.raises(ValueError, match="routing weights.*constant"):
            model.merge_and_unload()
        torch.testing.assert_close(model(tokens).logits, before, rtol=0, atol=0)
        assert state.keys() == model.state_dict().keys()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)

    def test_olmoe_standalone_dense_mlp_merges_and_reloads(self):
        # Transformers' OLMoE decoder only constructs packed experts. Exercise its separate native FFN directly.
        pytest.importorskip("transformers", minversion="5.17.0")
        from transformers import OlmoeConfig
        from transformers.models.olmoe.modeling_olmoe import OlmoeMLP

        config = OlmoeConfig(hidden_size=16, intermediate_size=24)
        base = NativeFFNModel(OlmoeMLP(config), config).eval()
        inputs = torch.randn(2, 4, 16)
        baseline = base(inputs).detach()
        model = get_peft_model(base, CortexConfig(k=3)).eval()
        assert model.peft_config["default"].target_roles == {"mlp": "dense"}
        activate_branches(model)
        expected = model(inputs).detach()
        assert not torch.allclose(expected, baseline)

        merged = model.merge_and_unload(safe_merge=True)
        assert merged.config.intermediate_size == 27
        restored = NativeFFNModel(OlmoeMLP(merged.config), merged.config).eval()
        restored.load_state_dict(merged.state_dict(), strict=True)
        torch.testing.assert_close(restored(inputs), expected)

    @pytest.mark.parametrize("kind", ["qwen3_moe", "minimax_m3_vl_text"])
    def test_mixed_dense_and_sparse_decoder_widths(self, kind, tmp_path):
        pytest.importorskip("transformers", minversion="5.17.0")
        roles = {"model.layers.0.mlp": "dense", "model.layers.1.mlp.experts": "routed"}
        if kind == "qwen3_moe":
            base = tiny_model(kind, mlp_only_layers=[0]).eval()
            mode = "routed"
            widths = {"intermediate_size": 27, "moe_intermediate_size": 14}
        else:
            # MiniMax uses the same FFN class for dense and shared layers but serializes each width separately.
            base = tiny_model(
                kind,
                mlp_layer_types=["dense", "sparse"],
                dense_intermediate_size=20,
                shared_intermediate_size=16,
                rotary_dim=8,
            ).eval()
            mode = ["shared", "routed"]
            roles["model.layers.1.mlp.shared_experts"] = "shared"
            widths = {"dense_intermediate_size": 23, "shared_intermediate_size": 21, "intermediate_size": 26}

        tokens = torch.tensor([[1, 3, 7, 11]])
        baseline = base(tokens, use_cache=False).logits.detach()
        model = get_peft_model(base, CortexConfig(k=3, k_routed=2, k_shared=5, moe_target=mode)).eval()
        assert model.peft_config["default"].target_roles == roles
        activate_branches(model)
        logits = model(tokens, use_cache=False).logits
        assert not torch.allclose(logits, baseline)
        logits.square().mean().backward()
        for path in roles:
            assert torch.count_nonzero(base.get_submodule(path).cortex_down["default"].weight.grad)

        merged = model.merge_and_unload(safe_merge=True)
        for field, width in widths.items():
            assert getattr(merged.config, field) == width
        torch.testing.assert_close(merged(tokens, use_cache=False).logits, logits.detach())
        assert_native_reload(merged, tokens, logits.detach(), tmp_path)

    @pytest.mark.parametrize("kind", ARCHITECTURES)
    @pytest.mark.parametrize("mode", ["shared", "routed", ["shared", "routed"]])
    def test_native_blocks_adapter_merge_and_native_reload(self, kind, mode, tmp_path):
        base = BlockModel(kind).eval()
        clean = deepcopy(base)
        native_shared = shared_names(base.mlp)
        x = torch.randn(2, 5, 16) * 20
        reference = base(x).detach()
        config = CortexConfig(k=3, k_shared=5, k_routed=2, moe_target=mode)
        model = get_peft_model(base, config).eval()
        torch.testing.assert_close(model(x), reference, rtol=0, atol=0)
        activate_branches(model)
        expected = model(x).detach()
        assert not torch.allclose(expected, reference)
        model(x).square().mean().backward()
        for name, parameter in model.named_parameters():
            if "cortex_" in name:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            else:
                assert parameter.grad is None
        with model.disable_adapter():
            torch.testing.assert_close(model(x), reference, rtol=0, atol=0)
        if native_shared:
            roles = {module.role for module in model.modules() if isinstance(module, CortexLayer)}
            assert roles == ({"shared", "routed"} if mode == ["shared", "routed"] else {mode})
        if kind == "nemotron_h" and mode != "shared":
            assert model.get_base_model().mlp.experts.structure.input_dim == 8
            assert model.get_base_model().mlp.experts.structure.has_gate is False
        model.save_pretrained(tmp_path / "adapter", save_embedding_layers=False)
        loaded = PeftModel.from_pretrained(clean, tmp_path / "adapter", adapter_name="renamed").eval()
        torch.testing.assert_close(loaded(x), expected)
        value = x.clone().requires_grad_()
        before_grad = torch.autograd.grad(model(value).sum(), value)[0]
        merged = model.merge_and_unload(safe_merge=True)
        assert not any(isinstance(m, CortexLayer) for m in merged.modules())
        torch.testing.assert_close(merged(x), expected, rtol=2e-4, atol=3e-5)
        after_grad = torch.autograd.grad(merged(value).sum(), value)[0]
        torch.testing.assert_close(after_grad, before_grad, rtol=5e-4, atol=5e-5)
        native = BlockModel(kind, type(merged.config).from_dict(merged.config.to_dict())).eval()
        native.load_state_dict(merged.state_dict(), strict=True)
        torch.testing.assert_close(native(x), merged(x), rtol=0, atol=0)

    @pytest.mark.parametrize(
        "kind,mode",
        [
            ("llama", "routed"),
            ("qwen2", "routed"),
            ("qwen3", "shared"),
            ("qwen3_5_text", "shared"),
            ("gemma3_text", "routed"),
            ("qwen3_moe", "routed"),
            ("mixtral", "shared"),
            ("mixtral", "routed"),
            ("mixtral", ["shared", "routed"]),
            ("gpt_oss", "shared"),
            ("gpt_oss", "routed"),
            ("gpt_oss", ["shared", "routed"]),
            ("olmoe", "routed"),
            ("gemma4_text", "shared"),
            ("gemma4_text", "routed"),
            ("gemma4_text", ["shared", "routed"]),
        ],
    )
    def test_full_native_merge_unmerge_and_save_reload(self, kind, mode, tmp_path):
        # Exercise the Qwen3.5 text FFN independently of optional linear-attention kernels.
        options = {"layer_types": ["full_attention"] * 2} if kind == "qwen3_5_text" else {}
        base = tiny_model(kind, **options).eval()
        if kind == "gemma4_text":
            # Rebuild with a shared-KV, double-width MLP to test width coupling.
            base.config.num_kv_shared_layers = 1
            base.config.use_double_wide_mlp = True
            base.config.layer_types = ["full_attention", "full_attention"]
            base = AutoModelForCausalLM.from_config(base.config).eval()
        original_config = deepcopy(base.config.to_dict())
        original_state = deepcopy(base.state_dict())
        tokens = torch.randint(3, 32, (2, 5))
        baseline = base(tokens).logits.detach()
        model = get_peft_model(base, CortexConfig(k=3, moe_target=mode, adapter_bias=kind == "gpt_oss")).eval()
        activate_branches(model)
        if kind == "gpt_oss":
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if "cortex_" in name and name.endswith("bias"):
                        parameter.normal_(std=0.1)
        expected = model(tokens).logits.detach()
        assert not torch.allclose(expected, baseline)
        model.merge_adapter(safe_merge=True)
        torch.testing.assert_close(model(tokens).logits, expected)
        model.unmerge_adapter()
        torch.testing.assert_close(model(tokens).logits, expected, rtol=0, atol=0)
        assert model.get_base_model().config.to_dict() == original_config
        for name, module in model.get_base_model().named_modules():
            if isinstance(module, CortexLayer):
                for key, parameter in module.base_layer.state_dict().items():
                    torch.testing.assert_close(parameter, original_state[name + "." + key], rtol=0, atol=0)
        model.merge_adapter()
        merged = model.merge_and_unload(safe_merge=True)
        torch.testing.assert_close(merged(tokens).logits, expected)
        assert_native_reload(merged, tokens, expected, tmp_path)

    @pytest.mark.parametrize("mode", ["shared", ["shared", "routed"]])
    def test_olmoe_public_branch_is_not_multiplied_by_routing_mass(self, mode):
        base = tiny_model("olmoe").eval()
        model = get_peft_model(base, CortexConfig(k=3, moe_target=mode))
        layer = model.get_base_model().model.layers[0].mlp.experts
        activate_branches(model)
        x = torch.randn(5, 16, requires_grad=True)
        indices = torch.tensor([[0, 1], [1, 2], [2, 0], [1, 0], [0, 2]])
        weights = torch.tensor([[0.2, 0.3], [0.1, 0.2], [0.4, 0.5], [0.3, 0.1], [0.2, 0.4]], requires_grad=True)
        delta = layer(x, indices, weights) - layer.base_layer(x, indices, weights)
        prefix = "shared_" if mode == ["shared", "routed"] else ""
        expected = layer._branch(x, "default", prefix=prefix)
        if mode == ["shared", "routed"]:
            expected = expected + torch.stack(
                [
                    sum(weights[t, j] * layer._branch(x[t], "default", int(indices[t, j])) for j in range(2))
                    for t in range(5)
                ]
            )
        torch.testing.assert_close(delta, expected)
        if mode == "shared":
            grad = torch.autograd.grad(delta.sum(), weights)[0]
            torch.testing.assert_close(grad, torch.zeros_like(grad), rtol=0, atol=0)
        with pytest.raises(ValueError, match="routing weights.*constant"):
            model.merge_adapter()

    @pytest.mark.parametrize(
        "kind,init,width", [("qwen3", "random", 3), ("gemma3_text", "sample", 5), ("mistral", "sample", 3)]
    )
    def test_partial_layer_export_pads_untargeted_layers(self, kind, init, width, tmp_path):
        base = tiny_model(kind).eval()
        before = deepcopy(base.model.layers[1].mlp.state_dict())
        model = get_peft_model(base, CortexConfig(k=width, init_weights=init, target_modules=["model.layers.0.mlp"]))
        activate_branches(model)
        tokens = torch.randint(3, 32, (1, 5))
        expected = model(tokens).logits.detach()
        merged = model.merge_and_unload()
        assert merged.config.intermediate_size == 24 + width
        padded = merged.model.layers[1].mlp
        for projection_kind in ("gate", "up"):
            projection = getattr(padded, f"{projection_kind}_proj")
            torch.testing.assert_close(
                projection.weight[:24], before[f"{projection_kind}_proj.weight"], rtol=0, atol=0
            )
            assert torch.count_nonzero(projection.weight[24:]) == 0
        torch.testing.assert_close(padded.down_proj.weight[:, :24], before["down_proj.weight"], rtol=0, atol=0)
        assert merged.model.layers[1].mlp.down_proj.weight[:, 24:].count_nonzero() == 0
        tolerances = {"rtol": 2e-4, "atol": 3e-5} if kind == "gemma3_text" else {}
        assert_native_reload(merged, tokens, expected, tmp_path, **tolerances)

    def test_unregistered_mistral_discovers_both_layers_and_reloads_adapter(self, tmp_path):
        base = tiny_model("mistral").eval()
        clean = deepcopy(base)
        tokens = torch.tensor([[1, 3, 5, 7]])
        baseline = clean(tokens).logits.detach()

        with pytest.warns(UserWarning, match="Cortex inferred unregistered Transformers layouts") as warnings:
            model = get_peft_model(base, CortexConfig(k=3)).eval()
        assert len(warnings) == 1
        roles = {f"model.layers.{i}.mlp": "dense" for i in range(2)}
        assert model.peft_config["default"].target_roles == roles
        activate_branches(model)
        adapted = model(tokens).logits.detach()
        assert not torch.allclose(adapted, baseline)

        model.save_pretrained(tmp_path, save_embedding_layers=False)
        # Saved target paths name physical leaves. Replaying them as CLI selections could change a MoE role.
        with pytest.warns(UserWarning, match="Cortex inferred unregistered Transformers layouts"):
            loaded = PeftModel.from_pretrained(clean, tmp_path, adapter_name="restored").eval()
        assert loaded.peft_config["restored"].target_roles == roles
        torch.testing.assert_close(loaded(tokens).logits, adapted)

    @pytest.mark.parametrize("explicit_spec", [False, True])
    def test_explicit_inferred_target_warning_links_docs_and_respects_spec(self, explicit_spec):
        config = CortexConfig(
            k=2,
            target_modules=["model.layers.0.mlp"],
            ffn_specs={"MistralMLP": CortexFfnSpec()} if explicit_spec else {},
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            get_peft_model(tiny_model("mistral"), config)
        inferred = [
            item for item in caught if "Cortex inferred unregistered Transformers layouts" in str(item.message)
        ]
        assert len(inferred) == (0 if explicit_spec else 1)
        if inferred:
            message = str(inferred[0].message)
            assert "https://huggingface.co/docs/peft/main/package_reference/cortex#supported-architectures" in message
            assert "examples/cortex/validate_layout.py" not in message
        assert config.target_roles == {"model.layers.0.mlp": "dense"}

    @pytest.mark.parametrize("target_talker", [False, True])
    def test_registered_ffn_in_omni_talker_requires_explicit_selection(self, target_talker, tmp_path):
        pytest.importorskip("transformers", minversion="5.17.0")
        from transformers import Qwen2_5OmniConfig, Qwen2_5OmniForConditionalGeneration

        dims = {
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
        }
        config = Qwen2_5OmniConfig(thinker_config={"text_config": dims}, talker_config=dims)
        # Keep the native audio/vision tower structure, but allocate only the two FFNs used in this test.
        with torch.device("meta"):
            base = Qwen2_5OmniForConditionalGeneration(config)
        # Some Transformers releases nest the thinker layers under language_model.
        paths = {
            path.split(".", 1)[0]: path for path, layer in base.named_modules() if type(layer).__name__ == "Qwen2MLP"
        }
        assert paths.keys() == {"thinker", "talker"}
        for path in paths.values():
            layer = base.get_submodule(path).to_empty(device="cpu")
            for projection in (layer.gate_proj, layer.up_proj, layer.down_proj):
                projection.reset_parameters()
        clean = deepcopy(base)

        selected = paths["talker" if target_talker else "thinker"]
        adapter = CortexConfig(k=3, target_modules=[selected] if target_talker else None)
        model = get_peft_model(base, adapter)
        assert adapter.target_roles == {selected: "dense"}
        activate_branches(model)
        inputs = torch.randn(1, 4, 16)
        expected = base.get_submodule(selected)(inputs).detach()
        assert not torch.allclose(expected, clean.get_submodule(selected)(inputs))

        model.save_pretrained(tmp_path, save_embedding_layers=False)
        loaded = PeftModel.from_pretrained(clean, tmp_path)
        assert loaded.peft_config["default"].target_roles == {selected: "dense"}
        native = loaded.merge_and_unload(safe_merge=True)
        torch.testing.assert_close(native.get_submodule(selected)(inputs), expected)
        assert native.config.thinker_config.text_config.intermediate_size == (24 if target_talker else 27)
        assert native.config.talker_config.intermediate_size == (27 if target_talker else 24)

    def test_unregistered_dense_requires_the_standard_ffn_computation(self, monkeypatch):
        base = tiny_model("mistral")
        mlp_class = type(base.model.layers[0].mlp)
        native_forward = mlp_class.forward

        # Adding a branch after output normalization does not equal merging it before normalization.
        def normalize_output(self, x):
            return F.normalize(native_forward(self, x), dim=-1)

        monkeypatch.setattr(mlp_class, "forward", normalize_output)
        with pytest.raises(ValueError, match="MistralMLP"):
            get_peft_model(base, CortexConfig(k=3))
        assert not any(isinstance(module, CortexLayer) for module in base.modules())

    def test_unregistered_layout_inference_stays_in_multimodal_text_tower(self, tmp_path, monkeypatch, capsys):
        pytest.importorskip("transformers", minversion="4.57.0")
        from transformers import CLIPVisionConfig, LlavaConfig, LlavaForConditionalGeneration

        text = tiny_model("mistral").config
        vision = CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=8,
            patch_size=4,
        )
        base = LlavaForConditionalGeneration(LlavaConfig(text_config=text, vision_config=vision)).eval()
        vision_path = next(path for path, module in base.named_modules() if type(module).__name__ == "CLIPMLP")
        with pytest.raises(ValueError, match="text backbone"):
            get_peft_model(base, CortexConfig(target_modules=[vision_path]))

        prefix = "model.language_model.layers"
        config = CortexConfig(k=3, exclude_modules=[f"{prefix}.1.mlp"])
        with pytest.warns(UserWarning, match="Cortex inferred unregistered Transformers layouts"):
            model = get_peft_model(base, config)
        assert config.target_roles == {f"{prefix}.0.mlp": "dense"}
        tokens = torch.tensor([[1, 3, 5, 7]])
        activate_branches(model)
        expected = model(tokens).logits.detach()
        native = model.merge_and_unload(safe_merge=True)
        assert native.config.text_config.intermediate_size == 27
        assert native.config.vision_config.intermediate_size == 24
        native.save_pretrained(tmp_path)
        loaded, info = LlavaForConditionalGeneration.from_pretrained(tmp_path, output_loading_info=True)
        assert not any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
        torch.testing.assert_close(loaded.eval()(tokens).logits, expected)

        # The validator must apply this exclusion in LLaVA's namespace, just as training did.
        self._run_layout_probe(monkeypatch, tmp_path, "--k", "3", "--exclude-modules", f"{prefix}.1.mlp")
        output = capsys.readouterr().out
        assert f"{prefix}.0.mlp (dense" in output
        assert f"{prefix}.1.mlp (dense" not in output
        assert "Validated 1/1 cases" in output

    @staticmethod
    def _run_layout_probe(monkeypatch, model_dir, *args):
        script = Path(__file__).resolve().parents[1] / "examples/cortex/validate_layout.py"
        monkeypatch.setattr(sys, "argv", [str(script), "--model-name", str(model_dir), *args])
        with device_runtime("cpu"):
            runpy.run_path(str(script), run_name="__main__")

    @pytest.mark.parametrize("kind", ["mistral", "qwen3_moe"])
    def test_layout_probe_checks_each_mode_at_each_width(self, kind, tmp_path, monkeypatch, capsys):
        tiny_model(kind).config.save_pretrained(tmp_path)
        modes = ["dense"] if kind == "mistral" else ["shared", "routed", "shared+routed"]
        # A routing-weighted public branch can be merged even when Qwen3-MoE does not normalize top-k weights.
        args = [] if kind == "mistral" else ["--public-weighting", "routing"]
        self._run_layout_probe(monkeypatch, tmp_path, *args)

        output = capsys.readouterr().out
        for mode in modes:
            for width in (2, 4):
                assert f"Case {mode}, k={width}\n" in output
        count = len(modes) * 2
        assert f"Validated {count}/{count} cases" in output

    @pytest.mark.parametrize("full_config", [False, True])
    def test_layout_probe_accepts_fresh_structure_config(self, full_config, tmp_path, monkeypatch, capsys):
        base = tiny_model("llama")
        base.config.save_pretrained(tmp_path / "model")
        mlp_class = type(base.model.layers[0].mlp)
        original_init = mlp_class.__init__

        def renamed_init(self, config):
            original_init(self, config)
            for old, new in (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2"), ("act_fn", "activation")):
                setattr(self, new, getattr(self, old))
                delattr(self, old)

        def renamed_forward(self, inputs):
            return self.w2(self.activation(self.w1(inputs)) * self.w3(inputs))

        monkeypatch.setattr(mlp_class, "__init__", renamed_init)
        monkeypatch.setattr(mlp_class, "forward", renamed_forward)
        values = {
            "ffn_specs": {"LlamaMLP": {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2", "act_fn": "activation"}},
            "target_modules": ["model.layers.0.mlp"],
            "k": 3,
            "init_weights": "sample",
        }
        if full_config:
            # Serialized defaults include moe_target='shared', which has no effect on a dense model.
            CortexConfig(**values).save_pretrained(tmp_path / "fresh")
            config_path = tmp_path / "fresh/adapter_config.json"
        else:
            config_path = tmp_path / "cortex.json"
            config_path.write_text(json.dumps(values))
        self._run_layout_probe(monkeypatch, tmp_path / "model", "--cortex-config", str(config_path))
        output = capsys.readouterr().out
        assert "Case dense, k=3" in output
        assert "model.layers.0.mlp (dense" in output
        assert "model.layers.1.mlp (dense" not in output
        assert "Validated 1/1 cases" in output
        # Probing must not turn the user's fresh JSON into a checkpoint configuration.
        assert not json.loads(config_path.read_text()).get("target_roles")

    def test_layout_probe_fresh_specs_drive_moe_matrix_and_cli_overrides(self, tmp_path):
        from examples.cortex.validate_layout import fresh_config, probe_cases

        base_config = PretrainedConfig(ffn_channels=8, expert_channels=6)
        base = NativeFFNModel(RenamedMoe(base_config, shared=True), base_config)
        config_path = tmp_path / "cortex.json"
        values = {
            "ffn_specs": {name: vars(spec) for name, spec in renamed_ffn_specs().items()},
            "moe_specs": {
                "RenamedMoe": {
                    "experts": "bank",
                    "shared_modules": ["common"],
                    "routing_indices": "selected_ids",
                    "routing_weights": "selected_weights",
                }
            },
            "public_weighting": "routing",
            "k_shared": 3,
            "init_noise_std": 0,
        }
        config_path.write_text(json.dumps(values))
        args = argparse.Namespace(
            target_modules=None,
            target_regex=None,
            exclude_modules=None,
            exclude_regex=None,
            moe_target=None,
            public_weighting=None,
            k=None,
            k_shared=None,
            k_routed=None,
        )
        loaded = fresh_config(config_path)
        cases = probe_cases(base, args, None, loaded)
        assert [(config.moe_target, config.k) for config, _ in cases] == [
            (mode, width) for mode in ("shared", "routed", ["shared", "routed"]) for width in (2, 4)
        ]
        for config, _ in cases:
            assert config.moe_specs["RenamedMoe"].experts == "bank"
            assert config.ffn_specs["RenamedFfn"].down_proj == "w2"
            assert config.public_weighting == "routing" and config.k_shared == 3
            assert config.init_noise_std == 0 and not config.target_roles
        assert loaded == json.loads(config_path.read_text())

        # Explicit JSON settings constrain the matrix; CLI options can replace them.
        loaded.update(k=7, moe_target="shared", target_modules=["mlp"])
        cases = probe_cases(base, args, None, loaded)
        assert [(config.moe_target, config.k) for config, _ in cases] == [("shared", 7)]
        args.k, args.moe_target, args.k_shared = [1, 5], ["routed"], 2
        cases = probe_cases(base, args, None, loaded)
        assert [(config.moe_target, config.k) for config, _ in cases] == [("routed", 1), ("routed", 5)]
        assert all(config.k_shared == 2 for config, _ in cases)
        assert loaded["k"] == 7 and loaded["moe_target"] == "shared"

    @pytest.mark.parametrize("full_config", [False, True])
    @pytest.mark.parametrize("with_dense", [False, True])
    @pytest.mark.parametrize("selection", ["parent", "shared", "routed", "both"])
    def test_layout_probe_modes_follow_explicit_moe_targets(self, selection, with_dense, full_config, tmp_path):
        from examples.cortex.validate_layout import fresh_config, probe_cases

        base_config = PretrainedConfig(ffn_channels=8, expert_channels=6)
        base = NativeFFNModel(
            nn.Sequential(RenamedFfn(base_config), RenamedMoe(base_config, shared=True)), base_config
        )
        paths = {
            "parent": ["mlp.1"],
            "shared": ["mlp.1.common"],
            "routed": ["mlp.1.bank"],
            "both": ["mlp.1.common", "mlp.1.bank"],
        }
        targets = paths[selection] + (["mlp.0"] if with_dense else [])
        values = {
            "target_modules": targets,
            "ffn_specs": {name: vars(spec) for name, spec in renamed_ffn_specs().items()},
            "moe_specs": {
                "RenamedMoe": {
                    "experts": "bank",
                    "shared_modules": ["common"],
                    "routing_indices": "selected_ids",
                    "routing_weights": "selected_weights",
                }
            },
        }
        if full_config:
            # A serialized default controls a selected parent, but cannot change a leaf's role.
            CortexConfig(**values).save_pretrained(tmp_path / "fresh")
            path = tmp_path / "fresh/adapter_config.json"
            assert json.loads(path.read_text())["moe_target"] == "shared"
        else:
            path = tmp_path / "cortex.json"
            path.write_text(json.dumps(values))
        args = argparse.Namespace(
            target_modules=None,
            target_regex=None,
            exclude_modules=None,
            exclude_regex=None,
            moe_target=None,
            public_weighting=None,
            k=[2, 4],
            k_shared=None,
            k_routed=None,
        )
        modes = [["shared"], ["routed"], ["shared", "routed"]]
        if selection == "parent" and full_config:
            modes = [["shared"]]
        elif selection != "parent":
            modes = [["shared", "routed"] if selection == "both" else [selection]]
        cases = probe_cases(base, args, None, fresh_config(path))
        assert [(mode, config.k) for config, mode in cases] == [(mode, width) for mode in modes for width in (2, 4)]
        for config, mode in cases:
            assert config.target_modules == set(targets)
            model = get_peft_model(deepcopy(base), config)
            expected = {}
            if "shared" in mode:
                expected["mlp.1.common"] = "shared"
            if "routed" in mode:
                expected["mlp.1.bank"] = "routed"
            if with_dense:
                expected["mlp.0"] = "dense"
            assert model.peft_config["default"].target_roles == expected
        if selection == "routed":
            args.moe_target = ["shared"]
            cases = probe_cases(base, args, None, fresh_config(path))
            assert all(config.moe_target == "shared" and mode == ["routed"] for config, mode in cases)

    @pytest.mark.parametrize("full_config", [False, True])
    @pytest.mark.parametrize("selection", ["shared", "routed", "both"])
    def test_layout_probe_explicit_native_leaves_merge_and_reload(
        self, selection, full_config, tmp_path, monkeypatch, capsys
    ):
        config = BlockModel("qwen3_next").config
        config.layer_types = ["full_attention"] * config.num_hidden_layers
        config.save_pretrained(tmp_path / "model")
        prefix = "model.layers.0.mlp"
        leaves = {"shared": prefix + ".shared_expert", "routed": prefix + ".experts"}
        modes = ["shared", "routed"] if selection == "both" else [selection]
        values = {"target_modules": [leaves[mode] for mode in modes], "k": 2}
        if full_config:
            CortexConfig(**values).save_pretrained(tmp_path / "fresh")
            path = tmp_path / "fresh/adapter_config.json"
        else:
            path = tmp_path / "cortex.json"
            path.write_text(json.dumps(values))
        self._run_layout_probe(monkeypatch, tmp_path / "model", "--cortex-config", str(path))
        output = capsys.readouterr().out
        assert f"Case {'+'.join(modes)}, k=2" in output
        assert "Validated 1/1 cases" in output
        for mode in modes:
            assert f"{leaves[mode]} ({mode}, cortex_down" in output

    @pytest.mark.parametrize(
        "selection,expected_modes,expected_roles",
        [
            pytest.param(
                {"moe_target": ["shared", "routed"]},
                ["shared", "routed"],
                {
                    "mlp.0.common": "shared",
                    "mlp.0.bank": "routed",
                    "mlp.1.common": "shared",
                    "mlp.1.bank": "routed",
                    "mlp.2": "dense",
                },
                id="automatic",
            ),
            pytest.param(
                {"exclude_modules": ["mlp.0"], "moe_target": ["shared", "routed"]},
                ["shared", "routed"],
                {"mlp.1.common": "shared", "mlp.1.bank": "routed", "mlp.2": "dense"},
                id="automatic-excluded-parent",
            ),
            pytest.param(
                {"exclude_modules": ["mlp.0", "mlp.1"], "moe_target": ["shared", "routed"]},
                [],
                {"mlp.2": "dense"},
                id="automatic-excluded-moe-parents",
            ),
            pytest.param(
                {"target_modules": ["mlp.0"], "moe_target": ["shared", "routed"]},
                ["shared", "routed"],
                {"mlp.0.common": "shared", "mlp.0.bank": "routed"},
                id="parent",
            ),
            pytest.param(
                {"target_modules": ["mlp.0.common"], "moe_target": "routed"},
                ["shared"],
                {"mlp.0.common": "shared"},
                id="shared-leaf",
            ),
            pytest.param(
                {"target_modules": ["mlp.0.bank"], "moe_target": "shared"},
                ["routed"],
                {"mlp.0.bank": "routed"},
                id="routed-leaf",
            ),
            pytest.param(
                {"target_modules": ["mlp.0.common", "mlp.0.bank"]},
                ["shared", "routed"],
                {"mlp.0.common": "shared", "mlp.0.bank": "routed"},
                id="both-leaves",
            ),
            pytest.param(
                {"target_modules": ["mlp.0"], "exclude_modules": ["mlp.0.common"]},
                ["routed"],
                {"mlp.0.bank": "routed"},
                id="excluded-shared",
            ),
            pytest.param(
                {"target_modules": ["mlp.0"], "exclude_modules": ["mlp.0.bank"]},
                ["shared"],
                {"mlp.0.common": "shared"},
                id="excluded-routed",
            ),
            pytest.param(
                {"target_modules": ["mlp.0", "mlp.1.bank"], "moe_target": "shared"},
                ["shared", "routed"],
                {"mlp.0.common": "shared", "mlp.1.bank": "routed"},
                id="parent-and-fixed-leaf",
            ),
            pytest.param(
                {"target_modules": ["mlp.0", "mlp.1.bank", "mlp.2"], "moe_target": "shared"},
                ["shared", "routed"],
                {"mlp.0.common": "shared", "mlp.1.bank": "routed", "mlp.2": "dense"},
                id="mixed-dense-parent-and-leaf",
            ),
        ],
    )
    def test_layout_probe_and_tuner_follow_expected_target_roles(self, selection, expected_modes, expected_roles):
        from examples.cortex.validate_layout import probe_cases

        config = PretrainedConfig(ffn_channels=8, expert_channels=6)
        base = NativeFFNModel(
            nn.Sequential(RenamedMoe(config, True), RenamedMoe(config, True), RenamedFfn(config)), config
        )
        values = {
            "ffn_specs": renamed_ffn_specs(),
            "moe_specs": {
                "RenamedMoe": CortexMoeSpec(
                    experts="bank",
                    shared_modules=("common",),
                    routing_indices="selected_ids",
                    routing_weights="selected_weights",
                )
            },
            **selection,
        }
        args = argparse.Namespace(
            target_modules=None,
            target_regex=None,
            exclude_modules=None,
            exclude_regex=None,
            moe_target=None,
            public_weighting=None,
            k=[2],
            k_shared=None,
            k_routed=None,
        )
        cases = probe_cases(base, args, None, values)
        assert len(cases) == 1
        adapter, modes = cases[0]
        assert modes == expected_modes
        if "moe_target" in selection:
            assert adapter.moe_target == selection["moe_target"]
        model = get_peft_model(base, adapter)
        assert model.peft_config["default"].target_roles == expected_roles
        assert {path: module.role for path, module in base.named_modules() if isinstance(module, CortexLayer)} == (
            expected_roles
        )

    @pytest.mark.parametrize("saved_target", ["mlp.1", "mlp.1.common"])
    def test_layout_probe_excludes_modules_reserved_for_full_training(self, saved_target):
        from examples.cortex.validate_layout import probe_cases

        config = PretrainedConfig(ffn_channels=8, expert_channels=6)
        base = NativeFFNModel(nn.Sequential(RenamedFfn(config), RenamedMoe(config, True)), config)
        values = {
            "ffn_specs": renamed_ffn_specs(),
            "moe_specs": {
                "RenamedMoe": CortexMoeSpec(
                    experts="bank",
                    shared_modules=("common",),
                    routing_indices="selected_ids",
                    routing_weights="selected_weights",
                )
            },
            "modules_to_save": [saved_target],
        }
        args = argparse.Namespace(
            target_modules=None,
            target_regex=None,
            exclude_modules=None,
            exclude_regex=None,
            moe_target=None,
            public_weighting=None,
            k=None,
            k_shared=None,
            k_routed=None,
        )
        modes = [] if saved_target == "mlp.1" else ["routed"]
        cases = probe_cases(base, args, None, values)
        assert [(mode, adapter.k) for adapter, mode in cases] == [(modes, width) for width in (2, 4)]
        for adapter, _ in cases:
            assert adapter.modules_to_save == [saved_target]
            model = get_peft_model(deepcopy(base), adapter)
            expected = {"mlp.0": "dense"}
            if modes:
                expected["mlp.1.bank"] = "routed"
            assert model.peft_config["default"].target_roles == expected

    def test_layout_probe_reduces_custom_constructor_width_fields(self, monkeypatch):
        from examples.cortex import validate_layout

        class CustomConfig(PretrainedConfig):
            model_type = "cortex_probe"

        config = CustomConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=4,
            ffn_channels=1_000_000,
            shared_channels=2_000_000,
        )
        monkeypatch.setattr(validate_layout.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
        monkeypatch.setattr(validate_layout.AutoModelForCausalLM, "_model_mapping", {CustomConfig: NativeFFNModel})
        monkeypatch.setattr(
            validate_layout.AutoModelForCausalLM,
            "from_config",
            lambda config, **kwargs: NativeFFNModel(RenamedFfn(config), config),
        )
        spec = CortexFfnSpec(width_field="ffn_channels", shared_width_field="shared_channels")
        base, _, _ = validate_layout.build_base("unused", ffn_specs={"RenamedFfn": spec})
        assert base.mlp.w1.out_features == base.config.ffn_channels
        assert 0 < base.config.ffn_channels < base.config.shared_channels <= 32

    def test_layout_probe_fresh_and_saved_configs_have_distinct_contracts(self, tmp_path):
        from examples.cortex.validate_layout import fresh_config, saved_config

        path = tmp_path / "cortex.json"
        fresh = CortexConfig(k=3, ffn_specs={"LlamaMLP": CortexFfnSpec()})
        path.write_text(json.dumps(fresh.to_dict()))
        assert fresh_config(path)["k"] == 3
        with pytest.raises(ValueError, match="target_roles"):
            saved_config(path)
        fresh.target_roles = {"model.layers.0.mlp": "dense"}
        path.write_text(json.dumps(fresh.to_dict()))
        with pytest.raises(ValueError, match="target_roles|adapter-config"):
            fresh_config(path)
        assert saved_config(path).target_roles == fresh.target_roles
        path.write_text(json.dumps({"peft_type": "LORA"}))
        with pytest.raises(ValueError, match="Cortex|CORTEX"):
            fresh_config(path)

    @pytest.mark.parametrize("extra", [["--k", "3"], ["--cortex-config", "fresh.json"]])
    def test_layout_probe_saved_config_rejects_overrides(self, tmp_path, monkeypatch, extra):
        config = CortexConfig(k=2, target_roles={"model.layers.0.mlp": "dense"})
        config.save_pretrained(tmp_path / "adapter")
        with pytest.raises(SystemExit) as error:
            self._run_layout_probe(
                monkeypatch, tmp_path / "model", "--adapter-config", str(tmp_path / "adapter"), *extra
            )
        assert error.value.code == 2

    def test_layout_probe_preserves_the_rotary_width_of_split_attention(self, tmp_path, monkeypatch, capsys):
        config = BlockModel("deepseek_v3").config
        config.first_k_dense_replace = 0
        config.q_lora_rank = config.kv_lora_rank = 16
        config.head_dim = config.qk_rope_head_dim = 16
        config.save_pretrained(tmp_path)

        self._run_layout_probe(monkeypatch, tmp_path, "--k", "2", "--moe-target", "shared", "routed")
        output = capsys.readouterr().out
        assert "(shared, cortex_down" in output and "(routed, cortex_down" in output
        assert "Validated 1/1 cases" in output

    def test_layout_probe_replays_unmergeable_saved_public_role(self, tmp_path, monkeypatch, capsys):
        base = tiny_model("olmoe")
        assert not base.config.norm_topk_prob
        base.config.save_pretrained(tmp_path / "model")
        model = get_peft_model(base, CortexConfig(k=3, moe_target="shared"))
        model.peft_config["default"].save_pretrained(tmp_path / "adapter")

        with pytest.raises(SystemExit) as result:
            self._run_layout_probe(monkeypatch, tmp_path / "model", "--adapter-config", str(tmp_path / "adapter"))
        assert result.value.code == 1
        output = capsys.readouterr().out
        assert "(public, cortex_down" in output
        assert "Cannot merge an unweighted public branch" in output
        assert "Validated 0/1 cases" in output

    @pytest.mark.parametrize("modes", [["routed"], ["shared", "routed"]])
    def test_layout_probe_cannot_count_masked_experts_as_a_routed_effect(self, modes, tmp_path, monkeypatch, capsys):
        # The dense layer and the native shared FFN can change logits, but every routed expert is multiplied by zero.
        config = BlockModel("deepseek_v3").config
        config.qk_nope_head_dim = config.qk_rope_head_dim = config.v_head_dim = 8
        config.first_k_dense_replace = 1
        config.routed_scaling_factor = 0.0
        config.save_pretrained(tmp_path)
        with pytest.raises(SystemExit) as result:
            self._run_layout_probe(monkeypatch, tmp_path, "--k", "2", "--moe-target", *modes)
        assert result.value.code == 1
        output = capsys.readouterr().out
        assert "No measurable routed branch effect" in output
        assert "Validated 0/1 cases" in output

    @pytest.mark.parametrize("reason", ["mismatched", "ambiguous"])
    def test_unregistered_width_must_be_unambiguous_for_native_export(self, reason, tmp_path):
        base = tiny_model("mistral").eval()
        if reason == "mismatched":
            base.config.intermediate_size += 1
        else:
            base.config.dense_intermediate_size = base.config.intermediate_size
        clean = deepcopy(base)
        with pytest.warns(UserWarning, match="Cortex inferred unregistered Transformers layouts"):
            model = get_peft_model(base, CortexConfig(k=3))
        activate_branches(model)
        tokens = torch.tensor([[1, 3, 5, 7]])
        expected = model(tokens).logits.detach()
        assert not torch.allclose(expected, clean(tokens).logits)
        layer = base.model.layers[0].mlp
        weight = layer.base_layer.down_proj.weight
        width = base.config.intermediate_size

        with pytest.raises(ValueError, match="intermediate_size|[Aa]mbiguous"):
            model.merge_and_unload(safe_merge=True)
        assert not layer.merged and layer.base_layer.down_proj.weight is weight
        assert base.config.intermediate_size == width
        model.save_pretrained(tmp_path, save_embedding_layers=False)
        loaded = PeftModel.from_pretrained(clean, tmp_path).eval()
        torch.testing.assert_close(loaded(tokens).logits, expected)

    @pytest.mark.parametrize("mode", ["routed", ["shared", "routed"]])
    def test_unregistered_qwen2_moe_resolves_roles_and_native_export_widths(self, mode, tmp_path):
        pytest.importorskip("transformers", minversion="5.17.0")
        from transformers import Qwen2MoeConfig

        # The first layer is dense; the second has a shared FFN and packed routed experts.
        config = Qwen2MoeConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=12,
            shared_expert_intermediate_size=16,
            mlp_only_layers=[0],
        )
        base = AutoModelForCausalLM.from_config(config, experts_implementation="eager").eval()
        tokens = torch.tensor([[1, 3, 5, 7]])
        baseline = base(tokens).logits.detach()
        with pytest.warns(UserWarning, match="Cortex inferred unregistered Transformers layouts") as warnings:
            model = get_peft_model(base, CortexConfig(k=3, k_routed=2, k_shared=5, moe_target=mode)).eval()
        assert len(warnings) == 1
        roles = {"model.layers.0.mlp": "dense", "model.layers.1.mlp.experts": "routed"}
        if "shared" in mode:
            roles["model.layers.1.mlp.shared_expert"] = "shared"
        assert model.peft_config["default"].target_roles == roles

        activate_branches(model)
        adapted = model(tokens).logits.detach()
        assert not torch.allclose(adapted, baseline)
        merged = model.merge_and_unload(safe_merge=True)
        assert merged.config.intermediate_size == 27
        assert merged.config.moe_intermediate_size == 14
        assert merged.config.shared_expert_intermediate_size == (21 if "shared" in mode else 16)
        torch.testing.assert_close(merged(tokens).logits, adapted)
        assert_native_reload(merged, tokens, adapted, tmp_path)

    def test_unregistered_moe_does_not_silently_select_dense_siblings(self):
        pytest.importorskip("transformers", minversion="5.17.0")
        from transformers.models.llama4.modeling_llama4 import Llama4TextConfig, Llama4TextMoe

        # Llama4 preweights its inputs. Its expert bank never receives token IDs and weights.
        config = Llama4TextConfig(hidden_size=16, intermediate_size=24, num_local_experts=4, num_experts_per_tok=2)
        base = nn.Sequential(tiny_model("qwen3").model.layers[0].mlp, Llama4TextMoe(config))
        with pytest.raises(ValueError, match=r"Llama4|1\.experts"):
            get_peft_model(base, CortexConfig(k=3, moe_target="routed"))
        assert not any(isinstance(module, CortexLayer) for module in base.modules())

        allowed = get_peft_model(base, CortexConfig(k=3, moe_target="routed", exclude_modules=["1"]))
        assert allowed.peft_config["default"].target_roles == {"0": "dense"}

    @pytest.mark.parametrize("kind", ["granitemoe", "granitemoeshared"])
    def test_inferred_moe_cannot_assume_shared_experts_are_absent(self, kind, tmp_path):
        pytest.importorskip("transformers", minversion="5.17.0")
        from transformers import AutoConfig

        config = AutoConfig.for_model(
            kind,
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_local_experts=4,
            num_experts_per_tok=2,
            shared_intermediate_size=20,
        )
        base = AutoModelForCausalLM.from_config(config, experts_implementation="eager").eval()
        clean = deepcopy(base)
        if kind == "granitemoeshared":
            # The decoder adds this sibling to the MoE output, outside block_sparse_moe.
            assert base.model.layers[0].shared_mlp is not None

        for mode in ("shared", ["shared", "routed"]):
            with pytest.raises(ValueError, match="Cannot infer.*shared"):
                get_peft_model(base, CortexConfig(k=3, moe_target=mode))
            assert not any(isinstance(layer, CortexLayer) for layer in base.modules())

        model = get_peft_model(base, CortexConfig(k=3, moe_target="routed")).eval()
        roles = {"model.layers.0.block_sparse_moe.experts": "routed"}
        assert model.peft_config["default"].target_roles == roles
        activate_branches(model)
        tokens = torch.tensor([[1, 3, 8, 2]])
        expected = model(tokens).logits.detach()
        assert not torch.allclose(expected, clean(tokens).logits)
        model.save_pretrained(tmp_path, save_embedding_layers=False)
        loaded = PeftModel.from_pretrained(clean, tmp_path).eval()
        assert loaded.peft_config["default"].target_roles == roles
        torch.testing.assert_close(loaded(tokens).logits, expected)

        # A checkpoint can explicitly declare a public branch; replay it without rediscovery.
        saved_public = CortexConfig(
            k=3, target_modules=list(roles), target_roles=dict.fromkeys(roles, "public"), public_weighting="routing"
        )
        fresh = AutoModelForCausalLM.from_config(config, experts_implementation="eager")
        original = deepcopy(fresh)
        public_model = get_peft_model(fresh, saved_public).eval()
        activate_branches(public_model)
        expected = public_model(tokens).logits.detach()
        public_model.save_pretrained(tmp_path / "public", save_embedding_layers=False)
        restored = PeftModel.from_pretrained(original, tmp_path / "public").eval()
        assert restored.peft_config["default"].target_roles == dict.fromkeys(roles, "public")
        torch.testing.assert_close(restored(tokens).logits, expected)
        torch.testing.assert_close(restored.merge_and_unload(safe_merge=True)(tokens).logits, expected)

    @pytest.mark.parametrize("kind", ["mistral", "qwen2_moe"])
    def test_unregistered_ffn_requires_unmodified_forward(self, kind):
        if kind == "qwen2_moe":
            pytest.importorskip("transformers", minversion="5.17.0")
        base = tiny_model(kind)
        mlp = base.model.layers[0].mlp
        if kind == "mistral":
            mlp.up_proj.register_forward_hook(lambda layer, inputs, result: 3 * result)
        else:
            native_forward = mlp.experts.forward

            @wraps(native_forward)
            def rescale(*args, **kwargs):
                return 2 * native_forward(*args, **kwargs)

            mlp.experts.forward = rescale

        with pytest.raises(ValueError, match="forward|hooks"):
            get_peft_model(base, CortexConfig(k=3, moe_target="routed"))
        assert not any(isinstance(layer, CortexLayer) for layer in base.modules())

    @pytest.mark.parametrize("sibling", ["shared_mlp", "other_ffn"])
    def test_inferred_routed_only_does_not_adapt_an_ambiguous_dense_sibling(self, sibling):
        pytest.importorskip("transformers", minversion="5.17.0")
        base = tiny_model("granitemoeshared")
        decoder = base.model.layers[0]
        setattr(decoder, sibling, tiny_model("mistral").model.layers[0].mlp)
        config = CortexConfig(k=3, moe_target="routed")
        if sibling == "other_ffn":
            with pytest.raises(ValueError, match="other_ffn"):
                get_peft_model(base, config)
            assert not any(isinstance(layer, CortexLayer) for layer in base.modules())
            config.exclude_modules = [sibling]

        model = get_peft_model(base, config)
        assert model.peft_config["default"].target_roles == {
            f"model.layers.{i}.block_sparse_moe.experts": "routed" for i in range(2)
        }

    def test_merge_preflight_rejects_nonfinite_weights_and_unsupported_bias(self):
        model = get_peft_model(tiny_model("qwen3"), CortexConfig(k=3))
        layers = [m for m in model.modules() if isinstance(m, CortexLayer)]
        with torch.no_grad():
            layers[-1].cortex_down["default"].weight[0, 0] = float("nan")
        first_weight = layers[0].base_layer.up_proj.weight
        with pytest.raises(ValueError, match="Non-finite"):
            model.merge_and_unload(safe_merge=True)
        assert layers[0].base_layer.up_proj.weight is first_weight
        assert not any(layer.merged for layer in layers)
        biased = get_peft_model(tiny_model("qwen3"), CortexConfig(k=3, adapter_bias=True))
        with pytest.raises(ValueError, match="bias-free"):
            biased.merge_and_unload()

    @pytest.mark.parametrize("kind", ["olmoe", "qwen3_moe"])
    def test_normalized_public_branch_merges_but_changed_router_is_rejected(self, kind):
        base = tiny_model(kind).eval()
        base.config.norm_topk_prob = True
        for layer in base.model.layers:
            layer.mlp.gate.norm_topk_prob = True
        model = get_peft_model(base, CortexConfig(k=3, moe_target=["shared", "routed"]))
        activate_branches(model)
        tokens = torch.randint(3, 32, (1, 5))
        expected = model(tokens).logits.detach()
        block = model.get_base_model().model.layers[-1].mlp
        block.gate.norm_topk_prob = False
        with pytest.raises(ValueError, match="normalization changed"):
            model.merge_and_unload()
        assert not any(m.merged for m in model.modules() if isinstance(m, CortexLayer))
        block.gate.norm_topk_prob = True
        merged = model.merge_and_unload()
        torch.testing.assert_close(merged(tokens).logits, expected)

    @pytest.mark.parametrize("kind,mode", [("gpt_oss", ["shared", "routed"]), ("olmoe", "routed")])
    def test_multiple_active_adapters_merge_and_unmerge(self, kind, mode):
        model = get_peft_model(tiny_model(kind).eval(), CortexConfig(k=3, moe_target=mode))
        model.add_adapter("second", CortexConfig(k=5, moe_target=mode))
        activate_branches(model)
        model.base_model.set_adapter(["default", "second"])
        tokens = torch.randint(3, 32, (1, 5))
        expected = model(tokens).logits.detach()
        model.merge_adapter()
        torch.testing.assert_close(model(tokens).logits, expected)
        model.unmerge_adapter()
        torch.testing.assert_close(model(tokens).logits, expected, rtol=0, atol=0)
        model.set_adapter("second")
        second = model(tokens).logits.detach()
        merged = model.merge_and_unload(adapter_names=["second"])
        torch.testing.assert_close(merged(tokens).logits, second)

    def test_gemma4_disabled_moe_discovers_dense_ffns_and_exports(self, tmp_path):
        torch.manual_seed(123)
        config = tiny_model("gemma4_text").config
        config.enable_moe_block = False
        config.num_kv_shared_layers = 1
        config.use_double_wide_mlp = True
        config.layer_types = ["full_attention", "full_attention"]
        base = AutoModelForCausalLM.from_config(config).eval()
        assert all(not layer.enable_moe_block for layer in base.model.layers)
        tokens = torch.randint(3, 32, (2, 5))
        original = base(tokens).logits.detach()
        model = get_peft_model(base, CortexConfig(k=3)).eval()
        assert model.peft_config["default"].target_roles == {
            "model.layers.0.mlp": "dense",
            "model.layers.1.mlp": "dense",
        }
        assert sum(isinstance(module, CortexLayer) for module in model.modules()) == 2
        activate_branches(model)
        expected = model(tokens).logits.detach()
        assert not torch.allclose(expected, original)
        native = model.merge_and_unload(safe_merge=True)
        assert not any(isinstance(module, CortexLayer) for module in native.modules())
        assert native.config.enable_moe_block is False
        assert native.config.intermediate_size == 27
        assert [layer.mlp.down_proj.in_features for layer in native.model.layers] == [27, 54]
        torch.testing.assert_close(native(tokens).logits, expected, rtol=2e-4, atol=3e-5)
        assert_native_reload(native, tokens, expected, tmp_path, rtol=2e-4, atol=3e-5)

    @pytest.mark.parametrize("init", ["random", "sample"])
    def test_gemma3_multiple_adapters_nonzero_gradients_checkpoint_and_native_export(self, init, tmp_path):
        base = tiny_model("gemma3_text").eval()
        clean = deepcopy(base)
        model = get_peft_model(base, CortexConfig(k=3, init_weights=init, task_type="CAUSAL_LM")).eval()
        model.add_adapter("second", CortexConfig(k=5, init_weights=init, task_type="CAUSAL_LM"))
        model.base_model.set_adapter(["default", "second"])
        activate_branches(model)
        tokens = torch.tensor([[1, 4, 7, 3, 2], [1, 5, 8, 4, 2]])
        actual = model(tokens, labels=tokens, use_cache=False)
        expected = actual.logits.detach()
        second = [p for name, p in model.named_parameters() if "cortex_" in name and ".second." in name]
        gradients = torch.autograd.grad(actual.loss, second)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(gradient.abs().sum() > 0 for gradient in gradients)
        model.save_pretrained(tmp_path / "adapter", save_embedding_layers=False)
        loaded = PeftModel.from_pretrained(clean, tmp_path / "adapter", is_trainable=True)
        loaded.load_adapter(tmp_path / "adapter" / "second", "second", is_trainable=True)
        loaded.base_model.set_adapter(["default", "second"])
        loaded.eval()
        torch.testing.assert_close(loaded(tokens, use_cache=False).logits, expected, rtol=0, atol=0)
        merged = model.merge_and_unload(safe_merge=True)
        assert merged.config.intermediate_size == 32
        torch.testing.assert_close(merged(tokens, use_cache=False).logits, expected, rtol=2e-4, atol=3e-5)
        assert_native_reload(merged, tokens, expected, tmp_path / "native", rtol=2e-4, atol=3e-5)


class TestCortexExpertOptimizer:
    @pytest.fixture(autouse=True)
    def deterministic_cpu(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            yield
        torch.set_num_threads(previous_threads)

    @pytest.mark.parametrize("kind", ["qwen3_moe", "gpt_oss"])
    def test_packed_adamw_delayed_expert_uses_shared_counter_and_inactive_momentum(self, kind):
        layer, parameters = routed_branch(kind)
        optimizer = torch.optim.AdamW(parameters, lr=0.01, weight_decay=0)
        packed = layer.cortex_gate["default"].weight
        first_visit = packed[2].detach().clone()
        routed_loss(layer, [0, 1]).backward()
        optimizer.step()
        torch.testing.assert_close(packed[2], first_visit, rtol=0, atol=0)
        assert optimizer.state[packed]["step"] == 1
        assert torch.count_nonzero(optimizer.state[packed]["exp_avg"][0]) > 0
        before_inactive = packed[0].detach().clone()
        optimizer.zero_grad(set_to_none=True)
        routed_loss(layer, [1, 2]).backward()
        assert torch.count_nonzero(packed.grad[0]) == 0
        assert packed.grad[2].abs().sum() > 0
        optimizer.step()
        assert optimizer.state[packed]["step"] == 2
        assert (packed[2] - first_visit).abs().max() > 0
        # A zero data gradient also permits an update from existing momentum.
        assert (packed[0] - before_inactive).abs().max() > 1e-6


class TestCortexContinualTraining:
    @pytest.fixture(autouse=True)
    def deterministic_cpu(self):
        with device_runtime("cpu"):
            yield

    @pytest.mark.parametrize("kind,mode", DEVICE_CASES)
    @pytest.mark.parametrize("init", ["random", "sample"])
    @pytest.mark.parametrize("cumulative", [False, True], ids=["isolated", "cumulative"])
    def test_three_task_training_freezes_old_adapters_and_survives_reload(
        self, kind, mode, init, cumulative, tmp_path
    ):
        clean = tiny_model(kind).eval()
        first_tokens = torch.tensor([[1, 4, 7, 3, 2], [1, 5, 8, 4, 2]])
        second_tokens = torch.tensor([[1, 9, 3, 6, 2], [1, 7, 6, 9, 2]])
        third_tokens = torch.tensor([[1, 11, 4, 8, 2], [1, 12, 9, 3, 2]])
        model = get_peft_model(deepcopy(clean), task_config(kind, mode, init, 3), adapter_name="task1")
        train_task(model, first_tokens, "task1")
        first_output = logits(model, first_tokens)
        second_initial = logits(model, second_tokens) if cumulative else logits(clean, second_tokens)
        model.add_adapter("task2", task_config(kind, mode, init, 5))
        activate_task(model, "task2", ["task1"], cumulative)
        torch.testing.assert_close(logits(model, second_tokens), second_initial, rtol=0, atol=0)
        train_task(model, second_tokens, "task2")
        second_output = logits(model, second_tokens)
        model.set_adapter("task1", inference_mode=True)
        torch.testing.assert_close(logits(model, first_tokens), first_output, rtol=0, atol=0)

        model.save_pretrained(tmp_path / "adapters", selected_adapters=["task1", "task2"], save_embedding_layers=False)
        loaded = PeftModel.from_pretrained(
            deepcopy(clean), tmp_path / "adapters/task1", adapter_name="task1", is_trainable=False
        )
        loaded.load_adapter(tmp_path / "adapters/task2", adapter_name="task2", is_trainable=True)
        activate_task(loaded, "task2", ["task1"], cumulative)
        torch.testing.assert_close(logits(loaded, second_tokens), second_output, rtol=0, atol=0)

        # A third task after checkpoint reload exercises more than two adapters.
        third_initial = logits(loaded, third_tokens) if cumulative else logits(clean, third_tokens)
        loaded.add_adapter("task3", task_config(kind, mode, init, 7))
        activate_task(loaded, "task3", ["task1", "task2"], cumulative)
        torch.testing.assert_close(logits(loaded, third_tokens), third_initial, rtol=0, atol=0)
        train_task(loaded, third_tokens, "task3")
        third_output = logits(loaded, third_tokens)
        loaded.set_adapter("task1", inference_mode=True)
        torch.testing.assert_close(logits(loaded, first_tokens), first_output, rtol=0, atol=0)
        activate_task(loaded, "task2", ["task1"], cumulative)
        torch.testing.assert_close(logits(loaded, second_tokens), second_output, rtol=0, atol=0)
        activate_task(loaded, "task3", ["task1", "task2"], cumulative)
        torch.testing.assert_close(logits(loaded, third_tokens), third_output, rtol=0, atol=0)

        if not cumulative:
            return
        names = ["task1", "task2", "task3"]
        layers = [module for module in loaded.modules() if isinstance(module, CortexLayer)]
        if any(layer.role in {"public", "routed_public"} and layer.public_scale is None for layer in layers):
            with pytest.raises(ValueError, match="routing weights.*constant"):
                loaded.merge_and_unload(adapter_names=names, safe_merge=True)
            torch.testing.assert_close(logits(loaded, third_tokens), third_output, rtol=0, atol=0)
            assert not any(layer.merged for layer in layers)
        else:
            native = loaded.merge_and_unload(adapter_names=names, safe_merge=True)
            assert not any(isinstance(module, CortexLayer) for module in native.modules())
            torch.testing.assert_close(logits(native, third_tokens), third_output, rtol=2e-4, atol=3e-5)
            assert_native_reload(native, third_tokens, third_output, tmp_path / "native", rtol=2e-4, atol=3e-5)

    @pytest.mark.parametrize("kind,mode", DEVICE_CASES)
    def test_merge_after_each_task_reinjects_into_frozen_widened_base(self, kind, mode):
        clean = tiny_model(kind).eval()
        if kind == "qwen3_moe" and mode in ("shared", ["shared", "routed"]):
            # Start with a normalized model, rather than changing a trained router
            # merely to make an otherwise unsupported public merge succeed.
            config = deepcopy(clean.config)
            config.norm_topk_prob = True
            clean = AutoModelForCausalLM.from_config(config).eval()
        tokens = torch.tensor([[1, 4, 7, 3, 2], [1, 5, 8, 4, 2]])
        first = get_peft_model(clean, task_config(kind, mode, "sample", 3), adapter_name="task1")
        train_task(first, tokens, "task1")
        first_output = logits(first, tokens)
        native = first.merge_and_unload(adapter_names=["task1"], safe_merge=True).eval()
        torch.testing.assert_close(logits(native, tokens), first_output, rtol=2e-4, atol=3e-5)
        frozen = [(parameter, parameter.detach().clone()) for parameter in native.parameters()]
        merged_first_output = logits(native, tokens)

        second = get_peft_model(native, task_config(kind, mode, "sample", 5), adapter_name="task2")
        torch.testing.assert_close(logits(second, tokens), merged_first_output, rtol=0, atol=0)
        assert all(not parameter.requires_grad for parameter, _ in frozen)
        train_task(second, tokens, "task2")
        for parameter, previous in frozen:
            torch.testing.assert_close(parameter, previous, rtol=0, atol=0)
        with second.disable_adapter():
            torch.testing.assert_close(logits(second, tokens), merged_first_output, rtol=0, atol=0)
        second_output = logits(second, tokens)
        merged = second.merge_and_unload(adapter_names=["task2"], safe_merge=True)
        torch.testing.assert_close(logits(merged, tokens), second_output, rtol=2e-4, atol=3e-5)


@pytest.fixture
def deterministic_cortex_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(previous)


@contextmanager
def _unchanged_adapter_model(model, pending_config):
    """Check that a rejected adapter operation preserves the existing model and config."""
    modules = dict(model.named_modules())
    parameters = dict(model.named_parameters())
    state = deepcopy(model.state_dict())
    trainable = {name: parameter.requires_grad for name, parameter in parameters.items()}
    gradients = {name: None if p.grad is None else p.grad.clone() for name, p in parameters.items()}
    configs = dict(model.peft_config)
    config_values = [deepcopy(config.to_dict()) for config in (*configs.values(), pending_config)]
    active = deepcopy(model.active_adapter)
    targets = model.base_model.targeted_module_names.copy()
    layer_states = {
        name: (layer.role, list(layer.active_adapters), list(layer.merged_adapters))
        for name, layer in modules.items()
        if isinstance(layer, CortexLayer)
    }

    yield

    assert dict(model.named_modules()) == modules
    current_parameters = dict(model.named_parameters())
    assert current_parameters.keys() == parameters.keys()
    assert all(current_parameters[name] is parameter for name, parameter in parameters.items())
    assert {name: p.requires_grad for name, p in current_parameters.items()} == trainable
    torch.testing.assert_close({name: p.grad for name, p in current_parameters.items()}, gradients, rtol=0, atol=0)
    torch.testing.assert_close(model.state_dict(), state, rtol=0, atol=0)
    assert model.peft_config.keys() == configs.keys()
    assert all(model.peft_config[name] is config for name, config in configs.items())
    assert [config.to_dict() for config in (*configs.values(), pending_config)] == config_values
    assert model.active_adapter == active
    assert model.base_model.targeted_module_names == targets
    assert {
        name: (layer.role, list(layer.active_adapters), list(layer.merged_adapters))
        for name, layer in model.named_modules()
        if isinstance(layer, CortexLayer)
    } == layer_states


@contextmanager
def _reject_parameter_assignment(projection, failure, monkeypatch):
    """Reject the first or every weight assignment through a setter or registration hook."""
    original_setattr = nn.Linear.__setattr__
    failed = False

    def reject(module, name, value):
        nonlocal failed
        if module is projection and name == "weight" and (failure != "once" or not failed):
            failed = True
            raise RuntimeError("injected parameter assignment failure")
        if failure != "registration_hook":
            return original_setattr(module, name, value)

    if failure != "registration_hook":
        with monkeypatch.context() as patch:
            patch.setattr(nn.Linear, "__setattr__", reject)
            yield
    else:
        handle = nn.modules.module.register_module_parameter_registration_hook(reject)
        try:
            yield
        finally:
            handle.remove()


@pytest.mark.usefixtures("deterministic_cortex_cpu")
class TestCortexStructuralMerge:
    @pytest.mark.parametrize("saved_target", ["model.layers.1.mlp", "model.layers.1"])
    @pytest.mark.parametrize("selection", ["active", "inactive", "original"])
    def test_export_pads_retained_modules_to_save_copy_and_reloads(self, saved_target, selection, tmp_path):
        base = tiny_model("llama").eval()
        model = get_peft_model(
            base, CortexConfig(k=3, target_modules=["model.layers.0.mlp"], modules_to_save=[saved_target])
        ).eval()
        if selection != "active":
            model.add_adapter(
                "other",
                CortexConfig(
                    k=5,
                    target_modules=["model.layers.0.mlp"],
                    modules_to_save=[saved_target] if selection == "inactive" else None,
                ),
            )
        activate_branches(model)
        wrapper = base.get_submodule(saved_target)
        selected_adapter = "default" if selection == "active" else "other"
        retained = wrapper.original_module if selection == "original" else wrapper.modules_to_save[selected_adapter]
        retained_ffn = retained if saved_target.endswith(".mlp") else retained.mlp
        discarded = wrapper.original_module if selection == "active" else wrapper.modules_to_save["default"]
        discarded_ffn = discarded if saved_target.endswith(".mlp") else discarded.mlp
        if selection != "original":
            assert retained_ffn.config is not base.config
        with torch.no_grad():
            retained_ffn.down_proj.weight.add_(0.1)
        tokens = torch.randint(3, 32, (2, 5))
        model.set_adapter(selected_adapter)
        expected = model(tokens).logits
        # Explicit export must select and pad the requested copy even when another adapter is active.
        model.set_adapter("default")
        adapter_names = None if selection == "active" else [selected_adapter]
        native = model.merge_and_unload(safe_merge=True, adapter_names=adapter_names)
        width = 27 if selection == "active" else 29
        assert native.model.layers[1].mlp is retained_ffn
        assert native.config.intermediate_size == width
        assert retained_ffn.config.intermediate_size == width
        assert [layer.mlp.down_proj.in_features for layer in native.model.layers] == [width, width]
        assert discarded_ffn.down_proj.in_features == 24
        torch.testing.assert_close(native(tokens).logits, expected)
        assert_native_reload(native, tokens, expected, tmp_path)

    def test_export_rejects_ambiguous_saved_module_selection_before_mutation(self):
        config = CortexConfig(k=3, target_modules=["model.layers.0.mlp"], modules_to_save=["model.layers.1.mlp"])
        model = get_peft_model(tiny_model("llama"), deepcopy(config)).eval()
        model.add_adapter("other", deepcopy(config))
        activate_branches(model)
        with _unchanged_adapter_model(model, config):
            with pytest.raises(ValueError, match="Only one adapter"):
                model.merge_and_unload(adapter_names=["default", "other"])

    def test_export_keeps_distinct_serialized_subconfigs(self):
        class Composite(nn.Module):
            def __init__(self):
                super().__init__()
                self.left = tiny_model("llama")
                right_config = deepcopy(self.left.config)
                right_config.intermediate_size = 40
                self.right = AutoModelForCausalLM.from_config(right_config)
                self.config = PretrainedConfig(left_config=self.left.config, right_config=self.right.config)

        base = Composite()
        model = get_peft_model(
            base,
            CortexConfig(k=3, target_modules=["left.model.layers.0.mlp"], modules_to_save=["left.model.layers.1.mlp"]),
        )
        native = model.merge_and_unload()
        assert native.config.left_config.intermediate_size == 27
        assert native.config.right_config.intermediate_size == 40
        assert [layer.mlp.down_proj.in_features for layer in native.left.model.layers] == [27, 27]
        assert [layer.mlp.down_proj.in_features for layer in native.right.model.layers] == [40, 40]

    @pytest.mark.parametrize("kind", ["llama", "gpt_oss"])
    def test_merged_state_follows_dtype_and_preserves_unmerge(self, kind):
        model = get_peft_model(tiny_model(kind), CortexConfig(k=3, adapter_bias=kind == "gpt_oss")).eval()
        activate_branches(model)
        if kind == "gpt_oss":
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if "cortex_" in name and "bias" in name:
                        parameter.normal_(std=0.1)
        reference = deepcopy(model).double()
        model.merge_adapter(safe_merge=True)
        model.double()
        model.unmerge_adapter()
        assert all(p.dtype == torch.float64 for p in model.parameters())
        torch.testing.assert_close(model.state_dict(), reference.state_dict(), rtol=0, atol=0)
        tokens = torch.randint(3, 32, (2, 5))
        torch.testing.assert_close(model(tokens).logits, reference(tokens).logits, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "dtypes",
        [(torch.bfloat16,) * 3, (torch.float32, torch.float64, torch.float32)],
        ids=["uniform-bf16", "mixed"],
    )
    def test_adapter_follows_each_native_projection_dtype(self, dtypes, tmp_path):
        class FFN(nn.Module):
            def __init__(self):
                super().__init__()
                # An unrelated parameter registered on the FFN precedes all its child parameters.
                self.extra = nn.Parameter(torch.ones((), dtype=torch.float64))
                self.gate_proj = nn.Linear(4, 8, dtype=dtypes[0])
                self.up_proj = nn.Linear(4, 8, dtype=dtypes[1])
                self.down_proj = nn.Linear(8, 4, dtype=dtypes[2])
                self.act_fn = nn.SiLU()

            def forward(self, x):
                gate = self.act_fn(self.gate_proj(x.to(self.gate_proj.weight.dtype)))
                up = self.up_proj(x.to(self.up_proj.weight.dtype))
                return self.down_proj((gate * up).to(self.down_proj.weight.dtype))

        base = nn.Sequential(FFN())
        clean = deepcopy(base)
        x = torch.randn(2, 4)
        baseline = clean(x)
        model = get_peft_model(
            base, CortexConfig(k=2, target_modules=["0"], init_weights="sample"), autocast_adapter_dtype=False
        )
        layer = base[0]
        for kind, dtype in zip(("gate", "up", "down"), dtypes):
            projection = getattr(layer, f"cortex_{kind}")["default"]
            assert projection.weight.dtype == projection.bias.dtype == dtype

        activate_branches(model)
        expected = model(x)
        assert not torch.equal(baseline, expected)
        expected.sum().backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

        model.save_pretrained(tmp_path, save_embedding_layers=False)
        loaded = PeftModel.from_pretrained(
            clean, tmp_path, adapter_name="restored", autocast_adapter_dtype=False, low_cpu_mem_usage=True
        )
        torch.testing.assert_close(loaded(x), expected, rtol=0, atol=0)
        merged = loaded.merge_and_unload(safe_merge=True)
        torch.testing.assert_close(merged(x), expected)

    @pytest.mark.parametrize("mixed", [False, True], ids=["bf16", "mixed"])
    @pytest.mark.parametrize("autocast", [True, False], ids=["promoted", "native"])
    def test_low_memory_load_preserves_checkpoint_precision(self, mixed, autocast, tmp_path):
        base = tiny_model("llama").to(torch.bfloat16)
        if mixed:
            for block in base.model.layers:
                block.mlp.gate_proj.float()
                block.mlp.up_proj.float()
        clean = deepcopy(base)
        model = get_peft_model(base, CortexConfig(k=2, target_modules=["mlp"]))
        # These FP32 checkpoint values cannot round-trip through BF16.
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.copy_(torch.linspace(1.001, 1.043, param.numel()).reshape_as(param))
        model.save_pretrained(tmp_path, save_embedding_layers=False)

        saved = get_peft_model_state_dict(model, save_embedding_layers=False)
        for low_memory in (False, True):
            loaded = PeftModel.from_pretrained(
                deepcopy(clean),
                tmp_path,
                adapter_name="restored",
                low_cpu_mem_usage=low_memory,
                autocast_adapter_dtype=autocast,
            )
            state = get_peft_model_state_dict(loaded, adapter_name="restored", save_embedding_layers=False)
            assert state.keys() == saved.keys()
            for name, param in saved.items():
                dtype = torch.float32 if autocast or (mixed and "cortex_down" not in name) else torch.bfloat16
                torch.testing.assert_close(state[name], param.to(dtype), rtol=0, atol=0)

    def test_preflight_does_not_construct_wide_ffns(self, monkeypatch):
        model = get_peft_model(tiny_model("llama"), CortexConfig(k=3))
        original = merge_utils.merged_components
        calls = []

        def counted(layer, names, safe_merge=False):
            calls.append(id(layer))
            return original(layer, names, safe_merge)

        monkeypatch.setattr(merge_utils, "merged_components", counted)
        model.merge_adapter(safe_merge=True)
        assert len(calls) == len(set(calls)) == 2

    @pytest.mark.parametrize("cause", ["base_nan", "cast_overflow", "bias_sum_overflow"])
    def test_safe_preflight_detects_nonfinite_results_before_any_layer_changes(self, cause):
        kind = "gpt_oss" if cause == "bias_sum_overflow" else "llama"
        model = get_peft_model(tiny_model(kind).half(), CortexConfig(k=3, adapter_bias=kind == "gpt_oss"))
        layers = [module for module in model.modules() if isinstance(module, CortexLayer)]
        with torch.no_grad():
            if cause == "base_nan":
                next(layers[-1].base_layer.parameters()).flatten()[0] = float("nan")
            elif cause == "cast_overflow":
                layers[-1].cortex_up["default"].weight.fill_(1e10)
            else:
                layers[-1].base_layer.down_proj_bias.fill_(60000)
                layers[-1].cortex_down["default"].bias.fill_(60000)
        parameters = list(model.parameters())
        with pytest.raises(ValueError, match="Non-finite"):
            model.merge_adapter(safe_merge=True)
        assert all(not layer.merged and layer._merge_state is None for layer in layers)
        assert all(p is before for p, before in zip(model.parameters(), parameters))

    def test_merge_allocation_failure_leaves_layer_unchanged(self, monkeypatch):
        model = get_peft_model(tiny_model("llama"), CortexConfig(k=3))
        layer = next(module for module in model.modules() if isinstance(module, CortexLayer))
        original = nn.Parameter
        parameters = list(layer.base_layer.parameters())
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated allocation failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(merge_utils.nn, "Parameter", fail_second)
        with pytest.raises(RuntimeError, match="simulated allocation failure"):
            layer.merge(safe_merge=True)
        assert all(p is before for p, before in zip(layer.base_layer.parameters(), parameters))
        assert layer.base_layer.up_proj.out_features == 24
        assert not layer.merged and layer._merge_state is None

    @pytest.mark.parametrize("mode", ["shared", "routed", ["shared", "routed"]])
    def test_unmerge_releases_expanded_storage_and_does_not_retain_original_weights(self, mode):
        model = get_peft_model(tiny_model("gpt_oss"), CortexConfig(k=3, moe_target=mode, adapter_bias=True))
        activate_branches(model)
        original = {name: value.clone() for name, value in model.state_dict().items()}
        layers = [module for module in model.modules() if isinstance(module, CortexLayer)]
        old_parameters = [weakref.ref(p) for layer in layers for p in layer.base_layer.parameters()]
        model.merge_adapter(safe_merge=True)
        gc.collect()
        assert all(reference() is None for reference in old_parameters)
        assert all(layer._original_down_bias.shape == (3, 16) for layer in layers)
        assert not any("_original_down_bias" in name for name in model.state_dict())
        wide_parameters = [weakref.ref(p) for layer in layers for p in layer.base_layer.parameters()]
        model.unmerge_adapter()
        gc.collect()
        assert all(reference() is None for reference in wide_parameters)
        torch.testing.assert_close(model.state_dict(), original, rtol=0, atol=0)
        for layer in layers:
            assert layer._merge_state is None and layer._original_down_bias is None
            for parameter in layer.base_layer.parameters():
                assert parameter.untyped_storage().nbytes() == parameter.numel() * parameter.element_size()

    def test_unmerge_restores_bfloat16_down_bias_without_subtraction_roundoff(self):
        model = get_peft_model(tiny_model("gpt_oss").to(torch.bfloat16), CortexConfig(k=3, adapter_bias=True))
        layers = [module for module in model.modules() if isinstance(module, CortexLayer)]
        with torch.no_grad():
            for layer in layers:
                layer.base_layer.down_proj_bias.fill_(1.0)
                layer.cortex_down["default"].bias.fill_(1 / 256)
        model.merge_adapter()
        model.unmerge_adapter()
        for layer in layers:
            torch.testing.assert_close(
                layer.base_layer.down_proj_bias, torch.ones_like(layer.base_layer.down_proj_bias), rtol=0, atol=0
            )

    @pytest.mark.parametrize(
        "operation,failure",
        [
            ("merge", "once"),
            ("merge", "setter"),
            ("merge", "registration_hook"),
            ("unmerge", "setter"),
            ("unmerge", "registration_hook"),
        ],
    )
    def test_assignment_failure_restores_layer(self, monkeypatch, operation, failure):
        model = get_peft_model(tiny_model("llama"), CortexConfig(k=3)).eval()
        activate_branches(model)
        layer = next(module for module in model.modules() if isinstance(module, CortexLayer))
        if operation == "unmerge":
            layer.merge(safe_merge=True)
        parameters = list(model.parameters())
        state = deepcopy(layer.state_dict())
        merge_state = layer._merge_state
        tokens = torch.randint(3, 32, (1, 5))
        expected = model(tokens).logits.detach()
        with _reject_parameter_assignment(layer.base_layer.down_proj, failure, monkeypatch):
            with pytest.raises(RuntimeError, match="injected parameter assignment failure"):
                if operation == "merge":
                    layer.merge(safe_merge=True)
                else:
                    layer.unmerge()
        assert all(actual is before for actual, before in zip(model.parameters(), parameters))
        width = 27 if operation == "unmerge" else 24
        assert layer.base_layer.up_proj.out_features == layer.base_layer.down_proj.in_features == width
        assert layer.merged_adapters == (["default"] if operation == "unmerge" else [])
        assert layer._merge_state is merge_state
        torch.testing.assert_close(layer.state_dict(), state, rtol=0, atol=0)
        torch.testing.assert_close(model(tokens).logits, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("operation", ["merge_adapter", "merge_and_unload", "unmerge_adapter"])
    def test_later_layer_failure_keeps_committed_ffns_and_checkpoint_can_restore(
        self, monkeypatch, operation, tmp_path
    ):
        clean = tiny_model("llama").eval()
        model = get_peft_model(deepcopy(clean), CortexConfig(k=3)).eval()
        activate_branches(model)
        tokens = torch.randint(3, 32, (1, 5))
        with torch.no_grad():
            expected = model(tokens).logits
        model.save_pretrained(tmp_path)
        if operation == "unmerge_adapter":
            model.merge_adapter()
        first, second = [module for module in model.modules() if isinstance(module, CortexLayer)]
        first_parameters = list(first.base_layer.parameters())
        second_parameters = list(second.parameters())
        second_state = deepcopy(second.state_dict())
        with _reject_parameter_assignment(second.base_layer.down_proj, "setter", monkeypatch):
            with pytest.raises(RuntimeError, match="injected parameter assignment failure"):
                getattr(model, operation)()
        first_is_merged = operation != "unmerge_adapter"
        assert first.merged == first_is_merged
        assert second.merged != first_is_merged
        assert first.base_layer.up_proj.out_features == (27 if first_is_merged else 24)
        assert second.base_layer.up_proj.out_features == (24 if first_is_merged else 27)
        assert all(actual is not before for actual, before in zip(first.base_layer.parameters(), first_parameters))
        assert all(actual is before for actual, before in zip(second.parameters(), second_parameters))
        torch.testing.assert_close(second.state_dict(), second_state, rtol=0, atol=0)
        assert model.get_base_model().model.layers[0].mlp is (
            first.base_layer if operation == "merge_and_unload" else first
        )
        assert model.get_base_model().model.layers[1].mlp is second
        # Do not continue training or exporting the partially committed instance.
        restored = PeftModel.from_pretrained(clean, tmp_path).eval()
        torch.testing.assert_close(restored(tokens).logits, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("operation", ["merge_adapter", "merge_and_unload", "unmerge_adapter"])
    def test_merge_operations_release_committed_ffn_parameters_before_next_layer(self, monkeypatch, operation):
        model = get_peft_model(tiny_model("llama"), CortexConfig(k=3)).eval()
        activate_branches(model)
        tokens = torch.randint(3, 32, (1, 5))
        with torch.no_grad():
            expected = model(tokens).logits
        if operation == "unmerge_adapter":
            model.merge_adapter()
        first, second = [module for module in model.modules() if isinstance(module, CortexLayer)]
        old_parameters = [weakref.ref(parameter) for parameter in first.base_layer.parameters()]
        original = merge_utils._apply_components
        checked = False

        def check_released(module, *args, **kwargs):
            nonlocal checked
            if module is second.base_layer:
                gc.collect()
                assert all(reference() is None for reference in old_parameters)
                checked = True
            return original(module, *args, **kwargs)

        monkeypatch.setattr(merge_utils, "_apply_components", check_released)
        result = getattr(model, operation)()
        assert checked
        result = result if operation == "merge_and_unload" else model
        torch.testing.assert_close(result(tokens).logits, expected)

    def test_export_failure_keeps_completed_updates_and_checkpoint_can_restore(self, monkeypatch, tmp_path):
        clean = tiny_model("llama").eval()
        base = deepcopy(clean)
        model = get_peft_model(
            base, CortexConfig(k=3, target_modules=["model.layers.0.mlp"], modules_to_save=["model.layers.1.mlp"])
        ).eval()
        activate_branches(model)
        retained = base.model.layers[1].mlp.modules_to_save["default"]
        with torch.no_grad():
            retained.down_proj.weight.add_(0.1)
        tokens = torch.randint(3, 32, (1, 5))
        with torch.no_grad():
            expected = model(tokens).logits
        model.save_pretrained(tmp_path)
        original_plan = model_utils.apply_export_plan

        def fail_after_export(plan):
            original_plan(plan)
            raise RuntimeError("failure after export updates")

        monkeypatch.setattr(model_utils, "apply_export_plan", fail_after_export)
        with pytest.raises(RuntimeError, match="failure after export updates"):
            model.merge_and_unload(safe_merge=True)
        assert base.config.intermediate_size == retained.config.intermediate_size == 27
        assert base.model.layers[1].mlp is retained
        assert [layer.mlp.down_proj.in_features for layer in base.model.layers] == [27, 27]
        assert not any(isinstance(module, CortexLayer) for module in base.modules())
        restored = PeftModel.from_pretrained(clean, tmp_path).eval()
        torch.testing.assert_close(restored(tokens).logits, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("failure", ["setter", "registration_hook"])
    def test_padding_assignment_failure_restores_bare_ffn(self, monkeypatch, failure):
        layer = tiny_model("llama").model.layers[0].mlp.eval()
        parameters = list(layer.parameters())
        state = deepcopy(layer.state_dict())
        inputs = torch.randn(2, 5, 16)
        with torch.no_grad():
            expected = layer(inputs)
        with _reject_parameter_assignment(layer.down_proj, failure, monkeypatch):
            with pytest.raises(RuntimeError, match="injected parameter assignment failure"):
                merge_utils.pad_width(layer, 27)
        assert all(actual is before for actual, before in zip(layer.parameters(), parameters))
        assert layer.up_proj.out_features == layer.down_proj.in_features == 24
        torch.testing.assert_close(layer.state_dict(), state, rtol=0, atol=0)
        torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=0)

    @pytest.mark.parametrize("operation", ["merge", "unmerge"])
    def test_merge_state_assignment_failure_restores_ffn_and_bias_buffer(self, monkeypatch, operation):
        model = get_peft_model(tiny_model("gpt_oss"), CortexConfig(k=3, adapter_bias=True)).eval()
        activate_branches(model)
        layer = next(module for module in model.modules() if isinstance(module, CortexLayer))
        if operation == "unmerge":
            layer.merge(safe_merge=True)
        parameters = list(layer.parameters())
        state = deepcopy(layer.state_dict())
        merge_state = layer._merge_state
        down_bias = layer._original_down_bias
        merged_adapters = layer.merged_adapters
        assert (down_bias is not None) == (operation == "unmerge")
        tokens = torch.randint(3, 32, (1, 5))
        with torch.no_grad():
            expected = model(tokens).logits
        original_setattr = CortexLayer.__setattr__
        saw_updated_weights_and_state = False

        def reject(self, name, value):
            nonlocal saw_updated_weights_and_state
            if self is layer and name == "merged_adapters":
                saw_updated_weights_and_state = (
                    merge_utils.describe_mlp(self.base_layer).intermediate_dim == (27 if operation == "merge" else 24)
                    and (self._original_down_bias is None) == (operation == "unmerge")
                    and (self._merge_state is None) == (operation == "unmerge")
                    and bool(value) == (operation == "merge")
                )
                raise RuntimeError("merge state assignment failure")
            return original_setattr(self, name, value)

        monkeypatch.setattr(CortexLayer, "__setattr__", reject)
        with pytest.raises(RuntimeError, match="merge state assignment failure"):
            getattr(layer, operation)()
        assert saw_updated_weights_and_state
        assert layer._merge_state is merge_state
        assert layer._original_down_bias is down_bias
        assert layer.merged_adapters is merged_adapters
        assert all(actual is before for actual, before in zip(layer.parameters(), parameters))
        torch.testing.assert_close(layer.state_dict(), state, rtol=0, atol=0)
        torch.testing.assert_close(model(tokens).logits, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("operation", ["merge", "unmerge"])
    @pytest.mark.parametrize("with_kwargs", [False, True])
    def test_failed_layer_merge_preserves_user_hook_handles(self, monkeypatch, operation, with_kwargs):
        model = get_peft_model(tiny_model("llama"), CortexConfig(k=3)).eval()
        activate_branches(model)
        layer = next(module for module in model.modules() if isinstance(module, CortexLayer))
        if operation == "unmerge":
            layer.merge()
        base = layer.base_layer
        calls = []

        def plain_hook(module, args):
            calls.append("plain")

        def kwargs_hook(module, args, kwargs):
            calls.append("kwargs")

        handle = base.register_forward_pre_hook(kwargs_hook if with_kwargs else plain_hook, with_kwargs=with_kwargs)
        hooks = base._forward_pre_hooks
        kwargs_flags = base._forward_pre_hooks_with_kwargs
        hook_contents, flag_contents = list(hooks.items()), list(kwargs_flags.items())
        parameters = list(layer.parameters())
        inputs = torch.randn(2, 5, 16)
        with torch.no_grad():
            expected = layer(inputs)
        with _reject_parameter_assignment(base.down_proj, "setter", monkeypatch):
            with pytest.raises(RuntimeError, match="injected parameter assignment failure"):
                getattr(layer, operation)()
        assert all(actual is before for actual, before in zip(layer.parameters(), parameters))
        assert base._forward_pre_hooks is hooks
        assert base._forward_pre_hooks_with_kwargs is kwargs_flags
        assert list(hooks.items()) == hook_contents and list(kwargs_flags.items()) == flag_contents
        calls.clear()
        torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=0)
        assert calls == (["kwargs"] if with_kwargs else ["plain"])
        calls.clear()
        handle.remove()
        assert handle.id not in hooks and handle.id not in kwargs_flags
        torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=0)
        assert not calls

    def test_unmerge_restores_only_dimension_metadata_changed_by_merge(self):
        class CustomFFN(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(4, 12, bias=False)
                self.up_proj = nn.Linear(4, 12, bias=False)
                self.down_proj = nn.Linear(12, 4, bias=False)
                self.act_fn = nn.SiLU()
                # Custom metadata need not be the tensor dimensions. Save its actual
                # original value when merge changes it instead of inferring a value.
                self.intermediate_dim, self.intermediate_size = 101, 102
                for index, projection in enumerate((self.gate_proj, self.up_proj, self.down_proj)):
                    projection.in_features = 200 + 2 * index
                    projection.out_features = 201 + 2 * index
                # Merge does not write these unrelated attributes.
                self.in_features = 301
                self.act_fn.intermediate_size = 302
                self.up_proj.intermediate_dim = 303

            def forward(self, inputs):
                return self.down_proj(self.act_fn(self.gate_proj(inputs)) * self.up_proj(inputs))

        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = CustomFFN()

            def forward(self, inputs):
                return self.mlp(inputs)

        base = CustomModel().eval()
        ffn = base.mlp
        projections = (ffn.gate_proj, ffn.up_proj, ffn.down_proj)
        original_dimensions = [(projection.in_features, projection.out_features) for projection in projections]
        original_weights = {name: parameter.detach().clone() for name, parameter in ffn.named_parameters()}
        model = get_peft_model(base, CortexConfig(k=3, target_modules=["mlp"], init_weights=False)).eval()
        inputs = torch.randn(2, 4)
        expected = model(inputs)
        model.merge_adapter(safe_merge=True)
        assert ffn.intermediate_dim == ffn.intermediate_size == 15
        assert ffn.down_proj.weight.shape == (4, 15)
        torch.testing.assert_close(model(inputs), expected)

        ffn.in_features = 401
        ffn.act_fn.intermediate_size = 402
        ffn.up_proj.intermediate_dim = 403
        model.unmerge_adapter()

        assert (ffn.intermediate_dim, ffn.intermediate_size) == (101, 102)
        assert [(projection.in_features, projection.out_features) for projection in projections] == original_dimensions
        for name, parameter in ffn.named_parameters():
            torch.testing.assert_close(parameter, original_weights[name], rtol=0, atol=0)
        torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
        assert ffn.in_features == 401
        assert ffn.act_fn.intermediate_size == 402
        assert ffn.up_proj.intermediate_dim == 403


@pytest.mark.usefixtures("deterministic_cortex_cpu")
class TestCortexTargetSelection:
    @pytest.mark.parametrize("kind", ["llama", "qwen3_moe"])
    def test_merged_target_rejected_before_injection_and_allows_retry(self, kind):
        model = get_peft_model(
            tiny_model(kind), CortexConfig(k=3, target_modules=["model.layers.1.mlp"], moe_target="routed")
        )
        activate_branches(model)
        model.merge_adapter()
        config = CortexConfig(k=3, moe_target="routed")

        # The unwrapped first layer is visited before the merged second layer.
        with _unchanged_adapter_model(model, config):
            with pytest.raises(ValueError, match="Unmerge before adding an adapter"):
                model.add_adapter("second", config)

        model.unmerge_adapter()
        model.add_adapter("second", config)
        layers = [module for module in model.modules() if isinstance(module, CortexLayer)]
        assert len(layers) == 2
        assert all("second" in layer.cortex_up for layer in layers)
        assert set(model.peft_config) == {"default", "second"}

    @pytest.mark.parametrize("kind", ["llama", "qwen3_moe"])
    def test_unselected_merged_target_does_not_prevent_injection(self, kind):
        model = get_peft_model(
            tiny_model(kind), CortexConfig(k=3, target_modules=["model.layers.1.mlp"], moe_target="routed")
        )
        model.merge_adapter()
        merged_layer = next(module for module in model.modules() if isinstance(module, CortexLayer))
        assert merged_layer.merged
        with pytest.warns(UserWarning, match="Unmerging the model first"):
            model.add_adapter("second", CortexConfig(k=3, target_modules=["model.layers.0.mlp"], moe_target="routed"))
        assert "default" in merged_layer.cortex_up
        assert "second" not in merged_layer.cortex_up
        assert any(isinstance(module, CortexLayer) and "second" in module.cortex_up for module in model.modules())

    def test_reused_config_cannot_silently_skip_new_layers(self):
        config = CortexConfig(k=3)
        first = get_peft_model(tiny_model("llama"), config)
        base = tiny_model("llama")
        base.config.num_hidden_layers = 3
        new_base = AutoModelForCausalLM.from_config(base.config)
        with pytest.raises(ValueError, match="Create a new CortexConfig"):
            get_peft_model(new_base, config)
        assert not any(isinstance(module, CortexLayer) for module in new_base.modules())
        assert first.peft_config["default"] is config
        assert len(first.peft_config["default"].target_roles) == 2
        second = get_peft_model(new_base, CortexConfig(k=3))
        assert len(second.peft_config["default"].target_roles) == 3
        first.add_adapter("second", config)
        assert first.peft_config["second"] is config
        assert len(config.target_roles) == 2

    @pytest.mark.parametrize("selection", [["routed", "shared"], ["shared", "routed"]])
    def test_moe_selection_normalizes_and_roundtrips(self, selection, tmp_path):
        config = CortexConfig(k=3, moe_target=selection)
        assert config.moe_target == ["shared", "routed"]
        model = get_peft_model(tiny_model("gpt_oss"), config)
        model.add_adapter("second", CortexConfig(k=2, moe_target=["shared", "routed"]))
        assert all(layer.role == "routed_public" for layer in model.modules() if isinstance(layer, CortexLayer))
        model.save_pretrained(tmp_path)
        saved = CortexConfig.from_pretrained(tmp_path)
        assert saved.moe_target == ["shared", "routed"]
        assert set(saved.target_roles.values()) == {"routed_public"}
        assert "expert_mode" not in saved.to_dict() and "resolved_targets" not in saved.to_dict()

    def test_default_moe_selection_is_shared(self):
        assert CortexConfig().moe_target == "shared"
        model = get_peft_model(tiny_model("gpt_oss"), CortexConfig(k=3))
        layers = [module for module in model.modules() if isinstance(module, CortexLayer)]
        assert layers and all(layer.role == "public" for layer in layers)
        assert all(not layer.per_expert for layer in layers)

    @pytest.mark.parametrize(
        "excluded,retained,role",
        [("experts", "shared_expert", "shared"), ("shared_expert", "experts", "routed")],
    )
    def test_resolved_moe_exclusion_survives_config_reuse_and_checkpoint(self, excluded, retained, role, tmp_path):
        base = BlockModel("qwen3_next").eval()
        clean = deepcopy(base)
        config = CortexConfig(
            k=3, target_modules=["mlp"], exclude_modules=[f"mlp.{excluded}"], moe_target=["shared", "routed"]
        )
        model = get_peft_model(base, config).eval()
        activate_branches(model)
        expected_roles = {f"mlp.{retained}": role}
        assert config.target_roles == expected_roles
        assert not isinstance(base.get_submodule(f"mlp.{excluded}"), CortexLayer)

        reused = get_peft_model(deepcopy(clean), config).eval()
        assert reused.peft_config["default"] is config
        assert config.target_roles == expected_roles
        assert not isinstance(reused.get_base_model().get_submodule(f"mlp.{excluded}"), CortexLayer)
        inputs = torch.randn(2, 5, 16)
        expected = model(inputs).detach()
        model.save_pretrained(tmp_path, save_embedding_layers=False)
        loaded_config = CortexConfig.from_pretrained(tmp_path)
        assert loaded_config.target_roles == expected_roles
        assert not hasattr(loaded_config, "_cortex_user_targets")
        loaded = PeftModel.from_pretrained(clean, tmp_path, config=loaded_config).eval()
        assert loaded.peft_config["default"].target_roles == expected_roles
        assert not isinstance(clean.get_submodule(f"mlp.{excluded}"), CortexLayer)
        torch.testing.assert_close(loaded(inputs), expected)

    def test_loaded_targets_stay_fixed_on_larger_model_and_conflicting_exclusion_fails(self, tmp_path):
        model = get_peft_model(
            tiny_model("llama"),
            CortexConfig(k=3, exclude_modules=["model.layers.1.mlp"]),
        )
        model.save_pretrained(tmp_path)
        enlarged_config = deepcopy(model.get_base_model().config)
        enlarged_config.num_hidden_layers = 3
        saved_config = CortexConfig.from_pretrained(tmp_path)
        expected_roles = {"model.layers.0.mlp": "dense"}
        assert saved_config.target_roles == expected_roles
        loaded = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_config(enlarged_config), tmp_path, config=saved_config
        )
        assert loaded.peft_config["default"].target_roles == expected_roles
        assert sum(isinstance(module, CortexLayer) for module in loaded.modules()) == 1
        assert not isinstance(loaded.get_base_model().model.layers[2].mlp, CortexLayer)

        conflicting_config = CortexConfig.from_pretrained(tmp_path)
        conflicting_config.exclude_modules = {"model.layers.0.mlp"}
        base = AutoModelForCausalLM.from_config(enlarged_config)
        parameters = list(base.parameters())
        with pytest.raises(ValueError, match="Resolved adapter targets do not match"):
            PeftModel.from_pretrained(base, tmp_path, config=conflicting_config)
        assert not any(isinstance(module, CortexLayer) for module in base.modules())
        assert all(actual is original for actual, original in zip(base.parameters(), parameters))
        assert conflicting_config.target_roles == expected_roles

    @pytest.mark.parametrize("kind", ["gpt_oss", "qwen3_moe"])
    @pytest.mark.parametrize("selection,expected_role", [("parent", "public"), ("leaf", "routed")])
    def test_explicit_expert_selection_and_saved_public_role(self, kind, selection, expected_role, tmp_path):
        base = tiny_model(kind).eval()
        clean = deepcopy(base)
        parent_path = "model.layers.0.mlp"
        expert_path = parent_path + ".experts"
        target = parent_path if selection == "parent" else expert_path
        model = get_peft_model(base, CortexConfig(k=3, target_modules=[target], moe_target="shared")).eval()
        expected_roles = {expert_path: expected_role}
        assert model.peft_config["default"].target_roles == expected_roles
        layer = base.get_submodule(expert_path)
        assert layer.role == expected_role
        expected_shape = (3, 3, 16) if expected_role == "routed" else (3, 16)
        assert tuple(layer.cortex_up["default"].weight.shape) == expected_shape
        per_branch_parameters = 166 if kind == "gpt_oss" else 144
        expected_parameters = per_branch_parameters * (3 if expected_role == "routed" else 1)
        assert (
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            == expected_parameters
        )
        activate_branches(model)
        tokens = torch.randint(3, 32, (2, 5))
        expected = model(tokens).logits.detach()
        assert not torch.allclose(expected, clean(tokens).logits)
        model.save_pretrained(tmp_path, save_embedding_layers=False)

        saved_config = CortexConfig.from_pretrained(tmp_path)
        assert saved_config.target_modules == {expert_path}
        assert saved_config.target_roles == expected_roles
        assert not hasattr(saved_config, "_cortex_user_targets")
        restored = PeftModel.from_pretrained(clean, tmp_path, config=saved_config).eval()
        assert restored.get_base_model().get_submodule(expert_path).role == expected_role
        assert restored.peft_config["default"].target_roles == expected_roles
        torch.testing.assert_close(restored(tokens).logits, expected)

    @pytest.mark.parametrize("kind", ["gpt_oss", "qwen3_moe"])
    @pytest.mark.parametrize("first_selection", ["parent", "leaf"])
    def test_conflicting_parent_and_expert_selections_fail_before_mutation(self, kind, first_selection):
        parent_path = "model.layers.1.mlp"
        expert_path = parent_path + ".experts"
        first_target, second_target = (
            (parent_path, expert_path) if first_selection == "parent" else (expert_path, parent_path)
        )
        config = CortexConfig(k=3, target_modules=[first_target], moe_target="shared")
        model = get_peft_model(tiny_model(kind), config).eval()
        activate_branches(model)
        # Layer zero is otherwise valid; resolving a later conflicting role must
        # fail before adding any part of the new adapter.
        second_config = CortexConfig(k=5, target_modules=["model.layers.0.mlp", second_target], moe_target="shared")
        tokens = torch.randint(3, 32, (2, 5))
        expected = model(tokens).logits.detach()

        with _unchanged_adapter_model(model, second_config):
            with pytest.raises(ValueError, match="role"):
                model.add_adapter("second", second_config)
        torch.testing.assert_close(model(tokens).logits, expected, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "kind,first_target,second_target,first_roles,second_roles",
        [
            pytest.param(
                "llama",
                "model.layers.0.mlp",
                "model.layers.0.mlp",
                {"model.layers.0.mlp": "dense"},
                {"model.layers.0.mlp": "dense"},
                id="dense-same-layer",
            ),
            pytest.param(
                "llama",
                "model.layers.0.mlp",
                "model.layers.1.mlp",
                {"model.layers.0.mlp": "dense"},
                {"model.layers.1.mlp": "dense"},
                id="dense-different-layers",
            ),
            pytest.param(
                "gpt_oss",
                "model.layers.0.mlp",
                "model.layers.1.mlp",
                {"model.layers.0.mlp.experts": "public"},
                {"model.layers.1.mlp.experts": "routed"},
                id="gpt-oss-different-layers",
            ),
            pytest.param(
                "qwen3_moe",
                "model.layers.0.mlp",
                "model.layers.1.mlp",
                {"model.layers.0.mlp.experts": "public"},
                {"model.layers.1.mlp.experts": "routed"},
                id="qwen3-moe-different-layers",
            ),
            pytest.param(
                "qwen3_next",
                "mlp",
                "mlp",
                {"mlp.shared_expert": "shared"},
                {"mlp.experts": "routed"},
                id="native-shared-and-routed",
            ),
        ],
    )
    def test_different_moe_selections_with_compatible_physical_roles(
        self, kind, first_target, second_target, first_roles, second_roles, tmp_path
    ):
        clean = (BlockModel(kind) if kind == "qwen3_next" else tiny_model(kind)).eval()
        model = get_peft_model(
            deepcopy(clean),
            CortexConfig(
                k=3,
                target_modules=[first_target],
                moe_target="shared",
                public_weighting="routing" if kind == "qwen3_moe" else "unweighted",
            ),
        ).eval()
        model.add_adapter("second", CortexConfig(k=5, target_modules=[second_target], moe_target="routed"))
        assert model.peft_config["default"].target_roles == first_roles
        assert model.peft_config["second"].target_roles == second_roles
        activate_branches(model)
        inputs = torch.randn(2, 5, 16) if kind == "qwen3_next" else torch.randint(3, 32, (2, 5))

        def output(current):
            result = current(inputs)
            return result if kind == "qwen3_next" else result.logits

        reference = output(clean).detach()
        expected = {}
        for adapter in ("default", "second"):
            model.set_adapter(adapter)
            model.zero_grad(set_to_none=True)
            result = output(model)
            expected[adapter] = result.detach()
            assert not torch.allclose(result, reference)
            result.square().mean().backward()
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            assert trainable
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable)
            assert any(parameter.grad.abs().sum() > 0 for parameter in trainable)
            assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)

        model.base_model.set_adapter(["default", "second"])
        expected_combined = output(model).detach()
        model.save_pretrained(tmp_path, save_embedding_layers=False)
        restored = PeftModel.from_pretrained(deepcopy(clean), tmp_path, is_trainable=True).eval()
        restored.load_adapter(tmp_path / "second", "second", is_trainable=True)
        assert restored.peft_config["default"].target_roles == first_roles
        assert restored.peft_config["second"].target_roles == second_roles
        assert restored.peft_config["default"].moe_target == "shared"
        assert restored.peft_config["second"].moe_target == "routed"
        for adapter in ("default", "second"):
            restored.set_adapter(adapter)
            torch.testing.assert_close(output(restored), expected[adapter])
        restored.base_model.set_adapter(["default", "second"])
        torch.testing.assert_close(output(restored), expected_combined)
        restored.merge_adapter(adapter_names=["default"], safe_merge=True)
        restored.zero_grad(set_to_none=True)
        partial = output(restored)
        torch.testing.assert_close(partial, expected_combined, rtol=2e-4, atol=3e-5)
        partial.square().mean().backward()
        remaining = [parameter for name, parameter in restored.named_parameters() if ".second." in name]
        assert remaining
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in remaining)
        assert any(parameter.grad.abs().sum() > 0 for parameter in remaining)
        restored.unmerge_adapter()
        torch.testing.assert_close(output(restored), expected_combined, rtol=0, atol=0)
        native = restored.merge_and_unload(safe_merge=True)
        assert not any(isinstance(module, CortexLayer) for module in native.modules())
        torch.testing.assert_close(output(native), expected_combined, rtol=2e-4, atol=3e-5)

    def test_invalid_targets_fail_before_replacement(self):
        model = tiny_model("qwen3")
        with pytest.raises(ValueError, match="Unsupported Cortex target"):
            get_peft_model(
                model, CortexConfig(k=3, moe_target="shared", task_type="CAUSAL_LM", target_modules=["mlp", "q_proj"])
            )
        assert not any(isinstance(m, CortexLayer) for m in model.modules())


@pytest.mark.usefixtures("deterministic_cortex_cpu")
class TestCortexAdapterOwnership:
    @pytest.mark.parametrize("saved_target", ["model.layers.0", "model.layers.0.mlp"])
    def test_auto_discovery_preserves_auxiliary_ownership_and_independent_checkpoints(self, saved_target, tmp_path):
        clean = tiny_model("llama").eval()
        model = get_peft_model(deepcopy(clean), CortexConfig(k=2, modules_to_save=[saved_target])).eval()
        wrapper = model.get_base_model().get_submodule(saved_target)
        saved_copy = wrapper.modules_to_save["default"]
        copied_ffn = saved_copy if saved_target.endswith(".mlp") else saved_copy.mlp
        with torch.no_grad():
            copied_ffn.down_proj.weight.add_(0.2)
        copy_parameters = list(saved_copy.parameters())
        copy_weights = deepcopy(saved_copy.state_dict())

        model.add_adapter("b", CortexConfig(k=3))
        assert model.peft_config["b"].target_roles == {"model.layers.1.mlp": "dense"}
        assert model.peft_config["b"].target_modules == {"model.layers.1.mlp"}
        assert not any(isinstance(module, CortexLayer) for module in wrapper.modules())
        assert all(actual is before for actual, before in zip(saved_copy.parameters(), copy_parameters))
        torch.testing.assert_close(saved_copy.state_dict(), copy_weights, rtol=0, atol=0)
        assert sum(parameter.numel() for name, parameter in model.named_parameters() if ".b." in name) == 144

        activate_branches(model)
        tokens = torch.randint(3, 32, (2, 5))
        expected = {}
        for adapter in ("default", "b"):
            model.set_adapter(adapter)
            expected[adapter] = model(tokens).logits.detach()
        assert not torch.allclose(expected["b"], clean(tokens).logits)
        assert not torch.allclose(expected["default"], expected["b"])
        model.zero_grad(set_to_none=True)
        model(tokens).logits.square().mean().backward()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        assert sum(parameter.numel() for parameter in trainable) == 144
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable)
        assert any(parameter.grad.abs().sum() > 0 for parameter in trainable)
        assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)

        model.save_pretrained(tmp_path / "b_only", selected_adapters=["b"], save_embedding_layers=False)
        model.save_pretrained(tmp_path / "first", selected_adapters=["default"], save_embedding_layers=False)
        locations = {"b": tmp_path / "b_only" / "b", "default": tmp_path / "first"}
        disk_weights = load_file(str(locations["b"] / "adapter_model.safetensors"))
        assert len(disk_weights) == 3
        assert sum(value.numel() for value in disk_weights.values()) == 144
        expected_weights = {
            name: get_peft_model_state_dict(model, adapter_name=name, save_embedding_layers=False)
            for name in ("default", "b")
        }
        torch.testing.assert_close(disk_weights, expected_weights["b"], rtol=0, atol=0)

        # Each checkpoint first loads independently; then load the other adapter.
        for first, second in (("b", "default"), ("default", "b")):
            fresh_base = deepcopy(clean)
            base_parameters = [(parameter, parameter.detach().clone()) for parameter in fresh_base.parameters()]
            restored = PeftModel.from_pretrained(fresh_base, locations[first], adapter_name=first).eval()
            torch.testing.assert_close(restored(tokens).logits, expected[first])
            for parameter, original in base_parameters:
                torch.testing.assert_close(parameter, original, rtol=0, atol=0)
            restored.load_adapter(locations[second], adapter_name=second)
            for adapter in ("default", "b"):
                restored.set_adapter(adapter)
                assert restored.peft_config[adapter].target_roles == {"model.layers.1.mlp": "dense"}
                restored_weights = get_peft_model_state_dict(
                    restored, adapter_name=adapter, save_embedding_layers=False
                )
                torch.testing.assert_close(restored_weights, expected_weights[adapter], rtol=0, atol=0)
                torch.testing.assert_close(restored(tokens).logits, expected[adapter])

    @pytest.mark.parametrize("selection", ["wrapper", "original", "copy", "regex", "saved_roles"])
    def test_explicit_auxiliary_targets_fail_before_adapter_mutation(self, selection, tmp_path):
        model = get_peft_model(tiny_model("llama"), CortexConfig(k=2, modules_to_save=["model.layers.1.mlp"])).eval()
        activate_branches(model)
        wrapper_path = "model.layers.1.mlp"
        invalid_path = (
            wrapper_path
            + {
                "wrapper": "",
                "original": ".original_module",
                "copy": ".modules_to_save.default",
                "regex": ".original_module",
                "saved_roles": ".modules_to_save.default",
            }[selection]
        )
        targets = (
            r"model\.layers\.(?:0\.mlp|1\.mlp\.original_module)"
            if selection == "regex"
            else ["model.layers.0.mlp", invalid_path]
        )
        config = CortexConfig(k=3, target_modules=targets)
        if selection == "saved_roles":
            config.target_roles = {"model.layers.0.mlp": "dense", invalid_path: "dense"}
            config.save_pretrained(tmp_path)
            config = CortexConfig.from_pretrained(tmp_path)
        tokens = torch.randint(3, 32, (2, 5))
        expected = model(tokens).logits.detach()

        with _unchanged_adapter_model(model, config):
            with pytest.raises(ValueError, match="auxiliary training wrapper"):
                model.add_adapter("b", config)
        torch.testing.assert_close(model(tokens).logits, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("exclusion", ["exclude_modules", "modules_to_save"])
    def test_current_adapter_exclusion_precedes_auxiliary_target_rejection(self, exclusion):
        wrapper_path = "model.layers.1.mlp"
        model = get_peft_model(tiny_model("llama"), CortexConfig(k=2, modules_to_save=[wrapper_path]))
        config = CortexConfig(
            k=3,
            target_modules=["model.layers.0.mlp", wrapper_path],
            **{exclusion: [wrapper_path]},
        )
        model.add_adapter("b", config)
        assert config.target_roles == {"model.layers.0.mlp": "dense"}
        wrapper = model.get_base_model().get_submodule(wrapper_path)
        assert isinstance(wrapper, ModulesToSaveWrapper)
        assert not any(isinstance(module, CortexLayer) for module in wrapper.modules())
        assert set(wrapper.modules_to_save) == ({"default", "b"} if exclusion == "modules_to_save" else {"default"})
        model.set_adapter("b")
        tokens = torch.randint(3, 32, (2, 5))
        assert torch.isfinite(model(tokens).logits).all()

    def test_auto_discovery_rejects_when_all_ffns_belong_to_auxiliary_wrapper(self):
        base = tiny_model("llama")
        base.model = ModulesToSaveWrapper(base.model, adapter_name="old")
        modules = list(base.modules())
        parameters = list(base.parameters())
        with pytest.raises(ValueError, match="FFN"):
            get_peft_model(base, CortexConfig(k=3))
        assert not any(isinstance(module, CortexLayer) for module in base.modules())
        assert len(list(base.modules())) == len(modules)
        assert all(actual is before for actual, before in zip(base.modules(), modules))
        assert all(actual is before for actual, before in zip(base.parameters(), parameters))

    @pytest.mark.parametrize("protected", ["experts", "shared_expert"])
    @pytest.mark.parametrize("selection", ["auto-both", "auto-retained", "explicit-retained"])
    def test_aux_moe_retained_branch_checkpoint_and_nonzero_output(self, protected, selection, tmp_path):
        clean, model, wrapper, retained, role = _protected_moe(protected)
        targets = None if selection.startswith("auto") else ["mlp"]
        mode = ["shared", "routed"] if selection == "auto-both" else role
        model.add_adapter("b", CortexConfig(k=3, target_modules=targets, moe_target=mode))
        assert model.peft_config["b"].target_roles == {f"mlp.{retained}": role}
        assert not any(isinstance(module, CortexLayer) for module in wrapper.modules())
        activate_branches(model)
        # Distinguish default's owned copy from the original base used by b alone.
        with torch.no_grad():
            next(wrapper.modules_to_save["default"].parameters()).add_(0.2)
        inputs = torch.randn(2, 5, 16)
        model.set_adapter("b")
        expected = model(inputs)
        with model.disable_adapter():
            baseline = model(inputs)
        assert not torch.allclose(expected, baseline)
        expected.square().mean().backward()
        parameters = [p for name, p in model.named_parameters() if "cortex_" in name and ".b." in name]
        assert parameters
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        assert all(p.grad.count_nonzero() > 0 for p in parameters)
        state = get_peft_model_state_dict(model, adapter_name="b")
        assert len(state) == 3
        assert all(f"mlp.{retained}.cortex_" in name for name in state)
        assert not any("original_module" in name or "modules_to_save" in name for name in state)
        model.save_pretrained(tmp_path, save_embedding_layers=False)
        loaded = PeftModel.from_pretrained(deepcopy(clean), tmp_path / "b", adapter_name="b").eval()
        torch.testing.assert_close(loaded(inputs), expected, rtol=0, atol=0)
        model.base_model.set_adapter(["default", "b"])
        cumulative = model(inputs).detach()
        loaded = PeftModel.from_pretrained(deepcopy(clean), tmp_path).eval()
        loaded.load_adapter(tmp_path / "b", adapter_name="b")
        loaded.base_model.set_adapter(["default", "b"])
        torch.testing.assert_close(loaded(inputs), cumulative, rtol=0, atol=0)
        native = loaded.merge_and_unload(safe_merge=True)
        torch.testing.assert_close(native(inputs), cumulative, rtol=2e-4, atol=3e-5)
        reloaded_native = BlockModel("qwen3_next", config=deepcopy(native.config)).eval()
        reloaded_native.load_state_dict(native.state_dict(), strict=True)
        torch.testing.assert_close(reloaded_native(inputs), cumulative, rtol=2e-4, atol=3e-5)

    @pytest.mark.parametrize("protected", ["experts", "shared_expert"])
    @pytest.mark.parametrize("selection", ["auto", "parent-protected", "parent-both"])
    def test_aux_moe_rejects_requested_protected_branch(self, protected, selection):
        _, model, wrapper, _, _ = _protected_moe(protected)
        protected_role = "routed" if protected == "experts" else "shared"
        config = CortexConfig(
            k=3,
            target_modules=None if selection == "auto" else ["mlp"],
            moe_target=["shared", "routed"] if selection == "parent-both" else protected_role,
        )
        message = None if selection == "auto" else "auxiliary training wrapper"
        with _unchanged_adapter_model(model, config):
            with pytest.raises(ValueError, match=message) as error:
                model.add_adapter("b", config)
        assert "older ModuleList" not in str(error.value)
        assert not any(isinstance(module, CortexLayer) for module in wrapper.modules())
        assert not any(".b." in name for name, _ in model.named_parameters())

    @pytest.mark.parametrize("exclusion", ["exclude_modules", "modules_to_save"])
    def test_aux_moe_current_config_exclusion_precedes_existing_expert_boundary(self, exclusion, tmp_path):
        clean, model, wrapper, retained, role = _protected_moe("experts")
        config = CortexConfig(
            k=3,
            target_modules=["mlp"],
            moe_target=["shared", "routed"],
            **{exclusion: ["mlp.experts"]},
        )
        model.add_adapter("b", config)
        assert config.target_roles == {f"mlp.{retained}": role}
        assert not any(isinstance(module, CortexLayer) for module in wrapper.modules())
        assert set(wrapper.modules_to_save) == ({"default", "b"} if exclusion == "modules_to_save" else {"default"})
        activate_branches(model)
        model.set_adapter("b")
        inputs = torch.randn(2, 5, 16)
        expected = model(inputs).detach()
        with model.disable_adapter():
            assert not torch.allclose(model(inputs), expected)
        model.save_pretrained(tmp_path, selected_adapters=["b"], save_embedding_layers=False)
        loaded = PeftModel.from_pretrained(clean, tmp_path / "b", adapter_name="b").eval()
        torch.testing.assert_close(loaded(inputs), expected, rtol=0, atol=0)

    @pytest.mark.parametrize("saved_target", ["model.layers.0", "model.layers.0.mlp"])
    def test_modules_to_save_cannot_enclose_existing_cortex_before_other_injection(self, saved_target):
        old_config = CortexConfig(k=3)
        model = get_peft_model(tiny_model("llama").eval(), old_config).eval()
        activate_branches(model)
        tokens = torch.randint(3, 32, (2, 5))
        result = model(tokens).logits
        expected = result.detach()
        result.square().mean().backward()
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())

        config = CortexConfig(k=2, target_modules=["model.layers.1.mlp"], modules_to_save=[saved_target])
        with _unchanged_adapter_model(model, config):
            with pytest.raises(ValueError, match="modules_to_save.*existing Cortex"):
                model.add_adapter("b", config)
        torch.testing.assert_close(model(tokens).logits, expected, rtol=0, atol=0)

    def test_modules_to_save_matches_module_names_without_matching_character_suffixes(self, tmp_path):
        clean, target = _module_name_boundary_model()
        model = get_peft_model(deepcopy(clean), CortexConfig(k=2, target_modules=[target])).eval()
        base = model.get_base_model()
        old_layer, original_mlp = base.foo_mlp, base.mlp
        model.add_adapter("b", CortexConfig(k=3, target_modules=[target], modules_to_save=["mlp"]))
        assert base.foo_mlp is old_layer
        assert {name for name, module in base.named_modules() if isinstance(module, ModulesToSaveWrapper)} == {"mlp"}
        assert base.mlp.original_module is original_mlp
        assert set(base.mlp.modules_to_save) == {"b"}
        assert not any(isinstance(module, CortexLayer) for module in base.mlp.modules())
        assert model.peft_config["b"].target_roles == {target: "dense"}
        activate_branches(model)
        with torch.no_grad():
            base.mlp.modules_to_save["b"].weight.add_(0.25)
            base.mlp.modules_to_save["b"].bias.add_(0.1)
        inputs = torch.randn(3, 4)
        baseline = clean(inputs)
        expected = {}
        for adapter in ("default", "b"):
            model.set_adapter(adapter)
            expected[adapter] = model(inputs).detach()
            assert not torch.allclose(expected[adapter], baseline)
        assert not torch.allclose(expected["default"], expected["b"])

        for adapter in ("default", "b"):
            expected_state = get_peft_model_state_dict(model, adapter_name=adapter, save_embedding_layers=False)
            folder = tmp_path / adapter
            model.save_pretrained(folder, selected_adapters=[adapter], save_embedding_layers=False)
            checkpoint = folder if adapter == "default" else folder / adapter
            loaded = PeftModel.from_pretrained(deepcopy(clean), checkpoint, adapter_name=adapter).eval()
            restored_state = get_peft_model_state_dict(loaded, adapter_name=adapter, save_embedding_layers=False)
            torch.testing.assert_close(restored_state, expected_state, rtol=0, atol=0)
            torch.testing.assert_close(loaded(inputs), expected[adapter], rtol=0, atol=0)

    @pytest.mark.parametrize("nested", [False, True], ids=["exact-ffn", "dotted-ancestor-suffix"])
    def test_modules_to_save_rejects_actual_cortex_owner_before_mutation(self, nested):
        clean, target = _module_name_boundary_model(nested)
        old_config = CortexConfig(k=2, target_modules=[target])
        model = get_peft_model(clean, old_config).eval()
        activate_branches(model)
        inputs = torch.randn(3, 4)
        result = model(inputs)
        expected = result.detach()
        result.square().sum().backward()
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())

        # The new adapter has a legitimate target which must not be modified before
        # discovering the forbidden saved module, matched exactly or after a dot.
        config = CortexConfig(k=3, target_modules=["other_ffn"], modules_to_save=["foo_mlp"])
        with _unchanged_adapter_model(model, config):
            with pytest.raises(ValueError, match="modules_to_save.*existing Cortex"):
                model.add_adapter("b", config)
        torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)


@pytest.mark.usefixtures("deterministic_cortex_cpu")
class TestCortexRuntimeChecks:
    def test_example_warmup_arguments(self, tmp_path):
        from transformers import TrainingArguments

        from examples.cortex.training_utils import add_warmup_arguments, validate_warmup, warmup_kwargs

        parser = argparse.ArgumentParser()
        add_warmup_arguments(parser)
        cases = [
            ([], 10),
            (["--warmup-ratio", "0.25"], 25),
            (["--warmup-steps", "7"], 7),
            (["--warmup-steps", "0"], 0),
        ]
        for argv, expected in cases:
            args = parser.parse_args(argv)
            validate_warmup(args, parser)
            kwargs = warmup_kwargs(args.warmup_ratio, args.warmup_steps)
            training_args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none", **kwargs)
            assert training_args.get_warmup_steps(100) == expected

        with pytest.raises(SystemExit) as error:
            parser.parse_args(["--warmup-ratio", "0.1", "--warmup-steps", "7"])
        assert error.value.code == 2

    @pytest.mark.parametrize("signature", ["reordered", "renamed_input", "keyword_only"])
    def test_incompatible_expert_routing_rejected_before_injection(self, monkeypatch, signature):
        base = tiny_model("qwen3_moe")
        experts = base.model.layers[1].mlp.experts
        add_hook_to_module(experts, AlignDevicesHook(execution_device="cpu", offload=False))

        def reordered(hidden_states, top_k_weights, top_k_index):
            raise AssertionError("Injection should only inspect the signature.")

        def renamed_input(x, top_k_index, top_k_weights):
            raise AssertionError("Injection should only inspect the signature.")

        def keyword_only(hidden_states, *, top_k_index, top_k_weights):
            raise AssertionError("Injection should only inspect the signature.")

        # Accelerate's outer forward still advertises the original signature.
        forwards = {"reordered": reordered, "renamed_input": renamed_input, "keyword_only": keyword_only}
        monkeypatch.setattr(experts, "_old_forward", forwards[signature])
        with pytest.raises(ValueError, match=r"model\.layers\.1\.mlp\.experts"):
            get_peft_model(
                base,
                CortexConfig(
                    k=2,
                    moe_target="routed",
                    moe_specs={
                        type(base.model.layers[1].mlp).__name__: CortexMoeSpec(
                            routing_indices="top_k_index", routing_weights="top_k_weights"
                        )
                    },
                ),
            )
        assert not any(isinstance(module, CortexLayer) for module in base.modules())

    @pytest.mark.parametrize(
        "kind,index_key,weight_key",
        [("qwen3_moe", "top_k_index", "top_k_weights"), ("gpt_oss", "router_indices", "routing_weights")],
    )
    def test_native_expert_routing_keywords(self, kind, index_key, weight_key):
        base = tiny_model(kind)
        add_hook_to_module(base.model.layers[0].mlp.experts, AlignDevicesHook(execution_device="cpu", offload=False))
        model = get_peft_model(base, CortexConfig(k=2, moe_target="routed")).eval()
        activate_branches(model)
        experts = base.model.layers[0].mlp.experts
        inputs = torch.randn(3, base.config.hidden_size)
        indices = torch.tensor([[0], [1], [2]])
        weights = torch.rand(3, 1)

        expected = experts(inputs, indices, weights)
        assert not torch.allclose(expected, experts.base_layer(inputs, indices, weights))
        torch.testing.assert_close(experts(inputs, **{weight_key: weights, index_key: indices}), expected)
        torch.testing.assert_close(experts(inputs, indices, **{weight_key: weights}), expected)

    @pytest.mark.parametrize("invalid", ["index_dtype", "weight_dtype", "mask_shape"])
    def test_malformed_routing_tensors_rejected(self, invalid):
        base = tiny_model("qwen3_moe")
        get_peft_model(base, CortexConfig(k=2, moe_target="routed"))
        experts = base.model.layers[0].mlp.experts
        inputs = torch.randn(3, base.config.hidden_size)
        indices = torch.tensor([[0], [1], [2]])
        weights = torch.ones(3, 1)
        if invalid == "index_dtype":
            indices = indices.float()
        elif invalid == "weight_dtype":
            weights = weights.long()
        else:
            indices = F.one_hot(indices, num_classes=base.config.num_experts)
        with pytest.raises(ValueError, match="routing"):
            experts(inputs, indices, weights)

    @pytest.mark.parametrize("sequential_hook", [False, True])
    def test_offload_injection_is_rejected_before_mutation(self, sequential_hook):
        base = tiny_model("llama")
        cpu_offload(base, execution_device=torch.device("cpu"))
        if sequential_hook:
            add_hook_to_module(base.model.layers[0].mlp.up_proj, ModelHook(), append=True)
        parameters = list(base.parameters())
        with pytest.raises(ValueError, match="active Accelerate CPU/disk offload"):
            get_peft_model(base, CortexConfig(k=3))
        assert not any(isinstance(module, CortexLayer) for module in base.modules())
        assert all(p is before for p, before in zip(base.parameters(), parameters))

    @pytest.mark.parametrize("operation", ["merge_adapter", "merge_and_unload", "unmerge_adapter"])
    def test_offload_added_after_injection_is_rejected(self, operation):
        model = get_peft_model(tiny_model("llama"), CortexConfig(k=3))
        if operation == "unmerge_adapter":
            model.merge_adapter()
        cpu_offload(model, execution_device=torch.device("cpu"))
        parameters = list(model.parameters())
        with pytest.raises(ValueError, match="active Accelerate CPU/disk offload"):
            getattr(model, operation)()
        assert all(p is before for p, before in zip(model.parameters(), parameters))

    def test_delete_last_adapter_removes_only_its_moe_parent_checks(self):
        base = tiny_model("qwen3_moe")
        parents = [layer.mlp for layer in base.model.layers]
        unrelated = parents[0].register_forward_pre_hook(lambda module, args: None)
        model = get_peft_model(base, CortexConfig(k=3, moe_target="routed"))
        model.add_adapter("second", CortexConfig(k=3, target_modules={"model.layers.1.mlp"}, moe_target="routed"))

        def counts():
            return [
                sum(hook is check_moe_forward_hook for hook in parent._forward_pre_hooks.values())
                for parent in parents
            ]

        assert counts() == [1, 1]
        model.delete_adapter("default")
        assert counts() == [0, 1]
        assert isinstance(parents[0].experts, CortexLayer)
        assert not parents[0].experts._all_available_adapter_names()
        model.delete_adapter("second")
        assert counts() == [0, 0]
        assert unrelated.id in parents[0]._forward_pre_hooks
        model.add_adapter("again", CortexConfig(k=3, moe_target="routed"))
        assert counts() == [1, 1]
        model.add_adapter("extra", CortexConfig(k=3, moe_target="routed"))
        assert counts() == [1, 1]
        model.delete_adapter("extra")
        assert counts() == [1, 1]
        model.delete_adapter("again")
        assert counts() == [0, 0]
        unrelated.remove()

    @pytest.mark.parametrize("default_device", ["cpu", "mps"])
    def test_cpu_residency_and_meta_checkpoint_loading_remain_supported(self, tmp_path, monkeypatch, default_device):
        # An accelerator inferred by PEFT must not override this explicitly CPU-resident load.
        monkeypatch.setattr("peft.peft_model.infer_device", lambda: default_device)
        model = get_peft_model(tiny_model("llama"), CortexConfig(k=3))
        model.save_pretrained(tmp_path)
        base = tiny_model("llama")
        base.hf_device_map = {"": "cpu"}
        loaded = PeftModel.from_pretrained(
            base, tmp_path, low_cpu_mem_usage=True, device_map={"": "cpu"}, torch_device="cpu"
        )
        assert all(p.device.type == "cpu" for p in loaded.parameters())
        loaded.merge_and_unload(safe_merge=True)

    @pytest.mark.parametrize("when", ["before", "after"])
    def test_replaced_parent_kernel_rejected_before_bypass(self, when):
        base = tiny_model("gpt_oss").eval()
        parent = base.model.layers[0].mlp
        if when == "before":
            parent.forward = MethodType(bypassing_kernel, parent)
            with pytest.raises(ValueError, match="replaced MoE parent forward"):
                get_peft_model(base, CortexConfig(k=3))
            assert not any(isinstance(module, CortexLayer) for module in base.modules())
        else:
            model = get_peft_model(base, CortexConfig(k=3)).eval()
            parent.forward = MethodType(bypassing_kernel, parent)
            with pytest.raises(ValueError, match="replaced MoE parent forward"):
                model(torch.randint(3, 32, (1, 5)))
            with pytest.raises(ValueError, match="replaced MoE parent forward"):
                model.merge_and_unload()

    def test_kernel_annotation_native_fallback_and_unrelated_kernels_are_allowed(self, monkeypatch):
        base = tiny_model("gpt_oss").eval()
        parent = base.model.layers[0].mlp
        monkeypatch.setattr(type(parent), "kernel_layer_name", "MegaBlocksMoeMLP", raising=False)
        parent.forward = MethodType(type(parent).forward, parent)
        # A norm kernel is unrelated to the expert wrapper boundary.
        norm = base.model.layers[0].input_layernorm
        original = type(norm).forward
        norm.forward = MethodType(lambda self, x: original(self, x), norm)
        model = get_peft_model(base, CortexConfig(k=3)).eval()
        tokens = torch.randint(3, 32, (1, 5))
        expected = model(tokens).logits
        native = model.merge_and_unload(safe_merge=True)
        torch.testing.assert_close(native(tokens).logits, expected)
        assert not any(check_moe_forward_hook in module._forward_pre_hooks.values() for module in native.modules())

    @pytest.mark.parametrize("missing", ["is_transposed", "is_concatenated", "limit", "alpha"])
    def test_incomplete_packed_layout_fails_clearly(self, missing, monkeypatch):
        base = tiny_model("gpt_oss")
        experts = base.model.layers[0].mlp.experts
        monkeypatch.setattr(experts, missing, None)
        with pytest.raises(ValueError, match="metadata|clipping limit|SwiGLU alpha"):
            get_peft_model(base, CortexConfig(k=3))
        assert not any(isinstance(module, CortexLayer) for module in base.modules())

    def test_legacy_modulelist_experts_fail_clearly(self):
        base = tiny_model("qwen3_moe")
        base.model.layers[0].mlp.experts = nn.ModuleList([nn.Linear(16, 16)])
        with pytest.raises(ValueError, match="older ModuleList experts"):
            get_peft_model(base, CortexConfig(k=3))

    def test_routed_unbind_once_preserves_input_router_and_parameter_gradients(self, monkeypatch):
        model = get_peft_model(
            tiny_model("gpt_oss"), CortexConfig(k=3, moe_target="routed", adapter_bias=True, init_weights=False)
        )
        layer = next(module for module in model.modules() if isinstance(module, CortexLayer))
        reference = deepcopy(layer)
        inputs = torch.randn(5, 16, requires_grad=True)
        indices = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2], [2, 1]])
        weights = torch.rand(5, 2, requires_grad=True)
        params = [p for p in layer.parameters() if p.requires_grad]
        calls = []
        original = ExpertLinear.expert_views

        def counted(self):
            calls.append(id(self))
            return original(self)

        monkeypatch.setattr(ExpertLinear, "expert_views", counted)
        actual = layer(inputs, indices, weights)
        actual_grads = torch.autograd.grad(actual.square().sum(), [inputs, weights, *params])
        assert len(calls) == len(set(calls)) == 3

        def selecting(self, x, expert, views=None):
            return F.linear(x, self.weight[expert], None if self.bias is None else self.bias[expert])

        monkeypatch.setattr(ExpertLinear, "forward", selecting)
        expected = reference(inputs, indices, weights)
        reference_params = [p for p in reference.parameters() if p.requires_grad]
        expected_grads = torch.autograd.grad(expected.square().sum(), [inputs, weights, *reference_params])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    @pytest.mark.parametrize("mode", ["shared", "routed"])
    @pytest.mark.parametrize("order", ["native", "before", "after", "sequential"])
    def test_native_and_accelerate_paths_preserve_nonzero_adapter_and_gradients(self, mode, order):
        base = tiny_model("gpt_oss").eval()
        parent = base.model.layers[0].mlp
        if order == "before":
            add_hook_to_module(parent, AlignDevicesHook(execution_device="cpu", offload=False))
        model = get_peft_model(base, CortexConfig(k=3, moe_target=mode)).eval()
        activate_branches(model)
        if order in ("after", "sequential"):
            add_hook_to_module(parent, AlignDevicesHook(execution_device="cpu", offload=False))
        if order == "sequential":
            add_hook_to_module(parent, AlignDevicesHook(execution_device="cpu", offload=False), append=True)
        reference = deepcopy(model)
        remove_hook_from_module(reference.get_base_model().model.layers[0].mlp)
        tokens = torch.randint(3, 32, (2, 5))
        actual, expected = model(tokens).logits, reference(tokens).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with model.disable_adapter():
            baseline = model(tokens).logits
        assert not torch.allclose(actual, baseline)
        actual.square().mean().backward()
        expected.square().mean().backward()
        for (name, parameter), (_, other) in zip(model.named_parameters(), reference.named_parameters()):
            if "cortex_" in name:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                assert parameter.grad.count_nonzero() > 0
                torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
            else:
                assert parameter.grad is None
        native = model.merge_and_unload(safe_merge=True)
        torch.testing.assert_close(native(tokens).logits, expected, rtol=2e-4, atol=3e-5)

    @pytest.mark.parametrize("where", ["outer", "inner"])
    @pytest.mark.parametrize("when", ["before", "after"])
    def test_accelerate_metadata_does_not_hide_bypassing_forward(self, where, when):
        base = tiny_model("gpt_oss")
        parent = base.model.layers[0].mlp
        add_hook_to_module(parent, AlignDevicesHook(execution_device="cpu", offload=False))
        if when == "after":
            model = get_peft_model(base, CortexConfig(k=3))

        @wraps(type(parent).forward)
        def replacement(self, *args, **kwargs):
            raise AssertionError("The bypassing replacement must not run.")

        setattr(parent, "forward" if where == "outer" else "_old_forward", MethodType(replacement, parent))
        with pytest.raises(ValueError, match="replaced MoE parent forward"):
            if when == "before":
                get_peft_model(base, CortexConfig(k=3))
            else:
                model(torch.randint(3, 32, (1, 5)))

    def test_class_forward_replacement_after_injection_is_rejected(self, monkeypatch):
        base = tiny_model("gpt_oss")
        model = get_peft_model(base, CortexConfig(k=3))
        monkeypatch.setattr(type(base.model.layers[0].mlp), "forward", bypassing_kernel)
        with pytest.raises(ValueError, match="replaced MoE parent forward"):
            model(torch.randint(3, 32, (1, 5)))

    @pytest.mark.parametrize("observe_native_first", [False, True])
    def test_class_forward_check_depends_on_first_observation(self, monkeypatch, observe_native_first):
        # Isolate the cache so earlier tests cannot supply the native reference.
        monkeypatch.setattr(runtime_utils, "_native_forward", lru_cache()(runtime_utils._native_forward.__wrapped__))
        base = tiny_model("gpt_oss")
        parent = base.model.layers[0].mlp
        if observe_native_first:
            runtime_utils.check_moe_forward(parent)
        monkeypatch.setattr(type(parent), "forward", bypassing_kernel)
        if observe_native_first:
            with pytest.raises(ValueError, match="replaced MoE parent forward"):
                get_peft_model(base, CortexConfig(k=3))
            assert not any(isinstance(module, CortexLayer) for module in base.modules())
        else:
            # Earlier class patches are outside this guard's detection boundary;
            # successful injection does not make the replaced forward supported.
            model = get_peft_model(base, CortexConfig(k=3))
            with pytest.raises(AssertionError, match="The bypassing parent kernel must not execute"):
                model(torch.randint(3, 32, (1, 5)))

    def test_post_expert_norm_is_rejected_before_injection(self):
        base = tiny_model("gpt_oss")
        experts = base.model.layers[0].mlp.experts
        experts.has_post_expert_norm = True
        experts.post_expert_norm = nn.LayerNorm(base.config.hidden_size)
        with pytest.raises(ValueError, match="normalization inside individual expert outputs"):
            get_peft_model(base, CortexConfig(k=3))
        assert not any(isinstance(module, CortexLayer) for module in base.modules())


class TestCortexPublicRoutingConfiguration:
    @pytest.mark.parametrize("kind", ["olmoe", "qwen3_moe"])
    @pytest.mark.parametrize("mode", ["shared", ["shared", "routed"]])
    def test_old_public_checkpoint_missing_option_preserves_unweighted_behavior(self, kind, mode, tmp_path):
        with device_runtime("cpu"):
            base = tiny_model(kind).eval()
            clean = deepcopy(base)
            model = get_peft_model(base, public_config(mode, "unweighted")).eval()
            activate_branches(model)
            tokens = torch.randint(3, 32, (1, 5))
            expected = logits(model, tokens)
            model.save_pretrained(tmp_path, save_embedding_layers=False)
            path = tmp_path / "adapter_config.json"
            config = json.loads(path.read_text())
            del config["public_weighting"]
            path.write_text(json.dumps(config))
            assert PeftConfig.from_pretrained(tmp_path).public_weighting == "unweighted"
            loaded = PeftModel.from_pretrained(clean, tmp_path, adapter_name="old").eval()
            torch.testing.assert_close(logits(loaded, tokens), expected, rtol=0, atol=0)
            before = deepcopy(loaded.state_dict())
            with pytest.raises(ValueError, match="routing weights.*constant"):
                loaded.merge_and_unload(safe_merge=True)
            for name, value in loaded.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    @pytest.mark.parametrize("kind", ["olmoe", "qwen3_moe"])
    @pytest.mark.parametrize("mode", ["shared", ["shared", "routed"]])
    @pytest.mark.parametrize("previous_weighting", ["unweighted", "routing"])
    def test_public_routing_previous_task_freeze_mixed_checkpoint_and_partial_merge(
        self, kind, mode, previous_weighting, tmp_path
    ):
        with device_runtime("cpu"):
            base = tiny_model(kind).eval()
            clean = deepcopy(base)
            tokens = torch.randint(3, 32, (2, 6))
            model = get_peft_model(
                base, public_config(mode, previous_weighting, task_type="CAUSAL_LM"), adapter_name="task1"
            )
            train_task(model, tokens, "task1")
            first_output = logits(model, tokens)
            model.add_adapter("task2", public_config(mode, task_type="CAUSAL_LM"))
            activate_task(model, "task2", ["task1"], cumulative=True)
            train_task(model, tokens, "task2")
            joint_output = logits(model, tokens)
            model.set_adapter("task1")
            torch.testing.assert_close(logits(model, tokens), first_output, rtol=0, atol=0)
            activate_task(model, "task2", ["task1"], cumulative=True)
            model.save_pretrained(tmp_path, save_embedding_layers=False)
            loaded = PeftModel.from_pretrained(clean, tmp_path / "task1", adapter_name="first")
            loaded.load_adapter(tmp_path / "task2", adapter_name="second")
            loaded.base_model.set_adapter(["first", "second"])
            assert loaded.peft_config["first"].public_weighting == previous_weighting
            assert loaded.peft_config["second"].public_weighting == "routing"
            torch.testing.assert_close(logits(loaded, tokens), joint_output, rtol=0, atol=0)
            model.merge_adapter(adapter_names=["task2"], safe_merge=True)
            torch.testing.assert_close(logits(model, tokens), joint_output, rtol=2e-4, atol=3e-5)
            model.unmerge_adapter()
            torch.testing.assert_close(logits(model, tokens), joint_output, rtol=0, atol=0)
            if previous_weighting == "routing":
                native = model.merge_and_unload(adapter_names=["task1", "task2"], safe_merge=True)
                torch.testing.assert_close(logits(native, tokens), joint_output, rtol=2e-4, atol=3e-5)
            else:
                before = deepcopy(model.state_dict())
                with pytest.raises(ValueError, match="routing weights.*constant"):
                    model.merge_adapter(adapter_names=["task1", "task2"], safe_merge=True)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[name], rtol=0, atol=0)
                model.set_adapter("task2")
                second_output = logits(model, tokens)
                native = model.merge_and_unload(adapter_names=["task2"], safe_merge=True)
                torch.testing.assert_close(logits(native, tokens), second_output, rtol=2e-4, atol=3e-5)

    @pytest.mark.parametrize("kind", ["olmoe", "qwen3_moe"])
    @pytest.mark.parametrize("normalization", ["normalized", "all_experts"])
    def test_public_modes_agree_with_unit_routing_mass(self, kind, normalization):
        with device_runtime("cpu"):
            base = tiny_model(kind).eval()
            base.config.norm_topk_prob = normalization == "normalized"
            if normalization == "all_experts":
                base.config.num_experts_per_tok = base.config.num_experts
            base = AutoModelForCausalLM.from_config(base.config).eval()
            first = get_peft_model(deepcopy(base), public_config(["shared", "routed"], "unweighted")).eval()
            second = get_peft_model(deepcopy(base), public_config(["shared", "routed"])).eval()
            activate_branches(first)
            second.load_state_dict(first.state_dict(), strict=True)
            tokens = torch.randint(3, 32, (2, 5))
            torch.testing.assert_close(logits(first, tokens), logits(second, tokens), rtol=1e-5, atol=1e-6)
            first = first.merge_and_unload(safe_merge=True)
            second = second.merge_and_unload(safe_merge=True)
            for name, value in first.state_dict().items():
                torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)

    @pytest.mark.parametrize("kind", ["qwen3", "gemma3_text", "qwen3_next", "nemotron_h", "deepseek_v3"])
    def test_public_option_does_not_change_dense_native_shared_or_routed_branches(self, kind):
        with device_runtime("cpu"):
            dense = kind in {"qwen3", "gemma3_text"}
            base = tiny_model(kind) if dense else BlockModel(kind)
            first = get_peft_model(deepcopy(base), public_config(["shared", "routed"], "unweighted")).eval()
            second = get_peft_model(deepcopy(base), public_config(["shared", "routed"])).eval()
            assert all(
                m.role not in {"public", "routed_public"} for m in second.modules() if isinstance(m, CortexLayer)
            )
            activate_branches(first)
            second.load_state_dict(first.state_dict(), strict=True)
            inputs = torch.randint(3, 32, (2, 5)) if dense else torch.randn(2, 5, 16)
            output1, output2 = first(inputs), second(inputs)
            if dense:
                output1, output2 = output1.logits, output2.logits
            torch.testing.assert_close(output1, output2, rtol=0, atol=0)
            output1.square().mean().backward()
            output2.square().mean().backward()
            for (name, parameter), (other_name, other) in zip(first.named_parameters(), second.named_parameters()):
                assert name == other_name
                if parameter.grad is None:
                    assert other.grad is None
                else:
                    torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)

    @pytest.mark.parametrize("kind", ["olmoe", "qwen3_moe"])
    def test_delete_and_readd_adapter_removes_public_weighting_metadata(self, kind):
        with device_runtime("cpu"):
            model = get_peft_model(tiny_model(kind), public_config("shared"), adapter_name="task1")
            model.add_adapter("task2", public_config("shared", "unweighted"))
            model.delete_adapter("task2")
            for module in model.modules():
                if isinstance(module, CortexLayer):
                    assert module.public_weighting == {"task1": "routing"}
            model.add_adapter("task2", public_config("shared"))
            for module in model.modules():
                if isinstance(module, CortexLayer):
                    assert module.public_weighting == {"task1": "routing", "task2": "routing"}


class TestCortexPublicRouting(CortexPublicRoutingChecks):
    """Run the shared checks on CPU; GPU entries live in the GPU suite."""

    @pytest.fixture(params=["cpu"])
    def device(self, request):
        with device_runtime(request.param) as current:
            yield current

    @pytest.fixture(params=[torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
    def dtype(self, request, device):
        if device.type == "cuda" and request.param == torch.bfloat16:
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                pytest.skip("This CUDA GPU does not support native BF16.")
        return request.param


class TestCortexDevices(CortexDeviceChecks):
    """Run the shared checks on CPU; GPU entries live in the GPU suite."""

    @pytest.fixture(params=["cpu"])
    def device(self, request):
        with device_runtime(request.param) as current:
            yield current

    @pytest.fixture(params=[torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
    def dtype(self, request, device):
        if device.type == "cuda" and request.param == torch.bfloat16:
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                pytest.skip("This CUDA GPU does not support native BF16.")
        return request.param


@pytest.mark.usefixtures("deterministic_cortex_cpu")
class TestCortexDistributedCheckpoints:
    @pytest.mark.parametrize("mode", ["shared", "routed", ["shared", "routed"]])
    @pytest.mark.parametrize(
        "init,dtype,bias_init",
        [("random", dtype, bias) for dtype in (torch.float32, torch.bfloat16) for bias in ("uniform", "zeros")]
        + [("sample", torch.float32, "uniform")],
    )
    def test_distributed_initialization_check_accepts_configured_biases(self, mode, init, dtype, bias_init):
        base = tiny_model("gpt_oss").to(dtype)
        from .training.cortex_distributed import assert_initialization

        if init == "sample":
            with torch.no_grad():
                for name, parameter in base.named_parameters():
                    if "experts" in name and "bias" in name:
                        parameter.normal_()
        config = CortexConfig(
            k=4, moe_target=mode, init_weights=init, init_noise_std=0, adapter_bias=True, adapter_bias_init=bias_init
        )
        model = get_peft_model(base, config, autocast_adapter_dtype=False)
        assert_initialization(model, stage=None, config=config)
        layer = next(module for module in model.modules() if isinstance(module, CortexLayer))
        bias = layer.cortex_gate["default"].bias
        if init == "random" and bias_init == "uniform":
            assert torch.count_nonzero(bias) > 0
        for invalid in (0.5, float("nan")):
            with torch.no_grad():
                bias.fill_(invalid)
            with pytest.raises(AssertionError):
                assert_initialization(model, stage=None, config=config)

    def test_distributed_checkpoint_paths_support_deepspeed_globs(self, tmp_path):
        from .training.cortex_distributed import DISTRIBUTED_CASES, run_name

        assert DISTRIBUTED_CASES == [(kind, mode, "unweighted") for kind, mode in DEVICE_CASES] + [
            (kind, mode, "routing") for kind, mode in PUBLIC_ROUTING_CASES
        ]

        names = [
            run_name(kind, mode, init, dtype, weighting)
            for kind, mode, weighting in DISTRIBUTED_CASES
            for init in ("random", "sample")
            for dtype in ("float32", "bfloat16")
        ]
        assert names and len(names) == len(set(names))
        checkpoints = (
            ("mp_rank_00_model_states.pt", "mp_rank_*_model_states.pt"),
            ("zero_pp_rank_0_mp_rank_00_model_states.pt", "zero_pp_rank_0_mp_rank_*_model_states.pt"),
        )
        for name in names:
            folder = tmp_path / name / "deepspeed" / "step2"
            folder.mkdir(parents=True)
            for filename, pattern in checkpoints:
                checkpoint = folder / filename
                checkpoint.touch()
                # DeepSpeed globs the whole path; brackets in a list-valued mode
                # must not turn an existing checkpoint directory into a pattern.
                assert glob.glob(str(folder / pattern)) == [str(checkpoint)], name

    def test_zero3_construction_uses_existing_torch_process_group(self, tmp_path):
        # The worker exercises both Gemma3 and the validated packed GPT-OSS layout.
        pytest.importorskip("transformers", minversion="5.17.0")
        pytest.importorskip("deepspeed", minversion="0.18.9")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                "--module",
                "tests.training.cortex_deepspeed_bootstrap",
                "--output",
                str(tmp_path),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "DS_ACCELERATOR": "cpu", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
        reports = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
        assert all(report["status"] == "passed" and len(report["cases"]) == 2 for report in reports)


class TestCortexMoEFormula:
    @pytest.fixture(autouse=True)
    def deterministic_cpu(self):
        previous_threads = torch.get_num_threads()
        torch.manual_seed(123)
        torch.set_num_threads(1)
        yield
        torch.set_num_threads(previous_threads)

    @pytest.mark.parametrize("kind", ["qwen3_moe", "gpt_oss"])
    @pytest.mark.parametrize("mode", ["shared", "routed"])
    def test_moe_matches_explicit_mathematical_reference_and_gradients(self, kind, mode):
        model = tiny_model(kind)
        native_block = deepcopy(model.model.layers[0].mlp)
        adapted = get_peft_model(model, cortex_config(mode, adapter_bias=True))
        block = adapted.get_base_model().model.layers[0].mlp
        layer = block.experts

        # Large inputs exercise GPT-OSS's clipping, not just the near-zero regime.
        x = (torch.randn(5, 16) * 20).requires_grad_()
        indices = torch.tensor([[0, 2], [1, 2], [2, 0], [1, 0], [2, 1]])
        weights = torch.tensor([[0.2, 0.3], [0.7, 0.1], [0.4, 0.5], [0.6, 0.2], [0.1, 0.8]], requires_grad=True)
        activate_branches(layer)
        down = layer.cortex_down["default"]

        explicit_branch = partial(reference_branch, layer, kind)

        if mode == "shared":
            actual_output = block(x.unsqueeze(0))
            base_output = native_block(x.unsqueeze(0))
            actual = actual_output[0] if isinstance(actual_output, tuple) else actual_output
            base_value = base_output[0] if isinstance(base_output, tuple) else base_output
            if isinstance(base_output, tuple):
                torch.testing.assert_close(actual_output[1], base_output[1], rtol=0, atol=0)
            expected = base_value + explicit_branch(x).unsqueeze(0)
            parameters = (x, down.weight)
        else:
            actual = layer(x, indices, weights)
            correction = torch.stack(
                [sum(weights[t, j] * explicit_branch(x[t], int(indices[t, j])) for j in range(2)) for t in range(5)]
            )
            expected = native_block.experts(x, indices, weights) + correction
            parameters = (x, down.weight, weights)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        actual_grads = torch.autograd.grad(actual.square().mean(), parameters, retain_graph=True)
        expected_grads = torch.autograd.grad(expected.square().mean(), parameters)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=2e-4, rtol=2e-4)
