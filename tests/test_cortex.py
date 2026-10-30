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

import gc
import glob
import json
import os
import subprocess
import sys
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

from peft import CortexConfig, PeftConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from peft.tuners.cortex import merge as merge_utils
from peft.tuners.cortex import model as model_utils
from peft.tuners.cortex import runtime as runtime_utils
from peft.tuners.cortex.layer import CortexExperts, CortexLayer, CortexMLP, ExpertLinear
from peft.tuners.cortex.runtime import check_moe_forward_hook
from peft.tuners.cortex.structure import describe_mlp, is_moe, shared_names
from peft.utils.other import ModulesToSaveWrapper

from .testing_utils import ARCHITECTURES, BlockModel, activate_branches, configured_branch
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


class TestCortexNativeModels:
    @pytest.fixture(autouse=True)
    def deterministic_cpu(self):
        previous_threads = torch.get_num_threads()
        torch.manual_seed(123)
        torch.set_num_threads(1)
        yield
        torch.set_num_threads(previous_threads)

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
            ("gemma3_text", "routed"),
            ("qwen3_moe", "routed"),
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
        base = tiny_model(kind).eval()
        if kind == "gemma4_text":
            # Rebuild with a shared-KV, double-width MLP to test width coupling.
            base.config.num_kv_shared_layers = 1
            base.config.use_double_wide_mlp = True
            base.config.layer_types = ["full_attention", "full_attention"]
            base = AutoModelForCausalLM.from_config(base.config).eval()
        original_config = deepcopy(base.config.to_dict())
        original_state = deepcopy(base.state_dict())
        model = get_peft_model(base, CortexConfig(k=3, moe_target=mode, adapter_bias=kind == "gpt_oss")).eval()
        activate_branches(model)
        if kind == "gpt_oss":
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if "cortex_" in name and name.endswith("bias"):
                        parameter.normal_(std=0.1)
        tokens = torch.randint(3, 32, (2, 5))
        expected = model(tokens).logits.detach()
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

    @pytest.mark.parametrize("kind,init,width", [("qwen3", "random", 3), ("gemma3_text", "sample", 5)])
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

    def test_safe_merge_failure_is_atomic_and_bias_free_export_rejected(self):
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
    """Reject the first weight assignment or every assignment, including recovery attempts."""
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
    def test_export_pads_retained_modules_to_save_copy_and_reloads(self, saved_target, tmp_path):
        base = tiny_model("llama").eval()
        model = get_peft_model(
            base, CortexConfig(k=3, target_modules=["model.layers.0.mlp"], modules_to_save=[saved_target])
        ).eval()
        activate_branches(model)
        wrapper = base.get_submodule(saved_target)
        retained = wrapper.modules_to_save["default"]
        retained_ffn = retained if saved_target.endswith(".mlp") else retained.mlp
        original_ffn = wrapper.original_module if saved_target.endswith(".mlp") else wrapper.original_module.mlp
        assert retained_ffn.config is not base.config
        with torch.no_grad():
            retained_ffn.down_proj.weight.add_(0.1)
        tokens = torch.randint(3, 32, (2, 5))
        expected = model(tokens).logits
        native = model.merge_and_unload(safe_merge=True)
        assert native.model.layers[1].mlp is retained_ffn
        assert native.config.intermediate_size == 27
        assert retained_ffn.config.intermediate_size == 27
        assert [layer.mlp.down_proj.in_features for layer in native.model.layers] == [27, 27]
        assert original_ffn.down_proj.in_features == 24
        torch.testing.assert_close(native(tokens).logits, expected)
        assert_native_reload(native, tokens, expected, tmp_path)

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
        block = tiny_model(kind).model.layers[0].mlp
        spec = describe_mlp(block)
        config = cortex_config(mode, adapter_bias=True)
        if mode == "shared":
            layer = CortexMLP(block, "default", config, spec)
        else:
            layer = CortexExperts(block.experts, "default", config, spec, role="routed")
        # Large inputs exercise GPT-OSS's clipping, not just the near-zero regime.
        x = (torch.randn(5, 16) * 20).requires_grad_()
        indices = torch.tensor([[0, 2], [1, 2], [2, 0], [1, 0], [2, 1]])
        weights = torch.tensor([[0.2, 0.3], [0.7, 0.1], [0.4, 0.5], [0.6, 0.2], [0.1, 0.8]], requires_grad=True)
        activate_branches(layer)
        down = layer.cortex_down["default"]

        explicit_branch = partial(reference_branch, layer, kind)

        if mode == "shared":
            actual_output = layer(x.unsqueeze(0))
            base_output = block(x.unsqueeze(0))
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
            expected = block.experts(x, indices, weights) + correction
            parameters = (x, down.weight, weights)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        actual_grads = torch.autograd.grad(actual.square().mean(), parameters, retain_graph=True)
        expected_grads = torch.autograd.grad(expected.square().mean(), parameters)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=2e-4, rtol=2e-4)
