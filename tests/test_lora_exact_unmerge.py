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
"""`temporarily_merge_adapter` restores LoRA base weights bit for bit, also in float16/bfloat16."""

import copy

import pytest
import torch
from torch import nn

from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(100, 64)
        self.lin = nn.Linear(64, 64)
        self.conv = nn.Conv2d(4, 8, 3)
        self.param_holder = nn.Linear(64, 32, bias=False)  # targeted by `target_parameters`


TARGETS = {"target_modules": ["emb", "lin", "conv"], "target_parameters": ["param_holder.weight"]}


def make_model(dtype, adapter_scale=1e-2, seed=0):
    torch.manual_seed(seed)
    base = Model()
    with torch.no_grad():
        for p in base.parameters():
            p.normal_(0, 0.02)
    model = get_peft_model(base.to(dtype), LoraConfig(r=8, lora_alpha=16, init_lora_weights=False, **TARGETS))
    scale_lora_b(model, adapter_scale)
    return model


def scale_lora_b(model, scale):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoraLayer):
                for params in (module.lora_B, module.lora_embedding_B):
                    for p in params.values():
                        (p.weight if isinstance(p, nn.Module) else p).normal_(0, scale)


def lora_layers(model):
    return [m for m in model.modules() if isinstance(m, LoraLayer)]


def base_weights(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" not in n and p.is_floating_point()}


def changed(before, after):
    return [name for name, tensor in before.items() if not torch.equal(tensor, after[name])]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("safe_merge", [False, True])
def test_round_trip_is_exact(dtype, safe_merge):
    # the adapter changes between round trips, like a training loop that syncs merged weights every step
    model = make_model(dtype)
    before = base_weights(model)
    for step in range(20):
        with model.temporarily_merge_adapter(safe_merge=safe_merge):
            pass
        assert not changed(before, base_weights(model))
        scale_lora_b(model, 1e-2 * (1 + step % 3))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_plain_merge_unmerge_drifts_and_records_nothing(dtype):
    # documents why the context manager exists, and that plain merges keep their memory footprint
    model = make_model(dtype)
    before = base_weights(model)
    model.merge_adapter()
    assert not any(layer._unmerge_residuals for layer in lora_layers(model))
    model.unmerge_adapter()
    assert changed(before, base_weights(model))


def test_float32_records_nothing():
    model = make_model(torch.float32)
    with model.temporarily_merge_adapter():
        assert not any(layer._unmerge_residuals for layer in lora_layers(model))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_merged_weights_unchanged(dtype):
    model = make_model(dtype)
    reference = copy.deepcopy(model)
    reference.merge_adapter()
    with model.temporarily_merge_adapter():
        assert not changed(base_weights(reference), base_weights(model))


def test_multiple_adapters():
    model = make_model(torch.bfloat16)
    model.add_adapter("other", LoraConfig(r=4, init_lora_weights=False, **TARGETS))
    scale_lora_b(model, 1e-2)
    before = base_weights(model)
    with model.temporarily_merge_adapter(adapter_names=["default", "other"]):
        pass
    assert not changed(before, base_weights(model))


def test_residuals_freed_and_flag_cleared_on_exit():
    model = make_model(torch.bfloat16)
    with model.temporarily_merge_adapter():
        assert any(layer._unmerge_residuals for layer in lora_layers(model))
    assert not any(layer._unmerge_residuals for layer in lora_layers(model))
    assert not any("_exact_unmerge" in layer.__dict__ for layer in lora_layers(model))


def test_exception_inside_context_still_restores():
    model = make_model(torch.bfloat16)
    before = base_weights(model)
    with pytest.raises(RuntimeError, match="boom"):
        with model.temporarily_merge_adapter():
            raise RuntimeError("boom")
    assert not changed(before, base_weights(model))
    assert not any(layer.merged for layer in lora_layers(model))


def test_requires_unmerged_on_entry():
    model = make_model(torch.bfloat16)
    model.merge_adapter()
    with pytest.raises(ValueError, match="unmerged"):
        with model.temporarily_merge_adapter():
            pass
