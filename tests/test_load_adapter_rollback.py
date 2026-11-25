# Copyright 2023-present the HuggingFace Inc. team.
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

"""A failed adapter weight read must not leave a registered, selectable adapter behind."""

from pathlib import Path

import pytest
import torch
from safetensors import SafetensorError
from torch import nn

from peft import LoraConfig, get_peft_model


pytest.importorskip("accelerate")


def corrupt(path: Path) -> None:
    (path / "adapter_model.safetensors").write_bytes(b"bad")


def build(r: int = 8):
    torch.manual_seed(1)
    return get_peft_model(nn.Sequential(nn.Linear(4, 4)), LoraConfig(target_modules=["0"], r=r))


def test_failed_weight_read_registers_no_adapter(tmp_path):
    model = build()
    x = torch.ones(1, 4)
    expected = model(x).detach()
    model.save_pretrained(tmp_path)
    corrupt(tmp_path)

    with pytest.raises(SafetensorError):
        model.load_adapter(tmp_path, adapter_name="broken", torch_device="cpu", low_cpu_mem_usage=True)

    assert "broken" not in model.peft_config
    assert "broken" not in model.base_model.peft_config
    assert model.active_adapters == ["default"]
    assert not any(p.is_meta for n, p in model.named_parameters() if ".broken." in n)
    # the adapter that was already loaded still produces its own predictions
    assert torch.allclose(model(x), expected)


def test_retry_same_name_after_failure(tmp_path):
    """The first failure must not poison the name: retrying with a different rank should work."""
    model = build()
    model.save_pretrained(tmp_path)
    corrupt(tmp_path)

    with pytest.raises(SafetensorError):
        model.load_adapter(tmp_path, adapter_name="broken", torch_device="cpu", low_cpu_mem_usage=True)

    other = build(r=4)
    other.save_pretrained(tmp_path / "r4")
    model.load_adapter(tmp_path / "r4", adapter_name="broken", torch_device="cpu", low_cpu_mem_usage=True)

    assert "broken" in model.peft_config
    assert model.peft_config["broken"].r == 4
    assert not any(p.is_meta for n, p in model.named_parameters() if ".broken." in n)


def test_failed_reload_keeps_existing_adapter(tmp_path):
    """Rollback must only remove what this call added, never an adapter that was already there."""
    model = build()
    model.add_adapter("other", LoraConfig(target_modules=["0"], r=2))
    model.save_pretrained(tmp_path)
    before = list(model.peft_config)
    assert before == ["default", "other"]

    corrupt(tmp_path / "other")

    with pytest.raises(SafetensorError):
        model.load_adapter(tmp_path / "other", adapter_name="other", torch_device="cpu", low_cpu_mem_usage=True)

    assert list(model.peft_config) == before
    model.set_adapter("other")
    model(torch.ones(1, 4))
