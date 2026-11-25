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

import pytest
from torch import nn

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict


class FsdpNamingWrapper(nn.Module):
    # Mimics how FSDP wraps a model: `named_parameters()` gets the `_fsdp_wrapped_module` level, while the state dict
    # and its loading are forwarded to the wrapped module, which keeps the original keys.
    def __init__(self, module):
        super().__init__()
        self._fsdp_wrapped_module = module

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._fsdp_wrapped_module, name)

    def state_dict(self, *args, **kwargs):
        return self._fsdp_wrapped_module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self._fsdp_wrapped_module.load_state_dict(state_dict, strict=strict, assign=assign)


def test_load_adapter_into_wrapper_with_fsdp_names():
    # Regression test for #4223 that needs no distributed setup, so it also runs on Windows and macOS.
    config = LoraConfig(r=2, target_modules=["0", "1"])

    source = get_peft_model(nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 2)), config)
    for name, param in source.named_parameters():
        if "lora_B" in name:
            nn.init.ones_(param)

    target = FsdpNamingWrapper(get_peft_model(nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 2)), config))
    assert all(name.startswith("_fsdp_wrapped_module.") for name, _ in target.named_parameters())  # sanity check

    result = set_peft_model_state_dict(target, get_peft_model_state_dict(source))

    assert not result.unexpected_keys
    loaded_lora_b = sum(p.detach().abs().sum().item() for n, p in target.named_parameters() if "lora_B" in n)
    assert loaded_lora_b == pytest.approx(20.0)