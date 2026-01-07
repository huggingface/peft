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

import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from peft import PrefixTuningConfig, get_peft_model

from .testing_utils import hub_online_once


TINY_CAUSAL_LM = "trl-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture
def model_id():
    return TINY_CAUSAL_LM


@pytest.fixture
def base_model(model_id):
    with hub_online_once(model_id):
        return AutoModelForCausalLM.from_pretrained(model_id)


def test_prefix_tuning_offsets_position_ids_in_forward(monkeypatch, base_model):
    base = base_model
    peft_config = PrefixTuningConfig(num_virtual_tokens=4, task_type="CAUSAL_LM", prefix_projection=False)
    model = get_peft_model(base, peft_config)

    captured = {}

    def fake_forward(*args, **kwargs):
        captured["position_ids"] = kwargs.get("position_ids")
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        batch, seq_len = input_ids.shape
        logits = torch.zeros((batch, seq_len, base.config.vocab_size), device=input_ids.device)
        return CausalLMOutputWithPast(logits=logits)

    monkeypatch.setattr(model.base_model, "forward", fake_forward)

    input_ids = torch.randint(0, base.config.vocab_size, (1, 3))
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    _ = model(input_ids=input_ids, position_ids=position_ids)

    assert captured["position_ids"] is not None
    assert torch.equal(captured["position_ids"], position_ids + peft_config.num_virtual_tokens)
