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
import os

import pytest
import torch

from peft import LoraConfig, TaskType, get_peft_model


if not os.getenv("PEFT_RUN_RWKV_TESTS"):
    pytest.skip("RWKV tests are disabled by default; set PEFT_RUN_RWKV_TESTS=1 to enable.", allow_module_level=True)

transformers = pytest.importorskip("transformers")


@pytest.mark.parametrize("seq_len", [4])
def test_rwkv_lora_forward_backward(seq_len: int):
    config = transformers.RwkvConfig(
        hidden_size=32,
        attention_hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        vocab_size=64,
        context_length=seq_len,
    )
    model = transformers.RwkvForCausalLM(config)

    lora_config = LoraConfig(r=4, lora_alpha=16, lora_dropout=0.0, task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)

    input_ids = torch.randint(0, config.vocab_size, (2, seq_len))
    output = model(input_ids=input_ids)
    loss = output.logits.float().mean()
    loss.backward()

    grads = [param.grad for name, param in model.named_parameters() if "lora_" in name and param.requires_grad]
    assert grads and all(g is not None for g in grads)
