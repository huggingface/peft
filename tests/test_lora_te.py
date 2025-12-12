# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
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

import copy

import pytest
import torch
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from peft import LoraConfig, get_peft_model
from peft.import_utils import is_te_available

from .testing_utils import require_torch_gpu


pytestmark = pytest.mark.skipif(not is_te_available(), reason="transformer_engine is not available")


if is_te_available():
    from peft.tuners.lora.te import TELoRA


@pytest.fixture
def config():
    return AutoConfig.from_pretrained("nvidia/esm2_t6_8M_UR50D", trust_remote_code=True)


@pytest.fixture
def esm2_model(config):
    return AutoModelForTokenClassification.from_pretrained(
        "nvidia/esm2_t6_8M_UR50D", config=config, trust_remote_code=True, dtype="bfloat16"
    )


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("nvidia/esm2_t6_8M_UR50D")


@pytest.fixture
def tokenized_inputs(tokenizer):
    sequence = "MNEAAK"
    return tokenizer(sequence, return_tensors="pt")


@require_torch_gpu
def test_te_lora_wraps_te_linear_and_keeps_forward_working(config, esm2_model):
    cfg = LoraConfig(target_modules=["layernorm_qkv"], r=2, lora_alpha=8)

    lora_model = get_peft_model(esm2_model, cfg)
    lora_model = lora_model.to("cuda")

    wrapped_layernorm_qkv = lora_model.base_model.model.esm.encoder.layers[0].self_attention.layernorm_qkv

    assert isinstance(wrapped_layernorm_qkv, TELoRA)
    assert "default" in wrapped_layernorm_qkv.lora_A and "default" in wrapped_layernorm_qkv.lora_B
    assert wrapped_layernorm_qkv.get_base_layer().weight.requires_grad is False
    assert wrapped_layernorm_qkv.lora_A["default"].weight.requires_grad is True
    assert wrapped_layernorm_qkv.lora_B["default"].weight.requires_grad is True

    tokenizer = AutoTokenizer.from_pretrained("nvidia/esm2_t6_8M_UR50D")
    sequence = "MNEAAK"
    inputs = tokenizer(sequence, return_tensors="pt")

    inputs = {k: v.to(next(lora_model.parameters()).device) for k, v in inputs.items()}
    out = lora_model(**inputs)
    assert out.logits.shape == (1, len(sequence) + 2, config.num_labels)


@require_torch_gpu
def test_te_lora_forward_matches_base_before_backward(esm2_model, tokenized_inputs):
    dummy_model = copy.deepcopy(esm2_model).to("cuda")
    cfg = LoraConfig(target_modules=["layernorm_qkv"], r=4, lora_alpha=8)
    lora_model = get_peft_model(esm2_model, cfg).to("cuda")

    inputs = {k: v.to(next(lora_model.parameters()).device) for k, v in tokenized_inputs.items()}
    lora_model.eval()
    dummy_model.eval()
    with torch.no_grad():
        lora_result = lora_model(**inputs).logits
        dummy_result = dummy_model(**inputs).logits

    assert torch.allclose(lora_result, dummy_result, rtol=1e-3, atol=1e-3)


@require_torch_gpu
def test_te_lora_backward(esm2_model, tokenized_inputs):
    cfg = LoraConfig(target_modules=["layernorm_qkv"], r=4, lora_alpha=8)
    lora_model = get_peft_model(esm2_model, cfg).to("cuda")

    optimizer = torch.optim.AdamW(lora_model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    inputs = {k: v.to(next(lora_model.parameters()).device) for k, v in tokenized_inputs.items()}
    logits = lora_model(**inputs).logits
    labels = torch.randint(logits.shape[-1], logits.shape[:-1], device=logits.device)
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()
    optimizer.step()
