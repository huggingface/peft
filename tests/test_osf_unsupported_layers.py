# Copyright 2026-present the HuggingFace Inc. team.
import re

import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM

from peft import OSFConfig, get_peft_model


def test_osf_raises_on_conv1d_target():
    """gpt2-style Conv1D targets must error instead of wrapping 0 parameters (#3986)."""
    model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    config = OSFConfig(target_modules=["c_attn", "c_proj"])
    msg = "is not supported. Currently, only the following modules are supported: `torch.nn.Linear`."
    with pytest.raises(ValueError, match=re.escape(msg)):
        get_peft_model(model, config)


def test_osf_raises_on_embedding_target():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(10, 8)

        def forward(self, x):
            return self.emb(x)

    model = Toy()
    config = OSFConfig(target_modules=["emb"])
    msg = "is not supported. Currently, only the following modules are supported: `torch.nn.Linear`."
    with pytest.raises(ValueError, match=re.escape(msg)):
        get_peft_model(model, config)


def test_osf_still_wraps_linear():
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin0 = nn.Linear(8, 8)

        def forward(self, x):
            return self.lin0(x)

    model = get_peft_model(MLP(), OSFConfig(target_modules=["lin0"], effective_rank=2))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable > 0
