# Copyright 2024-present the HuggingFace Inc. team.
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
from __future__ import annotations

import collections

import torch
from torch import nn

from peft import LoraConfig, get_peft_model
from peft.optimizers import create_loraplus_optimizer

from .testing_utils import torch_device


class SimpleNet(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.embedding = nn.Embedding(100, 20)
        self.layer_norm = nn.LayerNorm(20)
        self.lin0 = nn.Linear(20, 20, bias=bias)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(20, 16, bias=bias)

    def forward(self, X):
        X = self.lin0(self.layer_norm(self.embedding(X)))
        X = self.relu(X)
        X = self.lin1(X)
        return X


def test_lora_plus_helper_sucess():
    model = get_peft_model(SimpleNet(), LoraConfig(target_modules=["embedding", "lin0", "lin1"]))
    optimizer_cls = torch.optim.AdamW
    lr = 5e-5
    optim_config = {
        "eps": 1e-6,
        "betas": (0.9, 0.999),
        "loraplus_weight_decay": 0.0,
    }
    loraplus_lr_ratio = 1.2
    loraplus_lr_embedding = 1e-6
    optim = create_loraplus_optimizer(
        model=model,
        optimizer_cls=optimizer_cls,
        lr=lr,
        loraplus_lr_ratio=loraplus_lr_ratio,
        loraplus_lr_embedding=loraplus_lr_embedding,
        **optim_config,
    )
    assert optim is not None
    assert len(optim.param_groups) == 4
    assert optim.param_groups[0]["lr"] == lr
    assert optim.param_groups[1]["lr"] == loraplus_lr_embedding
    assert optim.param_groups[2]["lr"] == optim.param_groups[3]["lr"] == (lr * loraplus_lr_ratio)


def test_lora_plus_optimizer_sucess():
    """
    Test if the optimizer is correctly created and step function runs without any exception
    """
    optimizer_cls = torch.optim.AdamW
    optim_config = {
        "eps": 1e-6,
        "betas": (0.9, 0.999),
        "loraplus_weight_decay": 0.0,
    }
    model = get_peft_model(SimpleNet(), LoraConfig(target_modules=["embedding", "lin0", "lin1"])).to(torch_device)
    optim = create_loraplus_optimizer(
        model=model,
        optimizer_cls=optimizer_cls,
        lr=5e-5,
        loraplus_lr_ratio=1.2,
        loraplus_lr_embedding=1e-6,
        **optim_config,
    )
    loss = torch.nn.CrossEntropyLoss()
    x = torch.randint(100, (2, 4, 10)).to(torch_device)
    output = model(x).permute(0, 3, 1, 2)
    label = torch.randint(16, (2, 4, 10)).to(torch_device)
    loss_value = loss(output, label)
    loss_value.backward()
    optim.step()


def test_lora_plus_embedding_lr():
    # LoRA weights of embedding layers must land in the embedding param group and thus use
    # loraplus_lr_embedding, see #1915
    model = get_peft_model(SimpleNet(), LoraConfig(target_modules=["embedding", "lin0"]))
    lr = 5e-5
    loraplus_lr_ratio = 1.2
    loraplus_lr_embedding = 1e-6
    optim = create_loraplus_optimizer(
        model=model,
        optimizer_cls=torch.optim.AdamW,
        lr=lr,
        loraplus_lr_ratio=loraplus_lr_ratio,
        loraplus_lr_embedding=loraplus_lr_embedding,
    )

    param_to_name = {id(param): name for name, param in model.named_parameters()}
    # Map each learning rate to the names of the parameters trained with it. The groups themselves are unnamed and the
    # two groupB groups share a learning rate, hence the sets are merged per learning rate.
    group_names = collections.defaultdict(set)
    for group in optim.param_groups:
        group_names[group["lr"]].update(param_to_name[id(param)] for param in group["params"])
    assert group_names[loraplus_lr_embedding] == {
        "base_model.model.embedding.lora_embedding_A.default",
        "base_model.model.embedding.lora_embedding_B.default",
    }
    assert group_names[lr] == {"base_model.model.lin0.lora_A.default.weight"}
    assert group_names[lr * loraplus_lr_ratio] == {"base_model.model.lin0.lora_B.default.weight"}


def test_lora_plus_model_without_peft_wrapper():
    # top-level parameters like "embedding.weight" have no tuner layer to resolve, this must not raise
    model = SimpleNet()
    optim = create_loraplus_optimizer(model=model, optimizer_cls=torch.optim.AdamW, lr=5e-5, loraplus_lr_ratio=1.2)
    assert not optim.param_groups[1]["params"]
