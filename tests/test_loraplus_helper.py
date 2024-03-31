import torch
from torch import nn
import bitsandbytes as bnb

from peft import LoraPlusConfig
from peft.helpers import create_loraplus_optimizer
from transformers import TrainingArguments
from transformers.trainer_pt_utils import get_parameter_names

class SimpleNet(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.embedding = nn.Embedding(100, 20)
        self.layer_norm = nn.LayerNorm(20)
        self.lin0 = nn.Linear(20, 20, bias=bias)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(20, 16, bias=bias)

    def forward(self, X):
        X = X.float()
        X = self.lin0(self.layer_norm(self.embedding(X)))
        X = self.relu(X)
        X = self.lin1(X)
        return X


def test_lora_plus_helper_sucess():
    model = SimpleNet()
    optimizer_cls = bnb.optim.Adam8bit
    optim_config = {
        'lr': 5e-5,
        'eps': 1e-6,
        'betas': (0.9, 0.999),
        'weight_decay': 0.0,
        "loraplus_lr_ratio": 0.2,
        "loraplus_lr_embedding": 1e-6
    }
    optim = create_loraplus_optimizer(model=model, optimizer_cls=optimizer_cls, optimizer_kwargs=optim_config)
    assert optim is not None
    assert len(optim.optimizer_grouped_parameters) == 4
