import torch

from torch import nn

from peft import LoraPlusConfig
from peft.helpers import create_loraplus_optimizer

class SimpleNet(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        # note: out_features must be > rank or else OFT will be an identity transform
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(20, 16, bias=bias)

    def forward(self, X):
        X = X.float()
        X = self.lin0(X)
        X = self.relu(X)
        X = self.lin1(X)
        return X


def test_lora_plus_helper_sucess():
    config = LoraPlusConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        loraplus_lr_ratio=0.2
    )
    optimizer_cls = torch.optim.AdamW
    optim_config = {'lr': 5e-5, 'eps': 1e-6, 'betas': (0.9, 0.999), 'weight_decay': 0.0}
    optim = create_loraplus_optimizer(model=SimpleNet(), config=config, optimizer_cls=optimizer_cls, optimizer_kwargs=optim_config)
    assert optim is not None