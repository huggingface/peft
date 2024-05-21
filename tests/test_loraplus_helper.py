import bitsandbytes as bnb
import torch
from torch import nn


from peft.optimizers import create_loraplus_optimizer


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
    model = SimpleNet()
    optimizer_cls = bnb.optim.Adam8bit
    optim_config = {
        'lr': 5e-5,
        'eps': 1e-6,
        'betas': (0.9, 0.999),
        'weight_decay': 0.0,
        "loraplus_lr_ratio": 0.2,
    }
    optim = create_loraplus_optimizer(model=model, optimizer_cls=optimizer_cls, optimizer_kwargs=optim_config, loraplus_lr_embedding=1e-6)
    assert optim is not None
    assert len(optim.param_groups) == 4

def test_lora_plus_optimizer_sucess():
    optimizer_cls = bnb.optim.Adam8bit
    optim_config = {
        'lr': 5e-5,
        'eps': 1e-6,
        'betas': (0.9, 0.999),
        'weight_decay': 0.0,
        "loraplus_lr_ratio": 0.2,
    }
    model: SimpleNet = SimpleNet().cuda()
    optim = create_loraplus_optimizer(model=model, optimizer_cls=optimizer_cls, optimizer_kwargs=optim_config, loraplus_lr_embedding=1e-6)
    loss = torch.nn.CrossEntropyLoss()
    bnb.optim.GlobalOptimManager.get_instance().register_parameters(model.parameters())
    x = torch.randint(100, (2, 4, 10)).cuda()
    output = model(x).permute(0, 3, 1, 2)
    label = torch.randint(16, (2,4,10,)).cuda()
    loss_value = loss(output, label)
    loss_value.backward()
    optim.step()
