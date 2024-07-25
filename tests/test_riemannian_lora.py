import torch
from torch import nn

from peft.optimizers import create_riemannian_optimizer


# adapt from test_loraplus_helper.py
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


def test_riemannian_helper_sucess():
    """
    Test if the optimizer is correctly created with desired properties
    """
    model = SimpleNet()
    optimizer_cls = torch.optim.AdamW
    optim_config = {
        "lr": 5e-5,
        "eps": 1e-6,
        "betas": (0.9, 0.999),
        "weight_decay": 0.0,
    }
    lr_embedding = 1e-6
    reg = 1e-4
    optim = create_riemannian_optimizer(
        model=model, optimizer_cls=optimizer_cls, optimizer_kwargs=optim_config, lr_embedding=lr_embedding, reg=reg
    )
    assert optim is not None
    assert len(optim.param_groups) == 3
    assert optim.param_groups[0]["is_lora"] is True
    assert optim.param_groups[0]["reg"] == reg
    assert optim.param_groups[1]["is_lora"] is False
    assert "reg" not in optim.param_groups[1].keys()
    assert optim.param_groups[2]["is_lora"] is False
    assert "reg" not in optim.param_groups[2].keys()


def test_riemannian_optimizer_sucess():
    """
    Test if the optimizer is correctly created and step function runs without any exception
    """
    optimizer_cls = torch.optim.AdamW
    optim_config = {"lr": 5e-5, "eps": 1e-6, "betas": (0.9, 0.999), "weight_decay": 0.0}
    model: SimpleNet = SimpleNet().cuda()
    lr_embedding = 1e-6
    reg = 1e-4
    optim = create_riemannian_optimizer(
        model=model, optimizer_cls=optimizer_cls, optimizer_kwargs=optim_config, lr_embedding=lr_embedding, reg=reg
    )
    loss = torch.nn.CrossEntropyLoss()
    x = torch.randint(100, (2, 4, 10)).cuda()
    output = model(x).permute(0, 3, 1, 2)
    label = torch.randint(
        16,
        (
            2,
            4,
            10,
        ),
    ).cuda()
    loss_value = loss(output, label)
    loss_value.backward()
    optim.step()
    output = model(x).permute(0, 3, 1, 2)
    loss_value = loss(output, label)

