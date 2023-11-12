import torch
import torch.nn.functional as F

X = torch.rand((1000, 20))
y = (X.sum(1) > 10).long()

n_train = 800
batch_size = 64

train_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X[:n_train], y[:n_train]),
    batch_size=batch_size,
    shuffle=True,
)
eval_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X[n_train:], y[n_train:]),
    batch_size=batch_size,
)