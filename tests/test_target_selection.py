import torch
import torch.nn as nn
from peft.utils.target_selection import KappaTuneSelector, find_kappa_target_modules


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

        with torch.no_grad():
            # Deterministic low-κ matrix
            eps = 0.05
            u, _, v = torch.linalg.svd(torch.randn(10, 20), full_matrices=False)
            s = 1 + eps * torch.randn(min(10, 20))
            self.fc1.weight.data = u @ torch.diag_embed(s) @ v

            # Deterministic high-κ matrix
            u2, _, v2 = torch.linalg.svd(torch.randn(20, 5), full_matrices=False)
            s2 = torch.tensor([1000.0, 1.0, 1.0, 1.0, 1.0])
            self.fc2.weight.data = u2 @ torch.diag_embed(s2) @ v2


def test_selector_basic():
    """top_p=0.5 should always return only the well-conditioned fc1."""
    torch.manual_seed(42)          # SVD fully deterministic
    model = SimpleMLP()
    selector = KappaTuneSelector(model)
    targets = selector.get_best_targets(top_p=0.5)

    assert len(targets) == 1
    assert targets[0] == "fc1"


def test_one_liner():
    """top_p=1.0 should return both modules."""
    torch.manual_seed(42)
    model = SimpleMLP()
    targets = find_kappa_target_modules(model, top_p=1.0)
    assert len(targets) == 2
    assert set(targets) == {"fc1", "fc2"}


def test_num_modules():
    """num_modules=1 should return the best one (fc1)."""
    torch.manual_seed(42)
    model = SimpleMLP()
    targets = KappaTuneSelector(model).get_best_targets(num_modules=1)
    assert len(targets) == 1
    assert targets[0] == "fc1"
