import pytest
import torch
import torch.nn as nn
from peft.utils.target_selection import KappaTuneSelector, find_kappa_target_modules

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

def test_selector_basic():
    model = SimpleMLP()
    selector = KappaTuneSelector(model)
    targets = selector.get_best_targets(top_p=0.5)
    assert len(targets) == 1
    assert targets[0] in ["fc1", "fc2"]

def test_one_liner():
    model = SimpleMLP()
    targets = find_kappa_target_modules(model, top_p=1.0)
    assert len(targets) == 2

def test_num_modules():
    model = SimpleMLP()
    targets = KappaTuneSelector(model).get_best_targets(num_modules=1)
    assert len(targets) == 1
