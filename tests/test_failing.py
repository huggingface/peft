# TODO this is just for discussion and will be removed before merging

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from peft import LoraConfig, get_peft_model


class ModelEmbConv1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(160, 5)
        self.conv1d = Conv1D(3, 5)

    def forward(self, X):
        # X.shape: 16, 10
        X = self.emb(X)
        # X.shape: 16, 10, 5
        X = self.conv1d(X)
        # X.shape: 16, 10, 3
        return X


X = torch.arange(160).view(16, 10)


def test_no_merge():
    config = LoraConfig(target_modules=["conv1d"])
    model = ModelEmbConv1D()
    model = get_peft_model(model, config)
    model(X)  # works


def test_merge_adapter():
    config = LoraConfig(target_modules=["conv1d"])
    model = ModelEmbConv1D()
    model = get_peft_model(model, config)
    model.merge_adapter()
    model(X)  # works


def test_merge_adapter_unmerge_adapter():
    config = LoraConfig(target_modules=["conv1d"])
    model = ModelEmbConv1D()
    model = get_peft_model(model, config)
    model.merge_adapter()
    model.unmerge_adapter()
    model(X)  # works


def test_merge_unload():
    config = LoraConfig(target_modules=["conv1d"])
    model = ModelEmbConv1D()
    model = get_peft_model(model, config)
    model.merge_and_unload()
    model(X)  # RuntimeError: mat1 and mat2 shapes cannot be multiplied (160x5 and 3x5)


def test_merge_unload_unmerge_adapter():
    config = LoraConfig(target_modules=["conv1d"])
    model = ModelEmbConv1D()
    model = get_peft_model(model, config)
    model.merge_and_unload()
    model.unmerge_adapter()
    model(X)  # RuntimeError: mat1 and mat2 shapes cannot be multiplied (160x5 and 3x5)
