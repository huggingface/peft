import pytest
import torch
from torch import nn

from peft import LoraConfig, get_peft_model


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(8, 8)
        self.lin2 = nn.Linear(8, 8)

    def forward(self, x):
        return self.lin2(self.lin1(x))


def _get_model():
    return get_peft_model(TinyModel(), LoraConfig(r=4, lora_alpha=8, target_modules=["lin1", "lin2"]))


def test_deleting_the_last_adapter_leaves_usable_state(tmp_path):
    model = _get_model()
    model.delete_adapter("default")

    # The state used to keep the stale "default" name, so active_peft_config raised KeyError
    # and even a plain forward pass crashed.
    out = model(torch.randn(2, 8))
    assert out.shape == (2, 8)
    assert model.get_base_model().__class__ is TinyModel

    with pytest.raises(ValueError, match="no adapters left to save"):
        model.save_pretrained(tmp_path)


def test_deleting_one_of_two_adapters_still_saves(tmp_path):
    model = _get_model()
    model.add_adapter("other", LoraConfig(r=2, lora_alpha=4, target_modules=["lin1", "lin2"]))
    model.delete_adapter("default")

    # The remaining adapter must still save without touching the model card logic that reads
    # the active config.
    model.save_pretrained(tmp_path)
    assert (tmp_path / "other" / "adapter_config.json").exists()
