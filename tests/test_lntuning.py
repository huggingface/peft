import tempfile

import pytest
import torch
from torch import nn

from peft import LNTuningConfig, PeftModel, get_peft_model, get_peft_model_state_dict


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(8)
        self.lin = nn.Linear(8, 8)

    def forward(self, x):
        return self.lin(self.ln(x))


@pytest.fixture
def trained_model():
    torch.manual_seed(0)
    model = get_peft_model(M(), LNTuningConfig(target_modules=["ln"])).eval()
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.add_(torch.randn_like(param))  # make the adapter non-trivial
    return model


def test_lntuning_unmerged_state_dict_contains_adapter_weights(trained_model):
    state_dict = get_peft_model_state_dict(trained_model)
    adapter = trained_model.base_model.model.ln.ln_tuning_layers["default"]

    weight_key = next(key for key in state_dict if key.endswith(".weight"))
    assert torch.allclose(state_dict[weight_key].flatten(), adapter.weight.flatten())
    bias_key = next(key for key in state_dict if key.endswith(".bias"))
    assert torch.allclose(state_dict[bias_key].flatten(), adapter.bias.flatten())


def test_lntuning_merged_state_dict_contains_trained_weights(trained_model):
    trained_model.merge_adapter()
    state_dict = get_peft_model_state_dict(trained_model)
    base_layer = trained_model.base_model.model.ln.base_layer

    weight_key = next(key for key in state_dict if key.endswith(".weight"))
    assert torch.allclose(state_dict[weight_key].flatten(), base_layer.weight.flatten())
    bias_key = next(key for key in state_dict if key.endswith(".bias"))
    assert torch.allclose(state_dict[bias_key].flatten(), base_layer.bias.flatten())


def test_lntuning_merged_state_dict_roundtrip(trained_model):
    # Merging swaps base_layer and ln_tuning_layers[adapter_name], so the trained weights end up
    # in base_layer. Saving must pick them up (issue #3884): before the fix, the state dict stored
    # the original base weights, and reloading silently lost all training.
    x = torch.randn(4, 8)
    expected = trained_model(x)

    trained_model.merge_adapter()
    assert torch.allclose(trained_model(x), expected)  # sanity: merge preserves outputs

    with tempfile.TemporaryDirectory() as tmp_dir:
        trained_model.save_pretrained(tmp_dir)
        torch.manual_seed(0)  # same init as when the original model was created
        loaded = PeftModel.from_pretrained(M(), tmp_dir).eval()

    assert torch.allclose(loaded(x), expected)
