from torch import nn

from peft import LoraConfig, get_peft_model


def test_layers_to_transform_filters_by_layer_not_expert_index():
    class ToyMoEBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(4, 4, bias=False)

            self.mlp = nn.Module()
            self.mlp.experts = nn.ModuleList([nn.Module() for _ in range(2)])
            for e in range(2):
                self.mlp.experts[e].up_proj = nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return x

    class ToyMoEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([ToyMoEBlock() for _ in range(4)])

        def forward(self, x):
            return x

    model = ToyMoEModel()

    config = LoraConfig(
        target_modules=["q_proj", "up_proj"],
        # layers_pattern="layers",
        layers_to_transform=[1],
        r=2,
        lora_alpha=4,
    )
    model = get_peft_model(model, config)
    targeted = set(model.targeted_module_names)

    assert "model.layers.1.self_attn.q_proj" in targeted
    assert "model.layers.1.mlp.experts.0.up_proj" in targeted
    assert "model.layers.1.mlp.experts.1.up_proj" in targeted
    assert "model.layers.2.mlp.experts.1.up_proj" not in targeted  # must not match by expert index
