# FeRA: Frequency–Energy Constrained Routing for Effective Diffusion Adaptation Fine-Tuning
[![arXiv](https://img.shields.io/badge/arXiv-2511.17979-b31b1b.svg)](https://arxiv.org/abs/2511.17979)

## A light Example

```python
import torch
from diffusers import StableDiffusionPipeline
from peft import get_peft_model, FeRAConfig


config = FeRAConfig(
    rank=4,
    num_experts=3,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"], 
    num_bands=3
)

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")
unet = pipe.unet
unet.requires_grad_(False)

model = get_peft_model(unet, config)
model.print_trainable_parameters()

latents = torch.randn(2, 4, 64, 64).to("cuda")
t = torch.randint(0, 1000, (2,), device="cuda").long()
text_emb = torch.randn(2, 77, 768).to("cuda")

model.prepare_forward(latents)

noise_pred = model(latents, t, encoder_hidden_states=text_emb).sample

print("Forward pass successful.")
```
