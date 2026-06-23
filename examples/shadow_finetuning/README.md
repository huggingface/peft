# ShadowPEFT: Shadow Network for Parameter-Efficient Fine-Tuning

## Introduction

[ShadowPEFT](https://arxiv.org/abs/2604.19254) augments a frozen base decoder-only model with a small, trainable
*shadow* network that runs in parallel with the backbone. At each decoder layer the shadow network injects a learned
correction into the base hidden states, while a gated update evolves the shadow hidden state as the base model
processes each layer. Only the shadow backbone and the lightweight injection/update adapters are trained; the base
model stays frozen.

The shadow module is architecturally decoupled from the backbone, so it can be attached/detached without modifying the
base weights, trained centrally, and even initialized from a smaller pre-trained model.

## Quick start

### Implicit shadow model

The shadow network is built automatically from the base model's config (fewer layers, optionally smaller MLP/attention):

```bash
python shadow_finetuning.py --base_model_name_or_path Qwen/Qwen3-0.6B --num_shadow_layers 1 --injection_hidden_size 16
```

Use `--shadow_only` at inference time to run the lightweight shadow path without a base forward pass. Note that the KV
cache is disabled by ShadowPEFT, so generation always uses `use_cache=False`.

### Explicit shadow model

Use a separate, (optionally smaller and pre-trained) model as the shadow network. When the shadow model's hidden size
differs from the base model's, ShadowPEFT inserts a trainable `shadow_hidden_projection` to bridge the two hidden
spaces. The explicit model is passed via `get_peft_model(model, config, shadow_model=...)` (and again via
`PeftModel.from_pretrained(model, path, shadow_model=...)` when reloading):

```bash
python shadow_explicit_finetuning.py \
    --base_model_name_or_path Qwen/Qwen3-8B \
    --shadow_model_name_or_path Qwen/Qwen3-0.6B
```

Add `--projected_shadow` to first package the small backbone + projection + the base `lm_head` into a single
`AutoModelForCausalLMWithHiddenProjection` (with the pseudo-inverse projection init) and use that as the explicit
shadow model. After training, the example also calls `export_shadow()` to extract a standalone shadow checkpoint.

## Citation

```bibtex
@article{li2026shadowpeft,
  title={ShadowPEFT: Shadow Network for Parameter-Efficient Fine-Tuning},
  author={Li, Xianming and Li, Zongxi and Lee, Tsz-fung Andrew and Li, Jing and Xie, Haoran and Li, Qing},
  journal={arXiv preprint arXiv:2604.19254},
  year={2026}
}
```
