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

### Mirror shadow backbone (default)

The shadow backbone is built automatically from the base model's config (fewer layers, optionally smaller
MLP/attention). This is the default `shadow_model="mirror"`:

```bash
python shadow_finetuning.py --base_model_name_or_path Qwen/Qwen3-0.6B --shadow_num_hidden_layers 1 --r 8
```

Note that the KV cache is disabled by ShadowPEFT, so generation always uses `use_cache=False`.

### Pretrained shadow backbone

Initialize the shadow backbone from a separate, (optionally smaller) pretrained model by passing its id/path as
`ShadowConfig(shadow_model=...)`. When the pretrained backbone's hidden size differs from the base model's, ShadowPEFT
inserts a trainable projection to bridge the two hidden spaces. After training, `unload_shadow()` returns the standalone
shadow network:

```bash
python shadow_explicit_finetuning.py \
    --base_model_name_or_path Qwen/Qwen3-8B \
    --shadow_model Qwen/Qwen3-0.6B
```

## Citation

```bibtex
@article{li2026shadowpeft,
  title={ShadowPEFT: Shadow Network for Parameter-Efficient Fine-Tuning},
  author={Li, Xianming and Li, Zongxi and Lee, Tsz-fung Andrew and Li, Jing and Xie, Haoran and Li, Qing},
  journal={arXiv preprint arXiv:2604.19254},
  year={2026}
}
```
