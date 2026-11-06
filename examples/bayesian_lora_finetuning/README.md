# Bayesian-LoRA: Probabilistic Low-Rank Adaptation

## Introduction ([Paper](https://huggingface.co/papers/2601.21003))

Bayesian-LoRA treats both LoRA factors as random variables. Each factor is parametrized by a small matrix of *inducing weights* with a variational posterior (optionally made more expressive with a masked autoregressive flow); the full factor is drawn from the conditional of a matrix-normal prior given the inducing weights. Training maximizes the evidence lower bound, i.e. the task loss plus a KL divergence term.

- In training mode, the factors are sampled in every forward pass.
- In eval mode, a deterministic point estimate is used: the adapter behaves like a regular LoRA adapter and can be merged.
- For calibrated predictive uncertainty, sample the weights at inference time with `peft.helpers.enable_probabilistic_sampling` and average the predictions of several forward passes (Monte Carlo prediction).

## Quick start

```python
import torch
from peft import BayesianLoraConfig, LoraConfig, get_peft_model
from peft.helpers import enable_probabilistic_sampling

config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
    bayesian_lora_config=BayesianLoraConfig(),
)
model = get_peft_model(base_model, config)

# training: add the KL divergence, scaled by the inverse of the number of batches per epoch
model.train()
loss = task_loss + kl_weight / steps_per_epoch * model.base_model._get_bayesian_lora_loss()

# inference: Monte Carlo prediction
model.eval()
with torch.no_grad(), enable_probabilistic_sampling(model):
    probs = torch.stack([model(**batch).logits.softmax(-1) for _ in range(10)]).mean(0)
```

## Example: ARC-Easy

`bayesian_lora_finetuning.py` fine-tunes a causal LM on [ARC-Easy](https://huggingface.co/datasets/allenai/ai2_arc) (multiple-choice science questions). The model is trained to predict the letter of the correct answer; on the test split, the script reports accuracy, expected calibration error (ECE) and negative log-likelihood (NLL), once for the deterministic point estimate and once for the Monte Carlo prediction.

```bash
python bayesian_lora_finetuning.py --base_model Qwen/Qwen3-0.6B --num_epochs 2 --num_mc_samples 10
```

Use `--check_reload` to also reload the saved adapter with `PeftModel.from_pretrained` and check that it gives the same predictions. Any causal LM works as long as its tokenizer encodes the answer letters as single tokens after `Answer:`, e.g. `--base_model meta-llama/Llama-2-7b-hf`.

With the default settings (Qwen3-0.6B, adapters on `q_proj` and `v_proj`, r=8, 2 epochs, seed 0), training takes about 9 minutes on a single H200 GPU and gives on the ARC-Easy test split:

| Prediction | Accuracy ↑ | ECE ↓ | NLL ↓ |
|:--|:-:|:-:|:-:|
| Point estimate | 77.9 | 4.0 | 0.576 |
| Monte Carlo (10 samples) | 78.1 | 3.5 | 0.577 |

A comparison with MonteCLoRA on Llama-2-7B (ARC-Easy and ARC-Challenge) is available [here](https://github.com/moulelin/Bayesian-LoRA/tree/main/peft_experiments).

## Notes

- The KL divergence returned by `_get_bayesian_lora_loss()` is a sum over all inducing weights, so it is large in absolute terms; scale it by `kl_weight / steps_per_epoch` as shown above. Values of `kl_weight` below 1 temper the posterior.
- Bayesian-LoRA supports `nn.Linear` layers, including bitsandbytes 4-bit and 8-bit quantized layers. `add_weighted_adapter` is not supported.
- Merged adapters are deterministic, so Monte Carlo prediction requires unmerged adapters.

## Citation

```bibtex
@inproceedings{lin2026bayesianlora,
  title     = {Bayesian-LoRA: Probabilistic Low-Rank Adaptation of Large Language Models},
  author    = {Lin, Moule and Guan, Shuhao and Patane, Andrea and Gregg, David and Botterweck, Goetz},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
  eprint    = {2601.21003},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2601.21003}
}
```
