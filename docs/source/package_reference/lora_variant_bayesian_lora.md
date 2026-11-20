<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Bayesian-LoRA

> [!NOTE]
> This is a variant of LoRA and therefore everything that is possible with LoRA is valid for this method except otherwise stated on this page.

A model fine-tuned with regular LoRA gives a single answer and tends to be overconfident, especially when it was fine-tuned on a small dataset. [Bayesian-LoRA](https://huggingface.co/papers/2601.21003) learns a distribution over the adapter weights instead of a single value. Averaging the predictions of several weight samples gives better calibrated probabilities, and the disagreement between the samples tells you how uncertain the model is.

You may want to consider Bayesian-LoRA when the confidence of the model matters, not only its accuracy, for example:

- classification or multiple-choice question answering where low-confidence predictions are rejected or passed on to a human;
- detecting inputs that are unlike the fine-tuning data (out-of-distribution detection);
- selecting the most informative examples to label next (active learning);
- fine-tuning on small datasets, where regular LoRA quickly becomes overconfident.

If you only need the accuracy, you can still use a Bayesian-LoRA adapter like a regular LoRA adapter: in eval mode it is deterministic and can be merged, at no extra inference cost.

## How it works

Bayesian-LoRA treats both LoRA factors as random variables. Each factor is parametrized by a small matrix of *inducing weights* with a variational posterior (optionally made more expressive with a masked autoregressive flow); the full factor is drawn from the conditional of a matrix-normal prior given the inducing weights (Matheron's rule). Because the inducing matrices are tiny, the number of extra parameters is small. Training maximizes the evidence lower bound: the task loss plus a KL divergence between the posterior and the prior of the inducing weights.

The inducing-weight parametrization builds on [Ritter et al. (2021)](https://arxiv.org/abs/2105.14594), "Sparse Uncertainty Representation in Deep Learning with Inducing Weights", and its implementation in [bayesianize](https://github.com/microsoft/bayesianize).

- In training mode, the factors are sampled in every forward pass.
- In eval mode, a deterministic point estimate is used, so the adapter behaves like a regular LoRA adapter.
- For calibrated predictive uncertainty, the weights can also be sampled at inference time with [`helpers.enable_probabilistic_sampling`] and the predictions of several forward passes averaged (Monte Carlo prediction).

## Usage

To enable Bayesian-LoRA, pass a `BayesianLoraConfig` to `LoraConfig`:

```py
from peft import BayesianLoraConfig, LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    bayesian_lora_config=BayesianLoraConfig(),
)
model = get_peft_model(base_model, config)
```

During training, add the KL divergence to the task loss:

```py
task_loss = ...  # standard loss returned by your model, averaged over the batch
kl = model.base_model._get_bayesian_lora_loss()  # call once after each forward pass
loss = task_loss + kl_weight / steps_per_epoch * kl
loss.backward()
```

The KL divergence is a sum over all inducing weights of the model, whereas the task loss is a mean over the examples of one batch, so the KL divergence has to be scaled down:

- `steps_per_epoch` is the number of batches in one pass over the training set, `len(train_dataloader)`. It does not depend on the number of epochs you train: the KL divergence belongs to the whole dataset, and dividing by the number of batches spreads it evenly over the batches of each epoch. With gradient accumulation, still use the number of batches, not the number of optimizer steps.
- `kl_weight` controls the strength of the regularization towards the prior. `kl_weight=1` corresponds to the evidence lower bound. Smaller values let the model fit the training data more closely, which usually gives a higher accuracy but less uncertainty; values between 0.1 and 1 are a good starting point (the example linked below uses 0.2). Increase `kl_weight` if the model is overconfident on held-out data, i.e. its confidence is higher than its accuracy; decrease it if the model underfits or its accuracy is clearly below that of regular LoRA.

For Monte Carlo prediction, use [`helpers.enable_probabilistic_sampling`]:

```py
from peft.helpers import enable_probabilistic_sampling

model.eval()
with torch.no_grad(), enable_probabilistic_sampling(model):
    probs = torch.stack([model(**batch).logits.softmax(-1) for _ in range(10)]).mean(0)
```

Outside of the context manager, the model is deterministic again. Merged adapters are deterministic, so sampling requires unmerged adapters.

Bayesian-LoRA supports `nn.Linear` layers, including bitsandbytes 4-bit and 8-bit quantized layers. `add_weighted_adapter` is not supported.

A complete working example is available at [`examples/bayesian_lora_finetuning`](https://github.com/huggingface/peft/tree/main/examples/bayesian_lora_finetuning). A benchmark on ARC-Easy and ARC-Challenge with Llama-2-7B, comparing Bayesian-LoRA with MonteCLoRA, is available [here](https://github.com/moulelin/Bayesian-LoRA/tree/main/peft_experiments).

# API

## BayesianLoraConfig

[[autodoc]] tuners.lora.config.BayesianLoraConfig
