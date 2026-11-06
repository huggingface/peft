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

[Bayesian-LoRA](https://huggingface.co/papers/2601.21003) treats both LoRA factors as random variables. Each factor is parametrized by a small matrix of *inducing weights* with a variational posterior (optionally made more expressive with a masked autoregressive flow); the full factor is drawn from the conditional of a matrix-normal prior given the inducing weights (Matheron's rule). Because the inducing matrices are tiny, the number of extra parameters is small. Training maximizes the evidence lower bound: the task loss plus a KL divergence between the posterior and the prior of the inducing weights.

- In training mode, the factors are sampled in every forward pass.
- In eval mode, a deterministic point estimate is used, so the adapter behaves like a regular LoRA adapter and can be merged, at no extra inference cost.
- For calibrated predictive uncertainty, the weights can also be sampled at inference time with [`helpers.enable_probabilistic_sampling`] and the predictions of several forward passes averaged (Monte Carlo prediction).

You may want to consider Bayesian-LoRA when you need well-calibrated confidence estimates from your fine-tuned model, e.g. on small datasets where regular LoRA tends to become overconfident.

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

During training, add the KL divergence to the task loss, typically scaled by the inverse of the number of training batches per epoch:

```py
task_loss = ...  # standard loss returned by your model
kl = model.base_model._get_bayesian_lora_loss()
loss = task_loss + kl_weight / steps_per_epoch * kl
loss.backward()
```

For Monte Carlo prediction, use [`helpers.enable_probabilistic_sampling`]:

```py
from peft.helpers import enable_probabilistic_sampling

model.eval()
with torch.no_grad(), enable_probabilistic_sampling(model):
    probs = torch.stack([model(**batch).logits.softmax(-1) for _ in range(10)]).mean(0)
```

Outside of the context manager, the model is deterministic again. Merged adapters are deterministic, so sampling requires unmerged adapters.

Bayesian-LoRA supports `nn.Linear` layers, including bitsandbytes 4-bit and 8-bit quantized layers. `add_weighted_adapter` is not supported.

A benchmark on ARC-Easy and ARC-Challenge with Llama-2-7B, comparing Bayesian-LoRA with MonteCLoRA, is available [here](https://github.com/moulelin/Bayesian-LoRA/tree/main/peft_experiments).

# API

## BayesianLoraConfig

[[autodoc]] tuners.lora.config.BayesianLoraConfig
