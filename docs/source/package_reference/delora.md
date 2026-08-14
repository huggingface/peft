<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# DeLoRA: Decoupled Low-rank Adaptation
[DeLoRA](https://huggingface.co/papers/2503.18225) is a parameter-efficient fine-tuning technique that implicitly maintains a Frobenius boundary with respect to the pretrained weights by normalizing and scaling learnable low-rank matrices. This effectively decouples the learning of directions (BA term) and magnitude (boundary term) of the weight updates, avoiding catastrophic shifts in the adapted weights and enhancing robustness to hyperparameter choices.

## How DeLoRA works

Like LoRA, DeLoRA represents the weight update with two trainable low-rank matrices, `A` and `B`. DeLoRA normalizes
the rows of `A` and columns of `B`, then scales the resulting update using the frozen base-weight norms and a
trainable `delora_lambda`. This separates the direction learned by `A` and `B` from the strength of the update.

The configured `delora_lambda` is the initial value of a trainable scalar for each adapted layer, not a fixed bound
throughout training. The update is also divided by `r`, so changing the rank affects both adapter capacity and its
initial scaling.

## When to use DeLoRA

DeLoRA is a good choice when:

- You want a LoRA-like workflow but need the update magnitude to be explicitly normalized and learned separately from
  its direction.
- Your fine-tuning setup is sensitive to the learning rate or training duration and would benefit from a bounded
  parameterization.
- You need an adapter that can be merged into the base weights for inference.

Standard LoRA may be a better choice when you need quantized base layers, want to adapt module types other than
`nn.Linear`, or depend on the broader set of LoRA-specific features and variants.

## Usage

```python
from peft import DeloraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
config = DeloraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    delora_lambda=15,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

With the default `init_weights=True`, `B` is initialized to zero, so applying DeLoRA does not change the model output
before training. The PEFT DeLoRA benchmarks use a learning rate of `1e-3`; in general, DeLoRA is intended to use a
learning rate roughly 10 to 100 times larger than typical LoRA settings. Avoid initializing `delora_lambda` too close
to zero; values around 10 to 15 are a useful starting range.

Use `rank_pattern` and `lambda_pattern` to override `r` or the initial `delora_lambda` for selected modules.
`module_dropout` applies dropout to the input of the adapter update during training and defaults to `0.0`.

DeLoRA currently has the following constraints:

- Only `nn.Linear` layers are supported.
- Quantized layers are not supported.

If these constraints don't work for your use case, consider other methods instead.

The abstract from the paper is:

> Parameter-Efficient FineTuning (PEFT) methods have recently gained significant popularity thanks to the widespread availability of large-scale pretrained models. These methods allow for quick adaptation to downstream tasks with minimal computational cost. However, popular finetuning methods such as LoRA exhibit limited robustness when it comes to hyperparameter choices or extended training regimes, preventing optimal out-of-the-box performance. In contrast, bounded approaches, such as ETHER, provide greater robustness but are limited to extremely low-rank adaptations and fixed-strength transformations, reducing their adaptation expressive power. In this work, we propose Decoupled Low-rank Adaptation (DeLoRA), a novel finetuning method that normalizes and scales learnable low-rank matrices. By bounding the distance of the transformation, DeLoRA effectively decouples the angular learning from the adaptation strength, enhancing robustness without compromising performance. Through evaluations on subject-driven image generation, natural language understanding, and instruction tuning, we show that DeLoRA matches or surpasses performance of competing PEFT methods, while exhibiting stronger robustness.

## Benchmark overview

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=DELORA"
	frameborder="0"
	width="850"
	height="1000"
></iframe>

# API

## DeloraConfig

[[autodoc]] tuners.delora.config.DeloraConfig

## DeloraModel

[[autodoc]] tuners.delora.model.DeloraModel
