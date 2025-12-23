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

# UniLoRA

[Uni-LoRA](https://huggingface.co/papers/2506.00799) is a PEFT method that shares a compact trainable
vector bank across low-rank adapter weights. Instead of learning every LoRA matrix element independently, UniLoRA
deterministically projects entries into shared `theta_d` values and learns the shared parameters used by the adapter
update.

## Quick Start

```python
from peft import UniLoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")

config = UniLoraConfig(
    r=32,
    theta_d_length=256,
    proj_seed=42,
    target_modules=["q_proj", "v_proj"],
    unilora_dropout=0.0,
    init_weights=True,
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
```

## Important Parameters

`r` controls the low-rank adapter dimension. Larger values increase adapter capacity and memory use.

`theta_d_length` controls the length of the shared UniLoRA vector bank. This is the main trainable storage shared by
the projected adapter entries.

`proj_seed` controls deterministic index generation for the fixed projections into `theta_d`. Reusing the same seed and
configuration makes the generated adapter indices reproducible.

`target_modules` selects which modules receive UniLoRA adapters. Use module suffixes such as `["q_proj", "v_proj"]`, a
regex string, or `"all-linear"` when supported by the model architecture.

`unilora_dropout` applies dropout inside UniLoRA adapter layers during training.

`init_weights` controls UniLoRA parameter initialization. Set it to `False` to keep a random `theta_d`
initialization when you need to manage initialization manually.

`save_indices` controls whether UniLoRA checkpoints save the generated index and scale tensors together with the
shared `theta_d` parameters. Keeping this disabled gives smaller checkpoints and regenerates indices from
`proj_seed`; enabling it makes saved adapters independent from future index-generation changes.

## UniLoraConfig

[[autodoc]] tuners.unilora.config.UniLoraConfig

## UniLoraModel

[[autodoc]] tuners.unilora.model.UniLoraModel
