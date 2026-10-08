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

# NoRA

[Normalized Low-Rank Adaptation (NoRA)](https://huggingface.co/papers/2608.31036) normalizes the LoRA down-projection matrices. The paper also describes applying normalization only at initialization. PEFT implements this initialization-only approach through `init_lora_weights="nora"` in [`LoraConfig`]. It normalizes the columns of the randomly initialized LoRA A weight and initializes the LoRA B weight to zero. Both matrices remain trainable, and the base weights are left unchanged by initialization.

## Initialization

For a linear layer, the adapter weight update is $\Delta W = sBA$, where $A$ has shape `(r, in_features)`, $B$ has shape `(out_features, r)`, and $s$ is the LoRA scaling factor.

Nora starts from the random A weight created by the LoRA linear layer and normalizes each column across the rank dimension:

$$
A_{:,j} \leftarrow \frac{A_{:,j}}{\max(\lVert A_{:,j} \rVert_2, 10^{-6})}, \qquad B \leftarrow 0.
$$

The norm is clamped to a minimum of `1e-6` to avoid division by zero. Since B is zero, the adapter weight update is zero before training. With `bias="none"` and `lora_bias=False`, the adapter initially preserves the base model output.

Normalization is performed only during initialization; it is not a constraint applied after each optimizer step. Nora does not require a calibration dataset or a weight decomposition.

## Usage

> [!IMPORTANT]
> For NoRA, set `lora_alpha` equal to `r` to keep a 1:1 ratio, for example, `r=8, lora_alpha=8`. With the default `use_rslora=False`, this gives a scaling factor of 1. Keep `use_rslora=False` for this setup; enabling rank-stabilized LoRA changes the scaling factor to `lora_alpha / sqrt(r)`.

The following example applies Nora to two linear layers in a small PyTorch model:

```py
from torch import nn

from peft import LoraConfig, get_peft_model

model = nn.Sequential(
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
)

config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["0", "2"],
    init_lora_weights="nora",
    bias="none",
    lora_bias=False,
    use_rslora=False,
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

For a Transformers model, set `target_modules` to the names of the linear layers you want to adapt, such as `["q_proj", "v_proj"]` when those names are present in the model.

Nora uses the existing LoRA configuration, training, saving, and loading APIs. There is no separate `NoraConfig` or `NoraModel` class. The scaling factor is `lora_alpha / r` by default, or `lora_alpha / sqrt(r)` when `use_rslora=True`.

## Supported layers

Use Nora with `torch.nn.Linear` target modules. Embedding initialization with Nora is not implemented. Explicitly select supported target modules when configuring the adapter.

## API

See [`LoraConfig`] and [`LoraModel`] for the configuration and model APIs shared with LoRA.
