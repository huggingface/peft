<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# GLoRA

Generalized Low-Rank Adaptation ([GLoRA](https://huggingface.co/papers/2306.07967)) is a PEFT method that generalizes LoRA and related approaches. GLoRA decomposes updates into configurable paths (A, B, C, D, E), where each path can use low-rank, vector, constant, or disabled parameterization depending on the path.

Each path supports one of four parameterization modes. They trade off **parameter count** against **expressiveness** (how rich the update can be):

- `"lora"`: Low-rank decomposition (like standard LoRA). Uses `r * (out + in)` parameters and can express rank-`r` corrections. Most expressive, most parameters.
- `"vector"`: A single vector (e.g. shape `(out, 1)`), broadcast across the matrix. Uses `O(out)` parameters; only per-channel scaling or shifts.
- `"constant"`: A single scalar shared across all elements. Uses 1 parameter; least expressive among the trainable options.
- `"none"`: Zeros with no trainable parameters; disables that path entirely.

Not every path accepts every mode (for example, `config_D_E` does not support `"lora"`). Choosing `"lora"` on more paths increases capacity and trainable parameters; `"vector"`, `"constant"`, or `"none"` reduce both.

GLoRA is especially useful for research and advanced applications where you want to experiment with structured update patterns and combine multiple adaptation mechanisms in a single layer.

At a high level, GLoRA modifies a frozen linear layer with:

$$
W_{\mathrm{eff}} = W_0 + W_0 \odot A + B
$$

$$
b_{\mathrm{eff}} = b_0 + b_0 \odot D + E + W_0 C
$$

where each path is independently parameterized.

## GloraConfig

[[autodoc]] tuners.glora.config.GloraConfig

### Key Configuration Options
- `r`: Rank used when a path is configured as `"lora"` (default: `8`).
- `target_modules`: List or regex of module names to adapt (e.g., `["q_proj", "v_proj"]`).
- `config_A_B`: Path type for A and B ("lora", "vector", "constant", "none").
- `config_C`: Path type for C ("lora", "vector", "none").
- `config_D_E`: Path type for D and E ("constant", "vector", "none").
- `bias`: Bias handling (`"none"`, `"all"`, or `"glora_only"`).
- `init_weights`: If `True` (default), GLoRA is initialized as a no-op. If `False`, uses kaiming initialization.

Notes:
- `config_D_E` does not support `"lora"`.
- `target_modules` can be omitted for supported model types (PEFT default mappings are used).

## GloraModel

[[autodoc]] tuners.glora.model.GloraModel

- Wraps a base model and injects GLoRA adapters into the specified modules.
- Supports multiple adapters, adapter switching, merging/unmerging, and mixed-batch inference.
- Use `set_adapter`, `merge_and_unload`, and related methods for adapter management.

## GloraLayer and GloraLinear

[[autodoc]] tuners.glora.layer.GloraLayer
[[autodoc]] tuners.glora.layer.GloraLinear

- `GloraLayer` is the core logic for generalized low-rank adaptation, supporting multiple adapters and flexible path configs.
- `GloraLinear` is a drop-in replacement for `nn.Linear` with GLoRA support.
- GLoRA currently supports plain `torch.nn.Linear` base layers.

## Example Usage

```python
from transformers import AutoModelForCausalLM
from peft import GloraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("your-model-id")
glora_config = GloraConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    config_A_B="lora",
    config_C="vector",
    config_D_E="constant",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, glora_config)
model.print_trainable_parameters()

# Switch adapters, merge, etc.
model.set_adapter("default")
model.merge_and_unload()
```

## Notes
- GLoRA is a superset of LoRA: setting all paths to "lora" recovers standard LoRA.
- You can use different path types for A/B/C/D/E to experiment with new adaptation strategies.
- GLoRA supports all standard PEFT adapter management features (add, delete, switch, merge, etc).

## See Also
- [Adapter methods overview](../methods/overview#adapter-methods)
- [LoRA reference](./lora.md)
- [Paper: https://huggingface.co/papers/2306.07967](https://huggingface.co/papers/2306.07967)
