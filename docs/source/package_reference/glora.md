<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# GLora

Generalized Low-Rank Adaptation (**GLora**) is a highly flexible PEFT method that generalizes LoRA and related approaches. GLora allows you to decompose weight updates into multiple configurable low-rank, vector, or constant paths, providing a superset of LoRA's expressivity. Each path (A, B, C, D, E) can be independently configured, enabling a wide range of adaptation strategies.

GLora is especially useful for research and advanced applications where you want to experiment with different low-rank or structured update patterns, or combine multiple adaptation mechanisms in a single layer.

## GLoraConfig

[[autodoc]] tuners.glora.config.GLoraConfig

### Key Configuration Options
- `r`: The rank of the low-rank matrices (default: 4).
- `target_modules`: List or regex of module names to adapt (e.g., `["q_proj", "v_proj"]`).
- `config_A_B`: Path type for A and B ("LoRA", "vector", "constant", "none").
- `config_C`: Path type for C ("LoRA", "vector", "none").
- `config_D_E`: Path type for D and E ("constant", "vector", "none").

Each path can be set independently, allowing for highly customized adaptation.

## GLoraModel

[[autodoc]] tuners.glora.model.GLoraModel

- Wraps a base model and injects GLora adapters into the specified modules.
- Supports multiple adapters, adapter switching, merging/unmerging, and mixed-batch inference.
- Use `set_adapter`, `merge_and_unload`, and related methods for adapter management.

## GLoraLayer and GLoraLinear

[[autodoc]] tuners.glora.layer.GLoraLayer
[[autodoc]] tuners.glora.layer.Linear

- `GLoraLayer` is the core logic for generalized low-rank adaptation, supporting multiple adapters and flexible path configs.
- `GLoraLinear` is a drop-in replacement for `nn.Linear` with GLora support.

## Example Usage

```python
from transformers import AutoModelForCausalLM
from peft import GLoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("your-model-id")
glora_config = GLoraConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    config_A_B="LoRA",
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
- GLora is a superset of LoRA: setting all paths to "LoRA" recovers standard LoRA.
- You can use different path types for A/B/C/D/E to experiment with new adaptation strategies.
- GLora supports all standard PEFT adapter management features (add, delete, switch, merge, etc).

## See Also
- [Adapter conceptual guide](../conceptual_guides/adapter.md)
- [LoRA reference](./lora.md)
