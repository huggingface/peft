<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# ShadowPEFT

[ShadowPEFT](https://arxiv.org/abs/2604.19254) augments a frozen base decoder-only model with a **lightweight, centralized, pretrainable and detachable** *shadow* network that runs in parallel with the backbone. At every decoder layer the shadow network injects a learned correction into the base hidden states, while a gated update evolves the shadow hidden state as the base model processes each layer. Only the shadow backbone and the small injection/update adapters are trained; the base model stays frozen.

Because the shadow module is architecturally decoupled from the backbone, it can be attached or detached without modifying the frozen base weights (modular deployment and independent versioning), and it can be initialized from a smaller pre-trained model that serves as a reusable adaptation module for a larger backbone.

```
Input
  ├──► Shadow model (small, trainable) ──► shadow_hidden_states
  └──► Base model (frozen, large)
         layer_0 ─────────────────────────────────────────► hidden_0
         layer_1 ◄── ShadowInjection(hidden_0, shadow[0]) ─► hidden_1
         layer_2 ◄── ShadowInjection(hidden_1, shadow[1]) ─► ...
                    [ShadowUpdate evolves the shadow state at each layer]
```

Two shadow variants are supported:

- **Implicit shadow** (default): a smaller copy of the base architecture (fewer layers, optionally smaller MLP/attention) is created automatically from the base model's config.
- **Explicit shadow**: a separately (pre-)trained model that you pass via `get_peft_model(model, config, shadow_model=...)`. When its hidden size differs from the base model, a trained `shadow_hidden_projection` linear layer bridges the gap. See [`AutoModelForCausalLMWithHiddenProjection`] for the standalone distribution format of such projected shadow models.

Compared to LoRA-style methods:

- **Pros**: the adapter is a self-contained network that can be exported and run on its own (`shadow_only` inference), trained centrally and reused across tasks, and initialized from a pre-trained small model. It also exposes a second, shadow-path set of logits (`shadow_logits`) for an auxiliary loss.
- **Cons**: ShadowPEFT wraps every decoder layer and runs a parallel network, so it adds more parameters and compute than a low-rank adapter and **disables the KV cache** (full-sequence processing is required). Always pass `use_cache=False` during generation. It targets decoder-only models with at least two decoder layers and does not merge into the base weights.

## Usage

```py
from transformers import AutoModelForCausalLM
from peft import ShadowConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
config = ShadowConfig(num_shadow_layers=1, injection_hidden_size=16, task_type="CAUSAL_LM")
model = get_peft_model(model, config)
model.print_trainable_parameters()

# Generation requires `use_cache=False` because the shadow path needs the full sequence.
out = model.generate(input_ids, use_cache=False, max_new_tokens=32)
```

To use an explicit, optionally pre-trained shadow model:

```py
shadow_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model = get_peft_model(base_model, ShadowConfig(task_type="CAUSAL_LM"), shadow_model=shadow_model)
```

When loading a checkpoint trained with an explicit shadow model, pass the same shadow model back:

```py
from peft import PeftModel

model = PeftModel.from_pretrained(base_model, "path/to/adapter", shadow_model=shadow_model)
```

# API

## ShadowConfig

[[autodoc]] tuners.shadow.config.ShadowConfig

## ShadowModel

[[autodoc]] tuners.shadow.model.ShadowModel

## AutoModelForCausalLMWithHiddenProjection

[[autodoc]] tuners.shadow.projected_causal_lm.AutoModelForCausalLMWithHiddenProjection
