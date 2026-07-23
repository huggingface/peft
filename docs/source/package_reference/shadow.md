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

# ShadowPEFT

[ShadowPEFT](https://arxiv.org/abs/2604.19254) augments a frozen base decoder-only model with a **lightweight, pretrainable** *shadow* network that runs in parallel with the backbone. A small shadow backbone produces an initial shadow state `s^(0)`, which then rides the base model's decoder loop: at every targeted block the discrepancy between the base hidden states and the shadow state is injected back into the block input (a low-rank correction), and the shadow state is advanced by a gated residual update computed from the block output. Only the shadow components are trained; the base model stays frozen.

```
Input
  ├──► Shadow backbone (small, trainable) ──► s^(0)
  └──► Base model (frozen, large)
         block_0 ◄── inject(h, s) ─► h_0 ──► update ─► s_1
         block_1 ◄── inject(h, s) ─► h_1 ──► update ─► s_2
         ...                       (the (hidden, shadow) pair rides the loop together)
```

Because the adaptation is an **input-dependent trajectory in layer space** (the shadow state evolves with the data) rather than a static weight-space delta, ShadowPEFT **cannot be merged** into the base weights. Calling `merge`, `merge_adapter`, or `merge_and_unload` raises an explicit error. To obtain the lightweight shadow network on its own (the analogue of `merge_and_unload`), use `model.base_model.unload_shadow()`, which returns a standalone [`~tuners.shadow.layers.DetachedShadowModel`].

ShadowPEFT follows the standard PEFT API: it is a [`~tuners.tuners_utils.BaseTuner`] whose wrapped blocks are [`~tuners.tuners_utils.BaseTunerLayer`] instances, so adding multiple adapters, switching between them with `set_adapter`, deleting them, and enabling/disabling them all work as with other methods. Only **one** adapter can be active at a time, because the shadow state is a single trajectory through the network.

The shadow backbone can be built in two ways, controlled by `ShadowConfig.shadow_model`:

- `"mirror"` (default): a smaller copy of the base architecture (fewer layers via `shadow_num_hidden_layers`, optionally smaller hidden size / heads / MLP) is created automatically from the base model's config and randomly initialized. When the shadow hidden size matches the base, the frozen base input embeddings are shared (`share_embeddings`); when it differs, a trained projection bridges the gap.
- a model id or local path: the backbone is loaded with Transformers `AutoModel`, letting you initialize the shadow network from a smaller pre-trained model.

Compared to LoRA-style methods, ShadowPEFT adds more parameters and compute (it runs a parallel network and wraps whole decoder blocks), but the adapter is a self-contained network that can be trained centrally, reused across tasks, and initialized from a pre-trained small model. An optional auxiliary loss (`auxiliary_loss_weight`) applies a copy of the task head to the final shadow state and adds it to the task loss, encouraging the shadow path to solve the task on its own.

## KV cache

ShadowPEFT supports incremental decoding with a **dual** KV cache: one for the frozen base model and one for the
shadow backbone. Inject/update are token-local, so a new token only needs its own shadow state `s`; causality keeps
prefix base keys/values (computed under injection) valid. The paired object is a [`~tuners.shadow.layers.ShadowCache`],
returned as `past_key_values` when `use_cache=True`. You can pass `use_cache=True` to `generate()` as usual.

```py
out = model.generate(input_ids, max_new_tokens=32)  # dual KV cache enabled by default
# or explicitly:
out = model.generate(input_ids, use_cache=True, max_new_tokens=32)
```

`use_cache=False` still works and reprocesses the full sequence each step (useful for debugging).

## Usage

```py
from transformers import AutoModelForCausalLM
from peft import ShadowConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
config = ShadowConfig(r=8, shadow_num_hidden_layers=1, task_type="CAUSAL_LM")
model = get_peft_model(model, config)
model.print_trainable_parameters()

out = model.generate(input_ids, max_new_tokens=32)
```

To initialize the shadow backbone from a smaller pre-trained model, pass its id or path as `shadow_model`:

```py
config = ShadowConfig(shadow_model="Qwen/Qwen3-0.6B", task_type="CAUSAL_LM")
model = get_peft_model(base_model, config)
```

## Evaluating the shadow path

By default the model output (`logits`) is the **shadow-adapted base model**: the shadow corrections are injected into
the base model's hidden states at every layer, so `logits` already reflects ShadowPEFT (use `model.disable_adapter()`
to get the plain base model for comparison). The auxiliary loss additionally trains the shadow path to solve the task
on its own.

To evaluate the **standalone shadow network** (the detachable, lightweight model — the ShadowPEFT analogue of
`merge_and_unload`), use `unload_shadow()`. It returns `head(projection(backbone(x)))` as a normal task model that you
can evaluate like any other: for a causal-LM task it is a generation-capable causal LM (supports `generate()` and KV
caching), and for a sequence-classification task it pools the last token and
returns class logits.

```py
shadow = model.base_model.unload_shadow()  # a DetachedShadowModel (a PreTrainedModel)
shadow.eval()
# causal LM:
out = shadow.generate(input_ids, max_new_tokens=32)
# sequence classification:
logits = shadow(input_ids=input_ids, attention_mask=attention_mask).logits  # (batch, num_labels)
shadow.save_pretrained("standalone-shadow")
```

# API

## ShadowConfig

[[autodoc]] tuners.shadow.config.ShadowConfig

## ShadowModel

[[autodoc]] tuners.shadow.model.ShadowModel

## ShadowCache

[[autodoc]] tuners.shadow.layers.ShadowCache
