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

# Cartridges

Cartridges are a prompt-learning method that stores a compressed long-context representation as a parameterized KV-cache
prefix. The core idea comes from the paper
[Cartridges: Lightweight and general-purpose long context representations via self-study](https://huggingface.co/papers/2506.06266).

For a high-level overview and motivation, see the blog post
[Cartridges: Storing long contexts in tiny caches with self-study](https://hazyresearch.stanford.edu/blog/2025-06-08-cartridges).

## How Cartridges differ from Prefix Tuning

Both Prefix Tuning and Cartridges are served by injecting `past_key_values` (a prefix KV cache) into the base model.

- Prefix Tuning learns virtual token embeddings (and optionally an MLP projection) and produces a KV prefix.
- Cartridges learn the KV prefix itself directly (the per-layer key/value vectors for `p` virtual tokens), and are
  designed to be initialized from real prefill KV (for example, the first `p` tokens of a corpus/system prompt).

The paper also recommends freezing the first token as an attention sink for stability (`num_frozen_tokens=1` is the
default).

## Usage (inference)

Load a trained CARTRIDGE adapter and run generation:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_path = "path/to/cartridge_adapter"

base = AutoModelForCausalLM.from_pretrained(model_id)
model = PeftModel.from_pretrained(base, adapter_path)

tok = AutoTokenizer.from_pretrained(model_id)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

out = model.generate(**tok("Question about the corpus:", return_tensors="pt"), max_new_tokens=64)
print(tok.decode(out[0], skip_special_tokens=True))
```

If you need to create and initialize a cartridge before training, see the initialization options below.

## Initialization options

The paper discusses a few practical initialization strategies:

- Random KV (default): create a `CartridgeConfig` and start training. This initializes the KV prefix randomly.
- KV from the first tokens of a prompt/corpus: use `initialize_kv_prefix_from_text(model, tokenizer, text=...)`. This
  runs a prefill on `text` and copies the resulting KV cache for the first `num_virtual_tokens` into the adapter.
- KV from an existing cache: use `initialize_kv_prefix_from_past_key_values(model, past_key_values=...)` if you already
  have a `past_key_values` object from a base-model prefill.

## Training

The Cartridges paper proposes a SELF-STUDY distillation objective (a frozen base model provides teacher logits; the
CARTRIDGE adapter is trained so the student matches the teacher’s next-token distribution over the target segment).
PEFT keeps training logic out of the core library; see
`https://github.com/huggingface/peft/tree/main/examples/cartridge_self_study` for a reference workflow.
The example scripts use the frozen base model as the teacher and the adapted model as the student, so both share the
same underlying checkpoint.

## Composition

To concatenate independently trained cartridges into a single adapter, use `compose_cartridge_adapters(...)`.

## CartridgeConfig

[[autodoc]] tuners.cartridge.config.CartridgeConfig

## CartridgeEncoder

[[autodoc]] tuners.cartridge.model.CartridgeEncoder
