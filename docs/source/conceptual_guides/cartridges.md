# Cartridges (CARTRIDGE)

Cartridges are a prompt-learning method that stores a *compressed long-context representation* as a parameterized
KV-cache prefix. The core idea comes from the paper [Cartridges: Lightweight and general-purpose long context representations via self-study](https://huggingface.co/papers/2506.06266).

For a high-level overview and motivation, see the blog post [Cartridges: Storing long contexts in tiny caches with self-study](https://hazyresearch.stanford.edu/blog/2025-06-08-cartridges).

In PEFT, Cartridges are implemented as a first-class PEFT adapter type: `PeftType.CARTRIDGE`.

## How Cartridges differ from Prefix Tuning

Both Prefix Tuning and Cartridges are served by injecting `past_key_values` (a prefix KV cache) into the base model.

Key differences:

- **Parameterization**
  - Prefix Tuning learns *virtual token embeddings* (and optionally an MLP projection) and *produces* a KV prefix.
  - Cartridges learn the **KV prefix itself** directly (the per-layer key/value vectors for `p` virtual tokens).
- **Initialization**
  - Prefix Tuning is commonly randomly initialized.
  - Cartridges are designed to be initialized from **real prefill KV** (e.g. the first `p` tokens of a corpus/system
    prompt), freezing the first token as an attention sink for stability (`num_frozen_tokens=1` is the default).
- **Composition**
  - The paper shows Cartridges can be composed by concatenating independently trained KV prefixes.
  - In PEFT, composition is exposed as an adapter-level utility (see below).

## Basic usage (inference)

Create a CARTRIDGE adapter and initialize it from a cached prefix (or from text):

```py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import CartridgeConfig, get_peft_model
from peft import initialize_cartridge_from_text

base = AutoModelForCausalLM.from_pretrained("gpt2")
tok = AutoTokenizer.from_pretrained("gpt2")

peft_cfg = CartridgeConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=256,     # cartridge length (p)
    num_frozen_tokens=1,        # attention-sink token (recommended; default is 1)
)

model = get_peft_model(base, peft_cfg)
initialize_cartridge_from_text(model, tok, text="...your corpus text...", use_chat_template=False)

out = model.generate(**tok("Question about the corpus:", return_tensors="pt"), max_new_tokens=64)
print(tok.decode(out[0], skip_special_tokens=True))
```

Notes:
- `initialize_cartridge_from_text` is a convenience wrapper that runs a no-grad prefill on the *base model* and copies
  the first `p` cached KV tokens into the adapter.
- For chat models, `use_chat_template=True` will use the tokenizer’s chat template (if available) with the corpus text
  as a system message.

## Training (SELF-STUDY distillation)

The Cartridges paper proposes training via SELF-STUDY:
- synthesize conversations about the corpus using a teacher model with the corpus chunk in-context
- optimize the cartridge so the student model (with the cartridge) matches the teacher next-token distribution

In PEFT, the core library focuses on the adapter/runtime pieces. A working reference implementation for
SELF‑STUDY‑style distillation (data synthesis + `transformers.Trainer`) is provided as an example:
`examples/cartridge_self_study/README.md`.

## Composition

To compose multiple independently trained cartridges (concatenate their KV prefixes) into a single adapter:

```py
from peft import compose_cartridge_adapters

compose_cartridge_adapters(
    ["adapter_corpus_a", "adapter_corpus_b"],
    output_path="adapter_composed",
)
```

This creates a new adapter whose `num_virtual_tokens` is the sum of the inputs.

## Saving to / loading from the Hub

Cartridges are saved as a standard PEFT adapter (an `adapter_config.json` plus weights containing
`prompt_embeddings`), so they can be uploaded to and loaded from the Hugging Face Hub like other PEFT adapters.

```py
# Push
model.push_to_hub("org/my-cartridge-adapter")

# Load
from peft import PeftModel
loaded = PeftModel.from_pretrained(base_model, "org/my-cartridge-adapter")
```

Compatibility note: consumers must use a PEFT version that includes `PeftType.CARTRIDGE`.
