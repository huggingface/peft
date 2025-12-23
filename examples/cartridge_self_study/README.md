# CARTRIDGE self-study distillation (example)

This folder shows an **example** workflow for training a `CARTRIDGE` adapter via a SELF‑STUDY‑style
context-distillation objective (see the [Cartridges paper](https://huggingface.co/papers/2506.06266)).

PEFT intentionally keeps this training logic out of the core library; treat this as a starting point you can adapt.

## Installation

```bash
pip install -r requirements.txt
```

## Files

- `synthesize.py`: generates synthetic single-turn conversations about a corpus into a JSONL file using vLLM.
- `train_distill.py`: trains a `CARTRIDGE` adapter to match a frozen teacher on that JSONL.
- `arxiv_synthesize.py`: like `synthesize.py`, but defaults to the Cartridges paper LaTeX and uses seed prompts.
- `arxiv_train.py`: like `train_distill.py`, with arxiv-specific defaults.

## How it works

1. **Synthesize**: Generate QA pairs where the model has access to the full document context
2. **Train**: The student (with cartridge) learns to match teacher outputs; both use the same base model but:
   - Teacher sees: document + question
   - Student sees: question only + cartridge KV cache
3. **Inference**: The trained cartridge provides compressed document knowledge as a KV cache prefix

## Run

### 1. Synthesize training data

```bash
python synthesize.py \
  --model Qwen/Qwen3-4B \
  --corpus_path /path/to/document.txt \
  --out_jsonl distill.jsonl \
  --num_samples 1024
```

### 2. Train cartridge

```bash
python train_distill.py \
  --model Qwen/Qwen3-4B \
  --document /path/to/document.txt \
  --distill_jsonl distill.jsonl \
  --output_dir cartridge_adapter \
  --num_virtual_tokens 256 \
  --num_frozen_tokens 1 \
  --max_steps 500
```

### 3. Load and use cartridge

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
model = PeftModel.from_pretrained(model, "cartridge_adapter")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
inputs = tokenizer("What is the document about?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## arXiv example

Convenience wrappers for training on the Cartridges paper LaTeX:

```bash
# Synthesize QA pairs (uses vLLM with prefix caching)
python arxiv_synthesize.py \
  --model Qwen/Qwen3-4B \
  --num_samples 1024

# Train cartridge
python arxiv_train.py \
  --model Qwen/Qwen3-4B \
  --document /path/to/cartridges.tex \
  --distill_jsonl distill.jsonl \
  --output_dir cartridge_adapter \
  --max_steps 500
```
