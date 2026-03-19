# CARTRIDGE self-study distillation (example)

This folder shows an **example** workflow for training a `CARTRIDGE` adapter via a SELF‑STUDY‑style
context-distillation objective (see the [Cartridges paper](https://huggingface.co/papers/2506.06266)).

PEFT intentionally keeps this training logic out of the core library; treat this as a starting point you can adapt.

## Installation

```bash
pip install -r requirements.txt
```

## Files

- `synthesize.py`: generates synthetic QA pairs about a corpus using vLLM with prefix caching.
- `train_distill.py`: trains a `CARTRIDGE` adapter via self-study distillation.
- `arxiv_synthesize.py`: like `synthesize.py`, with defaults for the Cartridges paper LaTeX.
- `arxiv_train.py`: like `train_distill.py`, with arxiv-specific defaults.

## How it works

1. **Synthesize**: Generate QA pairs where the model has access to the full document context
2. **Train**: Distill knowledge from teacher to student using a single model in memory:
   - Teacher (adapter disabled): document + question → logits
   - Student (adapter enabled): question + cartridge KV cache → logits
3. **Inference**: The trained cartridge provides compressed document knowledge as a KV cache prefix

## Run

### 1. Synthesize training data

```bash
python synthesize.py \
  --model Qwen/Qwen3-4B \
  --corpus_path /path/to/document.txt \
  --out_jsonl distill.jsonl \
  --num_samples 1024 \
  --use_vllm
```

With `--use_vllm`, the document is cached and reused across all samples via automatic prefix caching.

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

If you want to follow the arXiv paper example locally, you can use the LaTeX source included in this repo at
`examples/cartridge_self_study/data/cartridges.tex` (download it first):

```bash
mkdir -p examples/cartridge_self_study/data
curl -L -o examples/cartridge_self_study/data/cartridges.tex \
  https://raw.githubusercontent.com/HazyResearch/cartridges/refs/heads/main/examples/arxiv/cartridges.tex
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
# From the repo root:
# Synthesize QA pairs (uses vLLM with prefix caching)
python examples/cartridge_self_study/arxiv_synthesize.py \
  --model Qwen/Qwen3-4B \
  --corpus_path examples/cartridge_self_study/data/cartridges.tex \
  --num_samples 1024 \
  --use_vllm

# Train cartridge
python examples/cartridge_self_study/arxiv_train.py \
  --model Qwen/Qwen3-4B \
  --document examples/cartridge_self_study/data/cartridges.tex \
  --distill_jsonl distill.jsonl \
  --output_dir cartridge_adapter \
  --max_steps 500
```
