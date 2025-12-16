# CARTRIDGE self-study distillation (example)

This folder shows an **example** workflow for training a `CARTRIDGE` adapter via a SELF‑STUDY‑style
context-distillation objective (see the [Cartridges paper](https://huggingface.co/papers/2506.06266)).

PEFT intentionally keeps this training logic out of the core library; treat this as a starting point you can adapt.

## Files

- `synthesize.py`: generates synthetic single-turn conversations about a corpus into a JSONL file.
- `train_distill.py`: trains a `CARTRIDGE` adapter to match a frozen teacher on that JSONL.
- `arxiv_synthesize.py`: like `synthesize.py`, but defaults the corpus to `examples/cartridge_self_study/data/cartridges.tex` and uses the Cartridges seed-prompt set.
- `arxiv_train.py`: like `train_distill.py`, but defaults to training from `distill.jsonl` and writing the adapter to `cartridge_adapter/`.

## Run

```bash
python examples/cartridge_self_study/synthesize.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --corpus_path path/to/corpus.txt \
  --out_jsonl distill.jsonl \
  --num_samples 1024

python examples/cartridge_self_study/train_distill.py \
  --teacher_model Qwen/Qwen2.5-0.5B-Instruct \
  --student_model Qwen/Qwen2.5-0.5B-Instruct \
  --distill_jsonl distill.jsonl \
  --output_dir cartridge_adapter \
  --num_virtual_tokens 256 \
  --num_frozen_tokens 1
```

## arXiv wrappers

These wrappers are convenience entrypoints for running the same pipeline on the included LaTeX corpus:

```bash
python examples/cartridge_self_study/arxiv_synthesize.py
python examples/cartridge_self_study/arxiv_train.py --max_steps 2
```
