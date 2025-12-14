# CARTRIDGE self-study distillation (example)

This folder shows an **example** workflow for training a `CARTRIDGE` adapter via a SELF‑STUDY‑style
context-distillation objective (see the Cartridges paper).

PEFT intentionally keeps this training logic out of the core library; treat this as a starting point you can adapt.

## Files

- `synthesize.py`: generates synthetic single-turn conversations about a corpus into a JSONL file.
- `train_distill.py`: trains a `CARTRIDGE` adapter to match a frozen teacher on that JSONL.

## Run

```bash
source /Users/kashif/.venv/bin/activate

python examples/cartridge_self_study/synthesize.py \
  --model gpt2 \
  --corpus_path path/to/corpus.txt \
  --out_jsonl distill.jsonl \
  --num_samples 1024

python examples/cartridge_self_study/train_distill.py \
  --teacher_model gpt2 \
  --student_model gpt2 \
  --distill_jsonl distill.jsonl \
  --output_dir cartridge_adapter \
  --num_virtual_tokens 256 \
  --num_frozen_tokens 1
```

