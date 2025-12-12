# Transformer Engine ESM2 LoRA Fine-Tuning

This example demonstrates LoRA fine-tuning for Transformer Engine ESM2 token classification. It uses a synthetic dataset generated at runtime.

## Setup

Install Python dependencies:

```bash
pip install -r examples/lora_finetuning_te/requirements.txt
```

**Transformer Engine** must be installed separately and must match the system CUDA toolkit version.
See the [TE installation guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html)
or use the system-provided package (e.g., from the BioNeMo devcontainer).

Optional: authenticate with Hugging Face if your environment requires access tokens for model download.

## What this example does

- Loads a Transformer Engine ESM2 model for token classification
- Applies LoRA adapters via PEFT
- Generates random protein-like sequences
- Assigns synthetic SS3 labels (`H`, `E`, `C`) with a simple residue rule
- Trains/evaluates with `Trainer` (no explicit DDP setup)

## Run

```bash
python examples/lora_finetuning_te/lora_finetuning_te.py \
  --base_model nvidia/esm2_t6_8M_UR50D \
  --output_dir ./esm2_lora_output \
  --num_train_samples 256 \
  --num_eval_samples 64 \
  --num_epochs 1
```

## Customize

```bash
python examples/lora_finetuning_te/lora_finetuning_te.py \
  --base_model nvidia/esm2_t6_8M_UR50D \
  --output_dir ./esm2_lora_output \
  --max_length 256 \
  --batch_size 4 \
  --learning_rate 3e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1
```

## Outputs

After training, the script saves:

- PEFT adapter weights/config in `--output_dir`
- Tokenizer files in `--output_dir`

## More examples

For additional examples of TransformerEngine-accelerated transformers, visit
`https://github.com/NVIDIA/bionemo-framework/bionemo-recipes`.
