# Transformer Engine ESM2 LoRA Fine-Tuning

This example demonstrates LoRA fine-tuning for Transformer Engine ESM2 token classification. It uses a synthetic dataset generated at runtime.

## Setup

Choose one of the two options below.

### Option A: Docker (recommended)

Build a self-contained image based on the publicly available NVIDIA PyTorch container
(`nvcr.io/nvidia/pytorch:26.01-py3`), which already ships CUDA, cuDNN, and Transformer Engine:

```bash
docker build -t lora-te examples/lora_finetuning_transformer_engine
```

Run the training inside the container:

```bash
docker run --gpus all --rm lora-te \
  python lora_finetuning_te.py \
    --base_model nvidia/esm2_t6_8M_UR50D \
    --output_dir ./esm2_lora_output \
    --num_train_samples 256 \
    --num_eval_samples 64 \
    --num_epochs 1
```

Or start an interactive session to experiment:

```bash
docker run --gpus all --rm -it lora-te bash
```

### Option B: Virtual environment

Create and activate a virtual environment, then install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r examples/lora_finetuning_transformer_engine/requirements.txt
```

**Transformer Engine** must be installed separately and must match the system CUDA toolkit version.
See the [TE installation guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html)
for details.

## What this example does

- Loads a Transformer Engine ESM2 model for token classification
- Applies LoRA adapters via PEFT
- Generates random protein-like sequences
- Assigns synthetic SS3 labels (`H`, `E`, `C`) with a simple residue rule
- Trains/evaluates with `Trainer` (no explicit DDP setup)

## Run

```bash
python examples/lora_finetuning_transformer_engine/lora_finetuning_te.py \
  --base_model nvidia/esm2_t6_8M_UR50D \
  --output_dir ./esm2_lora_output \
  --num_train_samples 256 \
  --num_eval_samples 64 \
  --num_epochs 1
```

> **Note:** The default ESM2 models on Hugging Face Hub ship custom modeling code.
> You must pass `--trust_remote_code` to allow loading that code.

## Customize

```bash
python examples/lora_finetuning_transformer_engine/lora_finetuning_te.py \
  --base_model nvidia/esm2_t6_8M_UR50D \
  --trust_remote_code \
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
