# AdaDoRA (AdaLoRA + DoRA) Example

This example demonstrates how to use **AdaDoRA** - a combination of AdaLoRA's adaptive rank allocation with DoRA's weight-decomposition approach for parameter-efficient fine-tuning.

## Usage

```bash
python run_glue.py \
    --model_name_or_path roberta-base \
    --task_name sst2 \ 
    --peft_type ADALORA \ 
    --use_dora \ 
    --per_device_train_batch_size 16 \ 
    --learning_rate 1e-4 \ 
    --num_train_epochs 3 \ 
    --output_dir ./output
```

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--peft_type ADALORA` | Use AdaLoRA as the base PEFT method |
| `--use_dora` | Enable DoRA's magnitude-direction decomposition |
| `--init_r` | Initial rank for each layer (default: 12) |
| `--target_r` | Target rank after pruning (default: 8) |

## What is AdaDoRA?

AdaDoRA combines:
- **AdaLoRA**: Adaptive rank allocation based on importance scores, allowing different layers to have different ranks
- **DoRA**: Weight Decomposition into magnitude and direction components for more stable training

This combination provides both parameter efficiency through adaptive pruning and training stability through weight decomposition.
