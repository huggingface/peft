# AdaMSS Fine-tuning

## Introduction

AdaMSS (Adaptive Matrix Decomposition with Subspace Selection) is a parameter-efficient fine-tuning method that decomposes weight matrices using SVD into low-rank subspaces. It uses only **~0.07%** of original trainable parameters (e.g., 59K for ViT-Base vs 86M full fine-tuning) while maintaining competitive performance. 

The method optionally supports **ASA** (Adaptive Subspace Allocation) for dynamic subspace selection during training, further improving efficiency and performance.

See the [paper](https://neurips.cc/virtual/2025/poster/119606) for more details.

## Quick start

```python
import torch
from peft import AdaMSSConfig, get_peft_model, ASACallback
from transformers import AutoModelForImageClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load model and dataset
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=10,
    ignore_mismatched_sizes=True
)
dataset = load_dataset("cifar10")

# Configure AdaMSS
config = AdaMSSConfig(
    r=100,                          # SVD rank
    num_subspaces=10,               # Number of subspaces (K)
    subspace_rank=3,                # Rank per subspace (ri)
    target_modules=["query", "value"],
    use_asa=True,                   # Enable adaptive subspace allocation
    target_kk=5,                    # Target active subspaces
    modules_to_save=["classifier"], # Train classifier head
)

peft_model = get_peft_model(model, config)

# Setup ASA callback
asa_callback = ASACallback(
    target_kk=5,
    init_warmup=50,
    final_warmup=1000,
    mask_interval=100,
)

# Train
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    learning_rate=0.01,
    remove_unused_columns=False,  # Important: keep original columns for set_transform
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset["train"],
    callbacks=[asa_callback],
)
trainer.train()
peft_model.save_pretrained("adamss-vit-cifar10")
```

## Use the training example script

Run the provided script with your configuration:
```bash
python examples/adamss_finetuning/image_classification_adamss_asa.py \
    --model_name_or_path google/vit-base-patch16-224-in21k \
    --dataset_name cifar10 \
    --adamss_r 100 \
    --adamss_k 10 \
    --adamss_ri 3 \
    --use_asa \
    --target_kk 5 \
    --output_dir ./output
```

For CIFAR-100:
```bash
python examples/adamss_finetuning/image_classification_adamss_asa.py \
    --model_name_or_path google/vit-base-patch16-224-in21k \
    --dataset_name cifar100 \
    --adamss_r 100 \
    --adamss_k 10 \
    --adamss_ri 3 \
    --use_asa \
    --target_kk 5 \
    --output_dir ./output
```

## Full example

```bash
python image_classification_adamss_asa.py \
    --model_name_or_path google/vit-base-patch16-224-in21k \
    --dataset_name cifar10 \
    --adamss_r 100 \
    --adamss_k 10 \
    --adamss_ri 3 \
    --use_asa \
    --target_kk 5 \
    --asa_init_warmup 50 \
    --asa_final_warmup 1000 \
    --asa_mask_interval 100 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 0.01 \
    --weight_decay 0.0005 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --logging_steps 100 \
    --output_dir ./adamss_cifar10
```

## Installation & Quick Test

Install from local source:
```bash
cd peft-main && pip install -e .
pip install transformers datasets torch torchvision evaluate accelerate
```

Verify installation:
```bash
python -c "from peft import AdaMSSConfig, ASACallback; print('✅ AdaMSS ready')"
```

Quick test (< 2 minutes):
```bash
python test_adamss_quick.py  # Should output: ✅ Test PASSED
```

## Python API Details

```python
from peft import AdaMSSConfig, get_peft_model, ASACallback
from transformers import AutoModelForImageClassification, Trainer

# Load base model
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")

# Configure AdaMSS
config = AdaMSSConfig(
    r=100,              # SVD rank
    num_subspaces=10,   # Number of subspaces (K)
    subspace_rank=3,    # Rank per subspace (ri)
    target_modules=["query", "value"],  # Apply to attention modules
    use_asa=True,       # Enable adaptive subspace allocation
    target_kk=5,        # Target active subspaces
)

# Apply PEFT
model = get_peft_model(model, config)

# Setup ASA callback (if use_asa=True)
asa_callback = ASACallback(
    target_kk=5,
    init_warmup=50,
    final_warmup=1000,
    mask_interval=100,
)

# Train with Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[asa_callback],  # Add ASA callback
)

trainer.train()
```

### AdaMSSConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `r` | int | 100 | SVD decomposition rank |
| `num_subspaces` | int | 10 | Number of subspaces (K) |
| `subspace_rank` | int | 3 | Rank per subspace (ri) |
| `target_modules` | list | - | Modules to apply AdaMSS (e.g., ["query", "value"]) |
| `use_asa` | bool | False | Enable Adaptive Subspace Allocation |
| `target_kk` | int | None | Target active subspaces when ASA enabled |
| `modules_to_save` | list | None | Modules to train without decomposition |

### ASACallback Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_kk` | int | - | Target number of active subspaces |
| `init_warmup` | int | 50 | Steps before starting masking |
| `final_warmup` | int | 1000 | Steps to reach target active subspaces |
| `mask_interval` | int | 100 | Steps between subspace selection updates |
| `beta1` | float | 0.85 | EMA decay for importance tracking |
| `beta2` | float | 0.85 | EMA decay for uncertainty tracking |

## Important Notes

**⚠️ Using `set_transform` with Trainer**

When using `dataset.set_transform()` for lazy data loading, you must disable automatic column removal:
```python
training_args.remove_unused_columns = False
```
Without this, the Trainer will remove original columns (like `img`) that `set_transform` needs at runtime, causing `KeyError`.

**📊 Supported Datasets**

The examples auto-detect column names (`img`/`image`, `label`/`fine_label`). Tested on: `cifar10`, `cifar100`.

## Expected Results

### Vision Tasks (ViT-Base)

Image classification benchmarks *(results from paper)*:

| Method | Params | CIFAR-10 | Stanford Cars | Pets |
|--------|--------|----------|---------------|------|
| Full FT | 86M | ~98% | ~92% | ~95% |
| LoRA (r=8) | 147K | ~97% | ~90% | ~93% |
| **AdaMSS (rk=3)** | **59K** | **~97%** | **~91%** | **~94%** |
| **AdaMSS + ASA** | **59K** | **~98%** | **~92%** | **~95%** |

### NLU Tasks (RoBERTa-base)

GLUE benchmark - CoLA (Grammar Acceptability) *(tested and verified)*:

| Method | Params (Adapter Only) | Matthews Correlation |
|--------|----------------------|---------------------|
| Full FT | 124M | ~0.68 |
| **AdaMSS (r=100, K=10, ri=1)** | **42,432** | **~0.65** |
| **AdaMSS + ASA (K→5)** | **~32,000** | **~0.64-0.65** |

**Key Findings:**
- ✅ ASA reduces AdaMSS parameters by ~25% (42K → 32K)
- ✅ Performance maintained while using fewer parameters
- ✅ Tested on multiple random seeds showing consistent results

## Citation

If you use AdaMSS in your research, please cite:

```bibtex
@inproceedings{
zheng2025adamss,
    title={Ada{MSS}: Adaptive Multi-Subspace Approach for Parameter-Efficient Fine-Tuning},
    author={Jingjing Zheng and Wanglong Lu and Yiming Dong and Chaojie Ji and Yankai Cao and Zhouchen Lin},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/forum?id=8ZdWmpYxT0}
}
```

## Reference

- [AdaMSS Paper](https://neurips.cc/virtual/2025/loc/san-diego/poster/119606)
- [PEFT Documentation](https://huggingface.co/docs/peft)
