# Fine-tuning with AdaMSS and ASA

This directory contains examples of fine-tuning models using **AdaMSS** (Adaptive Matrix Decomposition with Subspace Selection) from 🤗 PEFT.

## What is AdaMSS?

AdaMSS is a parameter-efficient fine-tuning method that:
- Decomposes weight matrices using SVD into low-rank subspaces
- Clusters columns into K subspaces for efficient training
- Uses only **~0.07%** of original trainable parameters (e.g., 59K for ViT-Base)
- Optionally enables **ASA** (Adaptive Subspace Allocation) for dynamic subspace selection

## Installation

1. **Install from local PEFT source** (recommended for development):
```bash
cd peft-main
pip install -e .
```

2. **Install dependencies**:
```bash
pip install transformers datasets torch torchvision evaluate accelerate
```

3. **Verify installation**:
```bash
python -c "from peft import AdaMSSConfig, ASACallback; print('✅ AdaMSS installed')"
```

## Quick Test

Run the quick test to verify everything works (< 2 minutes):
```bash
python test_adamss_quick.py
```

Expected output: `✅ Test PASSED - AdaMSS example works correctly!`

## Examples

### Basic Image Classification
See `image_classification_adamss_asa.py` for a complete example of fine-tuning ViT on image classification tasks.

## Quick Start

```bash
python image_classification_adamss_asa.py \
    --model_name_or_path google/vit-base-patch16-224-in21k \
    --dataset_name cifar10 \
    --adamss_r 100 \
    --adamss_k 10 \
    --adamss_ri 3 \
    --use_asa \
    --target_kk 5 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --learning_rate 0.01 \
    --weight_decay 0.0005 \
    --output_dir ./output
```

### Python API

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

## Important Notes

### ⚠️ Using `set_transform` with Trainer

If you use `dataset.set_transform()` for lazy data loading (like in the examples), you **must** disable automatic column removal:

```python
training_args.remove_unused_columns = False
```

**Why?** The Trainer defaults to removing unused columns (like `img`) to save memory. But `set_transform` needs access to the original column during training. Without this setting, you'll get `KeyError: 'img'`.

### 🔧 Dataset Compatibility

The examples auto-detect image and label column names for flexibility:
- Image columns: `img`, `image`
- Label columns: `label`, `labels`, `fine_label`

Tested datasets: `cifar10`, `cifar100` (others should work if they follow similar structure)

## Configuration Parameters

### AdaMSSConfig

- **r** (int): SVD decomposition rank (default: 100)
- **num_subspaces** (int): Number of subspaces K (default: 10)
- **subspace_rank** (int): Rank per subspace ri (default: 3)
- **target_modules** (list): Modules to apply AdaMSS (e.g., ["query", "value"])
- **use_asa** (bool): Enable Adaptive Subspace Allocation (default: False)
- **target_kk** (int): Target number of active subspaces when ASA is enabled

### ASACallback Parameters

- **init_warmup** (int): Steps before starting subspace masking (default: 50)
- **final_warmup** (int): Steps to reach target active subspaces (default: 1000)
- **mask_interval** (int): Steps between subspace selection updates (default: 100)
- **beta1** (float): EMA decay for importance tracking (default: 0.85)
- **beta2** (float): EMA decay for uncertainty tracking (default: 0.85)

## Expected Results

For ViT-Base on image classification benchmarks:

| Method | Params | CIFAR-10 | Stanford Cars | Pets |
|--------|--------|----------|---------------|------|
| Full FT | 86M | ~98% | ~92% | ~95% |
| LoRA (r=8) | 147K | ~97% | ~90% | ~93% |
| **AdaMSS (rk=3)** | **59K** | **~97%** | **~91%** | **~94%** |
| **AdaMSS + ASA** | **59K** | **~98%** | **~92%** | **~95%** |

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
