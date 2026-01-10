# AdaMSS Fine-tuning

## Introduction

AdaMSS (Adaptive Matrix Decomposition with Subspace Selection) is a parameter-efficient fine-tuning method that decomposes weight matrices using SVD into low-rank subspaces. It uses only **~0.07%** of original trainable parameters (e.g., 59K for ViT-Base vs 86M full fine-tuning) while maintaining competitive performance. 

The method optionally supports **ASA** (Adaptive Subspace Allocation) for dynamic subspace selection during training, further improving efficiency and performance.

See the [paper](https://neurips.cc/virtual/2025/poster/119606) for more details.


## Installation & Quick Test

Install from local source:
```bash
cd peft-main && pip install -e .
pip install transformers datasets torch torchvision evaluate accelerate
```

Verify installation:
```bash
python -c "from peft import AdaMSSConfig, ASACallback; print('AdaMSS ready')"
```

## Detailed Code Explanation

**Core AdaMSS Configuration:**
```python
from peft import AdaMSSConfig, get_peft_model, ASACallback

# Configure AdaMSS with ASA
config = AdaMSSConfig(
    r=100,                          # SVD rank (full decomposition rank)
    num_subspaces=10,               # Number of subspaces (K) - initial capacity
    subspace_rank=3,                # Rank per subspace (ri) - use 1 for NLU, 3 for Vision
    target_modules=["query", "value"],  # Target attention layers
    use_asa=True,                   # Enable Adaptive Subspace Allocation
    target_kk=5,                    # Target active subspaces (ASA reduces K→5)
    modules_to_save=["classifier"], # Modules to train without decomposition
)
peft_model = get_peft_model(model, config)
```

**ASA Callback Setup:**
```python
asa_callback = ASACallback(
    target_kk=5,            # Gradually mask to 5 active subspaces
    init_warmup=50,         # Start ASA after 50 steps (Vision) or 5 epochs (NLU)
    final_warmup=1000,      # Complete masking by step 1000 (Vision) or epoch 95 (NLU)
    mask_interval=100,      # Update mask every 100 steps (Vision) or 10 epochs (NLU)
    verbose=True,           # Print ASA progress
)

# Integrate with Trainer
trainer = Trainer(
    model=peft_model,
    callbacks=[asa_callback],  # Add ASA callback
    # ... other arguments
)
```

**Key Points:**
- **Parameterization**: Total params = `r × (d_in + d_out)`, split into K subspaces of rank `ri` each
- **ASA Mechanism**: Dynamically selects `target_kk` most important subspaces from initial `num_subspaces`
- **Warmup Schedule**: ASA gradually increases masking strength from `init_warmup` to `final_warmup`
- **Vision vs NLU**: Use `subspace_rank=3` for vision, `subspace_rank=1` for NLU tasks

## Use the training example scripts

### Vision Tasks (Image Classification)

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

### NLU Tasks (GLUE Benchmark)

Run GLUE tasks (e.g., CoLA) with ASA:
```bash
python examples/adamss_finetuning/glue_adamss_asa_example.py \
    --dataset_name cola \
    --adamss_r 100 \
    --adamss_k 10 \
    --adamss_ri 1 \
    --use_asa \
    --target_kk 5 \
    --num_epochs 100 \
    --batch_size 32 \
    --output_dir ./output_cola_asa
```

Without ASA (fixed K=10):
```bash
python examples/adamss_finetuning/glue_adamss_asa_example.py \
    --dataset_name cola \
    --adamss_r 100 \
    --adamss_k 10 \
    --adamss_ri 1 \
    --num_epochs 100 \
    --batch_size 32 \
    --output_dir ./output_cola_no_asa
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


## Experimental Results

### NLU Tasks (GLUE Benchmark)

Results with AdaMSS + ASA (100 epochs, seed=0):

| Task | Model | AdaMSS Params | Metric | Score |
|------|-------|---------------|--------|-------|
| CoLA | RoBERTa-base | 27.0K (ASA K→5) | Matthews | **0.6466** |
| CoLA | RoBERTa-large | 64.8K (ASA K→5) | Matthews | **0.7093** |
| MRPC | RoBERTa-base | 27.2K (ASA K→5) | Accuracy | **0.8824** |
| MRPC | RoBERTa-large | 66.7K (ASA K→5) | Accuracy | **0.9044** |

**Notes:**
- Configuration: r=100, K=10→5 (ASA), ri=1
- AdaMSS active params with ASA (5 out of 10 subspaces selected)
- Full AdaMSS capacity: 97K (large) / 42K (base)
- Training: 100 epochs, batch_size=32, warmup_ratio=0.06

### Vision Tasks (Image Classification)

Results with AdaMSS on Stanford Cars (10 epochs, seed=0):

| Model | Method | AdaMSS Params | Test Accuracy |
|-------|--------|---------------|---------------|
| ViT-Base | AdaMSS (no ASA) | 121K (K=10) | **82.15%** |
| ViT-Base | AdaMSS + ASA | 75.0K (K→5) | **80.45%** |

**Notes:**
- Configuration: r=100, K=10, ri=3, 10 epochs, batch_size=32
- ASA dynamically selects 5 out of 10 subspaces (75K active from 121K total)



## Citation

If you use AdaMSS in your research, please cite:

```bibtex
@inproceedings{zheng2025adamss,
  title={AdaMSS: Adaptive Multi-Subspace Approach for Parameter-Efficient Fine-Tuning},
  author={Zheng, Jingjing and Lu, Wanglong and Dong, Yiming and Ji, Chaojie and Cao, Yankai and Lin, Zhouchen},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
}
```

## Reference

- [AdaMSS Paper](https://neurips.cc/virtual/2025/loc/san-diego/poster/119606)
- [PEFT Documentation](https://huggingface.co/docs/peft)
