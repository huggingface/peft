# LoRA-GA: Low-Rank Adaptation with Gradient Approximation

## Introduction

[LoRA-GA](https://huggingface.co/papers/2407.05000) improves upon standard LoRA by using gradient information during initialization instead of random initialization. By performing SVD on estimated gradients, LoRA-GA initializes adapter weights in a direction that aligns with full fine-tuning, achieving 2-4x faster convergence while maintaining the same final performance. The method is orthogonal to existing LoRA variants and can be easily integrated with techniques like DoRA and LoRA+.

## Quick start

```python
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from accelerate import Accelerator
from peft import LoraGAConfig, get_peft_model
from peft.utils import LoraGAContext, estimate_gradient, save_loraga_model_init, save_loraga_model_final

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Step 1: Estimate gradients
accelerator = Accelerator()
model_for_grad = accelerator.prepare(model)
dataloader = accelerator.prepare(train_dataloader)

named_grad = estimate_gradient(
    model_for_grad,
    dataloader,
    accelerator,
    iters=64,
)

# Step 2: Initialize LoRA-GA with gradients
lora_ga_config = LoraGAConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    direction="ArB2r",
    scale="stable",
)

with LoraGAContext(model, named_grad):
    peft_model = get_peft_model(model, lora_ga_config)

# Step 3: Save initial state
save_loraga_model_init(peft_model, "./output")

# Step 4: Train normally
trainer = Trainer(
    model=peft_model,
    train_dataset=dataset["train"],
    args=TrainingArguments(output_dir="./output", num_train_epochs=3),
)
trainer.train()

# Step 5: Save final state with delta
save_loraga_model_final(peft_model, "./output")
```

## Run the finetuning script

Simply run:

```bash
python examples/lora_ga_finetuning/lora_ga_finetuning.py \
    --base_model gpt2 \
    --dataset_name wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --output_dir ./lora_ga_output
```

### Customize LoRA-GA parameters

You can customize the direction and scaling strategies:

```bash
python examples/lora_ga_finetuning/lora_ga_finetuning.py \
    --base_model gpt2 \
    --direction ArB2r \
    --scale stable \
    --stable_gamma 16 \
    --grad_estimate_iters 64
```

### Full example with all parameters

```bash
python lora_ga_finetuning.py \
    --base_model "gpt2" \
    --dataset_name "wikitext" \
    --dataset_config "wikitext-2-raw-v1" \
    --output_dir "./lora_ga_output" \
    --r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --direction "ArB2r" \
    --scale "stable" \
    --stable_gamma 16 \
    --grad_estimate_iters 64 \
    --grad_estimate_batch_size 2 \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 3e-4
```

## Configuration Options

### Direction Strategies

Controls how SVD components are distributed to lora_A and lora_B:

- `ArBr`: A=U√S, B=√S×V^T - Balanced distribution
- `A2rBr`: A=U×S, B=V^T - More weight on A
- `ArB2r` (default): A=U, B=S×V^T - More weight on B
- `random`: Random initialization (equivalent to standard LoRA)

### Scaling Strategies

Controls initialization magnitude:

- `stable` (default): Conservative scaling using stable_gamma parameter
- `weight_svd`: Scales based on SVD of original weights
- `gd_scale`: Scales based on gradient descent step size
- `unit`: Unit scaling (no adjustment)

## Use the model on 🤗

You can load and use the model as any other 🤗 models:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("gpt2")
model = PeftModel.from_pretrained(model, "path/to/lora_ga_output")
```

## LoRA-GA vs. LoRA

Key differences and advantages:

1. **Faster Convergence**: LoRA-GA achieves 2-4x faster convergence compared to standard LoRA due to gradient-aligned initialization.

2. **Same Final Performance**: LoRA-GA maintains the same or better final performance as standard LoRA.

3. **Initialization Overhead**: LoRA-GA requires a gradient estimation phase (typically 1-2 minutes for 64 iterations), but this is quickly amortized during training.

4. **Orthogonal to Other Methods**: LoRA-GA can be combined with DoRA, LoRA+, quantization, and other LoRA enhancements.

## Tips

- **Gradient Estimation**: 64-128 iterations is typically sufficient. More iterations provide more accurate estimation but increase initialization time.

- **Batch Size**: Use smaller batch sizes (2-4) for gradient estimation to maximize gradient diversity.

- **Direction and Scale**: The default `direction="ArB2r"` and `scale="stable"` work well in most cases.

- **Delta Saving**: Always use `save_loraga_model_init()` and `save_loraga_model_final()` to properly track adapter deltas, since LoRA-GA modifies base weights during initialization.

## Citation

```bibtex
@article{loraga2024,
  title={LoRA-GA: Low-Rank Adaptation with Gradient Approximation},
  author={Authors},
  journal={arXiv preprint arXiv:2407.05000},
  year={2024}
}
```
