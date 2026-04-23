# LoRA-GA: Low-Rank Adaptation with Gradient Approximation

## Introduction

[LoRA-GA](https://huggingface.co/papers/2407.05000) improves upon standard LoRA by using gradient information during initialization instead of random initialization. By performing SVD on estimated gradients, LoRA-GA initializes adapter weights in a direction that aligns with full fine-tuning, achieving 2-4x faster convergence while maintaining the same final performance. The method is orthogonal to existing LoRA variants and can be easily integrated with techniques like DoRA and LoRA+.

## Quick start

This example script demonstrates how to fine-tune a language model using LoRA-GA on the WikiText-2 dataset. The script performs gradient estimation on a small number of batches, uses those gradients to initialize LoRA adapters, and then trains the model with the Hugging Face Trainer.

```python
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraGAConfig, preprocess_loraga

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Prepare dataloader for gradient estimation
train_dataloader = DataLoader(dataset["train"], batch_size=2, shuffle=True)

# Define train_step callback for gradient estimation
def train_step():
    """Run forward and backward passes for gradient estimation."""
    data_iter = iter(train_dataloader)
    for _ in range(64):  # 64 iterations
        batch = next(data_iter)

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

# Step 1: Create LoRA-GA config
lora_ga_config = LoraGAConfig(
    direction="ArB2r",
    scale="stable",
    stable_gamma=16,
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    init_lora_weights="lora_ga",
    lora_ga_config=lora_ga_config,
    task_type="CAUSAL_LM",
)

# Step 2: Preprocess with LoRA-GA to estimate gradients
preprocess_loraga(model, lora_config, train_step)

# Step 3: Create PEFT model with LoRA-GA initialization
peft_model = get_peft_model(model, lora_config)

# Step 4: Train normally
trainer = Trainer(
    model=peft_model,
    train_dataset=dataset["train"],
    args=TrainingArguments(output_dir="./output", num_train_epochs=3),
)
trainer.train()

# Step 5: Save the trained adapter
peft_model.save_pretrained("./output")
```

## Saving with Modified Base Weights

**Important**: LoRA-GA modifies the base model weights during initialization (unlike standard LoRA). This means you need to handle saving carefully if you want to restore the original base weights.

### Option 1: Save adapter only (default)

The standard `save_pretrained()` saves the adapter with the modified base weights embedded:

```python
# This saves the adapter - base weights remain modified
peft_model.save_pretrained("./output")
```

### Option 2: Restore original base weights

If you need to restore the original base weights (e.g., for model merging or sharing), use `path_initial_model_for_weight_conversion`:

```python
# Save the original model BEFORE LoRA-GA preprocessing
model.save_pretrained("./original_model")

# ... do preprocessing and training ...

# Save adapter and convert back to original base weights
peft_model.save_pretrained(
    "./output",
    path_initial_model_for_weight_conversion="./original_model"
)
```

This is useful when:
- You want to merge the adapter with the original base weights later
- You're sharing the adapter and want users to apply it to the unmodified base model
- You need the base model weights in their original state for other purposes

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
    --learning_rate 3e-5
```

## Configuration Options

### Direction Strategies

Controls how SVD components are distributed to lora_A and lora_B:

- `ArBr`: Alternating distribution - A takes odd indices, B takes even indices
- `A2rBr`: A takes second half, B takes first half
- `ArB2r` (default): A takes first half, B takes second half - typically performs best
- `random`: Random selection of singular vectors

### Scaling Strategies

Controls initialization magnitude:

- `stable` (default): Conservative scaling using stable_gamma parameter for stable training
- `weight_svd`: Scales based on SVD of original weights for better alignment
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

## API Design

LoRA-GA follows the same pattern as PiSSA, OLoRA, and CorDA:

1. **Preprocessing**: Use `preprocess_loraga(model, lora_config, train_step)` to estimate gradients and attach them to model layers
2. **Configuration**: Use `LoraGAConfig` as a sub-config within `LoraConfig` with `init_lora_weights="lora_ga"`
3. **Initialization**: Call `get_peft_model()` after preprocessing to create the PEFT model with LoRA-GA initialization
4. **Training**: Train normally using Hugging Face Trainer or your own training loop
5. **Saving**: Use standard `save_pretrained()` to save the trained adapter

## Using LoRA-GA with Quantized Models

LoRA-GA requires full-precision gradients during preprocessing. For quantized models (e.g., BitsAndBytes 4-bit/8-bit), use a two-stage workflow:

### Step 1: Estimate gradients with full-precision model

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraGAConfig, preprocess_loraga

# Load model in full precision for gradient estimation
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Configure LoRA-GA
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    init_lora_weights="lora_ga",
    lora_ga_config=LoraGAConfig(direction="ArB2r", scale="stable"),
)

# Define your train_step (same as before)
def train_step():
    for _ in range(64):
        # Your training logic here
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

# Estimate and cache gradients
preprocess_loraga(model, lora_config, train_step, cache_file="loraga_gradients.pt")

# Clean up full-precision model
del model
torch.cuda.empty_cache()
```

### Step 2: Load quantized model and apply LoRA-GA

```python
# Load model with quantization
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)

# Apply LoRA-GA - gradients will be loaded from cache automatically
peft_model = get_peft_model(model, lora_config)

# Train normally
trainer.train()
```

**Key points:**
- Gradient estimation must use a non-quantized model (full precision or bfloat16/float16)
- Cache gradients with `cache_file` parameter to avoid re-computation
- Cached gradients are automatically loaded when applying LoRA to the quantized model
- This workflow allows memory-efficient training with quantized models while benefiting from LoRA-GA's faster convergence

## Tips

- **Gradient Estimation**: 64-128 iterations is typically sufficient. More iterations provide more accurate estimation but increase initialization time.

- **Batch Size**: Use smaller batch sizes (2-4) for gradient estimation to maximize gradient diversity.

- **Direction and Scale**: The default `direction="ArB2r"` and `scale="stable"` work well in most cases.

- **User-Defined Callback**: The `train_step` callback gives you full control over the gradient estimation process. You can customize batching, loss functions, and more.

- **Gradient Accumulation**: Do NOT call `model.zero_grad()` or `optimizer.zero_grad()` inside your `train_step` callback. LoRA-GA relies on PyTorch's natural gradient accumulation across iterations.

## Citation

```bibtex
@article{wang2024loraga,
  title={LoRA-GA: Low-Rank Adaptation with Gradient Approximation},
  author={Wang, Shaowen and Zhu, Linxi and Ding, Hengyuan and Liu, Jiaqi and Chen, Jiaming and Zhu, Kaikai and Pang, Wei and Zhu, Jun and You, Yang},
  journal={arXiv preprint arXiv:2407.05000},
  year={2024}
}
```
