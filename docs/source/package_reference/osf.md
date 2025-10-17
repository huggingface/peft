<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# OSF (Orthogonal Subspace Fine-tuning)

Orthogonal Subspace Fine-tuning ([OSF](https://huggingface.co/papers/2504.07097)) is a PEFT method designed for continual learning that constrains parameter updates to be orthogonal to previously important directions. This approach enables full fine-tuning while preventing catastrophic forgetting without requiring additional parameters or storing previous gradients.

The abstract from the paper is:

*Continual learning in large language models (LLMs) is prone to catastrophic forgetting, where adapting to new tasks significantly degrades performance on previously learned ones. Existing methods typically rely on low-rank, parameter-efficient updates that limit the model's expressivity and introduce additional parameters per task, leading to scalability issues. To address these limitations, we propose a novel continual full fine-tuning approach leveraging adaptive singular value decomposition (SVD). Our method dynamically identifies task-specific low-rank parameter subspaces and constrains updates to be orthogonal to critical directions associated with prior tasks, thus effectively minimizing interference without additional parameter overhead or storing previous task gradients. We evaluate our approach extensively on standard continual learning benchmarks using both encoder-decoder (T5-Large) and decoder-only (LLaMA-2 7B) models, spanning diverse tasks including classification, generation, and reasoning. Empirically, our method achieves state-of-the-art results, up to 7% higher average accuracy than recent baselines like O-LoRA, and notably maintains the model's general linguistic capabilities, instruction-following accuracy, and safety throughout the continual learning process by reducing forgetting to near-negligible levels. Our adaptive SVD framework effectively balances model plasticity and knowledge retention, providing a practical, theoretically grounded, and computationally scalable solution for continual learning scenarios in large language models.*

## How OSF Works

OSF decomposes each weight matrix into high-rank (frozen) and low-rank (trainable) components using SVD:

```
W = U_high * S_high * V_high^T + U_low * S_low * V_low^T
```

Where:
- `U_high, S_high, V_high`: Preserve important directions from previous tasks (frozen)
- `U_low, S_low, V_low`: Allow adaptation to new tasks (trainable)

During training, gradients are projected to be orthogonal to the high-rank subspace, ensuring updates don't interfere with previously learned knowledge.

## Basic Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import OSFConfig, get_peft_model

# Load base model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Configure OSF
config = OSFConfig(
    target_modules=["c_attn", "c_proj"],  # Target attention layers
    effective_rank=8,                     # Default rank for decomposition
    rank_pattern={"c_attn": 16}          # Override rank for specific modules
)

# Apply OSF
model = get_peft_model(model, config)

# Train as usual
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer("Hello world", return_tensors="pt", padding=True)
loss = model(**inputs, labels=inputs.input_ids).loss
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

## Configuration Options

### Target Modules

You can specify target modules in several ways:

```python
# Specific module names
config = OSFConfig(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

# All linear layers
config = OSFConfig(target_modules="all-linear")

# Model-specific defaults (automatically detected)
config = OSFConfig()  # Uses model-appropriate defaults
```

### Effective Rank Configuration

Control the preserved/trainable subspaces:

```python
# Global preserved rank (applies to all target modules)
config = OSFConfig(effective_rank=16)  # preserves top-16 singular directions; trains the rest

# Automatic preserved rank (50% of the smaller matrix dimension per target)
config = OSFConfig(effective_rank=None)

# Per-module preserved-rank overrides
config = OSFConfig(
    effective_rank=8,
    rank_pattern={
        "q_proj": 16,      # Higher rank for query projection
        "gate_proj": 4     # Lower rank for gate projection
    }
)
 
# Fractional preserved rank is supported (interpreted per-target as fraction * min_dim)
config = OSFConfig(effective_rank=0.8)  # preserve 80% of min_dim; train remaining 20%
config = OSFConfig(rank_pattern={"q_proj": 0.5})  # preserve 50% on q_proj, others use global/default
```

Note: OSF's `effective_rank` is the preserved (frozen) rank, not the trainable rank. The trainable rank equals `min(weight.shape) - effective_rank`. This differs from LoRA's `r`, which directly specifies the trainable rank.


## Training Advice for Continual Learning

### Sequential Task Learning

OSF is specifically designed for learning tasks sequentially. Between tasks, recompute the SVD so the preserved subspace reflects the latest weights. One simple way is to re-wrap the updated base model with OSF again:

```python
# Task 1: train on domain A with initial preserved subspace
r = 8  # initial effective rank to preserve
model = get_peft_model(base_model, OSFConfig(effective_rank=r))
train_task(model, task_1_data)

# Task 2: recompute SVD on updated weights and increase preserved subspace
base_model = model.unload()  # unwrap base model without assuming internals
r += 4  # grow preserved subspace to include Task 1 knowledge
model = get_peft_model(base_model, OSFConfig(effective_rank=r))
train_task(model, task_2_data)

# Task 3: recompute again and expand preserved subspace further
base_model = model.unload()
r += 4
model = get_peft_model(base_model, OSFConfig(effective_rank=r))
train_task(model, task_3_data)
```

### Budget Allocation for Task Sequences

When training on a known sequence of n tasks, one effective strategy is to progressively allocate model capacity to balance learning new tasks while preserving previous knowledge:

- **Task 1**: Use full capacity (train everything)
- **Task 2**: Freeze 1/n of model capacity, train remaining (n-1)/n capacity  
- **Task 3**: Freeze 2/n of model capacity, train remaining (n-2)/n capacity
- **Task n**: Freeze (n-1)/n of model capacity, use 1/n capacity for final task

This approach ensures each task gets adequate learning capacity while progressively preserving more knowledge from previous tasks.

```python
# Example: 4-task sequence with progressive budget allocation
n_tasks = 4
max_preserved_rank = 512  # Upper bound for preserved rank per target (heuristic)

for task_id in range(n_tasks):
    # Freeze increases over time; trainable capacity shrinks
    preserved_fraction = (task_id + 1) / n_tasks
    preserved_rank = int(max_preserved_rank * preserved_fraction)

    config = OSFConfig(
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        effective_rank=preserved_rank,
    )

    print(
        f"Task {task_id + 1}: Preserving rank {preserved_rank} "
        f"({preserved_fraction:.1%} of max_preserved_rank - {max_preserved_rank} frozen); trainable rank = min_dim - preserved_rank"
    )

    model = get_peft_model(base_model, config)
    train_task(model, task_data[task_id])
```

### Best Practices

1. **Effective Rank Selection**: Start with `effective_rank=None` (auto sets rank to 50% of the smaller weight dimension per target module) and adjust based on task complexity
2. **Learning Rate**: Use smaller learning rates (1e-5 to 1e-4) compared to standard fine-tuning
3. **Task Importance**: Use `rank_pattern` to allocate more capacity to critical modules
4. **Model Architecture**: OSF works best with transformer architectures having clear attention and MLP separations
5. **Capacity Planning**: For known task sequences, use progressive budget allocation (1/n, 2/n, ..., (n-1)/n freezing) to balance plasticity and stability

### Memory Considerations

OSF modifies weights in-place and doesn't add parameters, making it memory-efficient:

```python
# Memory usage remains close to base model
print(f"Base model parameters: {base_model.num_parameters():,}")
print(f"OSF model parameters: {osf_model.num_parameters():,}")  # Similar count
```

## Advanced Usage

### Custom Target Modules

For models with non-standard architectures:

```python
config = OSFConfig(
    target_modules=["dense", "intermediate.dense"],  # Custom layer names
    effective_rank=12,
    rank_pattern={"dense": 8, "intermediate.dense": 16}
)
```

### Integration with Other Methods

OSF can be combined with other techniques:

```python
# Use with gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Apply weight decay selectively (regularizes low-rank factors to limit drift/overfitting in continual updates; keep small)
optimizer = torch.optim.AdamW([
    {"params": [p for n, p in model.named_parameters() if "U_low" in n], "weight_decay": 0.01},
    {"params": [p for n, p in model.named_parameters() if "S_low" in n], "weight_decay": 0.001},
    {"params": [p for n, p in model.named_parameters() if "V_low" in n], "weight_decay": 0.01},
], lr=1e-4)
```

## OSFConfig

[[autodoc]] tuners.osf.config.OSFConfig

## OSFModel

[[autodoc]] tuners.osf.model.OSFModel

## Utility Functions

### Weight Decomposition

[[autodoc]] tuners.osf.utils.decompose_weight_matrix

[[autodoc]] tuners.osf.utils.reconstruct_weight_matrix

### Gradient Projection

[[autodoc]] tuners.osf.utils.project_gradient_to_orthogonal_space
