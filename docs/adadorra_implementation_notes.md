# AdaDoRA Implementation Notes

This document describes the AdaDoRA implementation - a hybrid method combining AdaLoRA's adaptive rank allocation with DoRA's magnitude-direction decomposition.

---

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Implementation Details](#implementation-details)
4. [Code Review & Fixes](#code-review--fixes)
5. [Usage Guide](#usage-guide)
6. [Limitations](#limitations)

---

## Overview

### What is AdaDoRA?

AdaDoRA combines two parameter-efficient fine-tuning methods:

| Component | Source | Purpose |
|-----------|--------|---------|
| SVD decomposition (B, E, A) | AdaLoRA | Adaptive rank allocation via prunable singular values |
| Magnitude-direction separation | DoRA | Better training dynamics via weight decomposition |

### Key Innovation

Standard methods apply weight updates additively. AdaDoRA applies DoRA-style magnitude scaling to AdaLoRA's SVD-based adapter:

```
Standard LoRA:  W' = W + BA
AdaLoRA:        W' = W + B·diag(E)·A
DoRA:           W' = m · (W + BA) / ||W + BA||
AdaDoRA:        W' = m · (W + B·diag(E)·A) / ||W + B·diag(E)·A||
```

---

## Mathematical Foundation

### AdaLoRA Decomposition

AdaLoRA uses SVD-style decomposition with prunable singular values:

```
ΔW = B · diag(E) · A · (α / ranknum)
```

Where:
- `B ∈ ℝ^{d×r}` - left singular vectors
- `E ∈ ℝ^{r×1}` - singular values (can be pruned to zero)
- `A ∈ ℝ^{r×k}` - right singular vectors
- `α` - scaling factor
- `ranknum` - current effective rank

### DoRA Magnitude-Direction Decomposition

DoRA decomposes weights into magnitude and direction:

```
W = m · (V / ||V||_c)
```

Where `m` is a learnable magnitude vector and `||·||_c` is column-wise L2 norm.

### AdaDoRA Forward Pass

The complete forward pass:

```
h = (m / ||W + ΔW||) · (W + ΔW) · x + b
```

Expanded:
```
1. Base output:     z_base = W · x
2. Adapter output:  z_delta = ΔW · dropout(x)
3. Combined:        z = z_base + z_delta
4. Norm:            n = ||W + ΔW||_row  (detached from gradient)
5. Scale:           s = m / n
6. Final:           h = z · s + b
```

### Gradient Detachment

Per DoRA paper Section 4.3, the norm `||W + ΔW||` is detached from the gradient graph:

> "[...] we suggest treating ||V + ∆V||_c as a constant, thereby detaching it from the gradient graph."

This reduces training overhead while maintaining the benefits of magnitude-direction separation.

---

## Implementation Details

### File Location

```
src/peft/tuners/adalora/layer.py
```

### Key Classes

#### `AdaLoraLayer`

Base class with AdaDoRA extensions:

```python
class AdaLoraLayer(LoraLayer):
    def __init__(self, base_layer, use_dora_adaptive=True):
        self.lora_E = nn.ParameterDict({})      # singular values
        self.lora_A = nn.ParameterDict({})      # right singular vectors
        self.lora_B = nn.ParameterDict({})      # left singular vectors
        self.ranknum = nn.ParameterDict({})     # current rank (non-trainable)
        self.use_dora_adaptive = use_dora_adaptive
        if use_dora_adaptive:
            self.lora_magnitude = nn.ParameterDict({})  # DoRA magnitude
```

#### `SVDLinear`

Main layer implementation with two forward paths:

1. **AdaDoRA path** (`use_dora_adaptive=True`): Split-path with DoRA scaling
2. **Standard AdaLoRA path** (`use_dora_adaptive=False`): Original AdaLoRA

### Magnitude Initialization

Magnitude is initialized from the row-wise L2 norms of the base weight:

```python
actual_norm = weight.norm(p=2, dim=1, keepdim=True)
self.lora_magnitude[adapter_name] = nn.Parameter(actual_norm)
```

### Low-Rank Forward Path

The forward pass uses efficient low-rank computation:

```python
# efficient: two small matmuls O(d × r)
tmp = dropout(x) @ (lora_A * lora_E).T      # [batch, seq, r]
delta_out = (tmp @ lora_B.T) * scaling      # [batch, seq, out]

# NOT: one dense matmul O(d²)
# delta_out = F.linear(x, lora_B @ (lora_E * lora_A))
```

### Merge Operation

For AdaDoRA, merge applies the full transformation:

```python
if self.use_dora_adaptive:
    W_total = base_weight + delta_weight
    norm = W_total.norm(p=2, dim=1, keepdim=True)
    scale = magnitude / norm
    merged_weight = W_total * scale
else:
    merged_weight = base_weight + delta_weight
```

---

## Code Review & Fixes

### Issues Identified and Fixed

#### Issue 1: Dense Forward Computation (Critical)

**Problem**: Original implementation materialized full `[d×k]` delta weight matrix for forward pass.

**Impact**: ~256× more FLOPs for d=4096, r=8.

**Fix**: Use low-rank path for forward, only materialize for norm:
```python
# Forward (low-rank)
tmp = dropout(x) @ (A * E).T
delta_out = (tmp @ B.T) * scaling

# Norm computation (still needs dense)
delta_weight = (B @ (E * A)) * scaling
norm = (W + delta_weight).norm(...)
```

#### Issue 2: `fan_in_fan_out` Not Handled

**Problem**: Conv1D-style layers (GPT-2) use transposed weights. AdaDoRA path didn't handle this.

**Impact**: Silent wrong results on GPT-2 and similar models.

**Fix**: Added guard:
```python
if self.fan_in_fan_out:
    raise NotImplementedError("AdaDoRA does not support fan_in_fan_out=True")
```

#### Issue 3: Multi-Adapter Magnitude Composition

**Problem**: When multiple adapters active, which magnitude to use? Original code arbitrarily used first adapter's magnitude.

**Impact**: Unprincipled behavior with no theoretical basis.

**Fix**: Enforce single adapter:
```python
if len(self.active_adapters) > 1:
    raise NotImplementedError("AdaDoRA supports only one active adapter")
```

#### Issue 4: Incorrect Merge

**Problem**: Merge only added `ΔW` to `W`, ignoring DoRA scaling.

**Impact**: Merged model produces wrong outputs.

**Fix**: Apply full transformation:
```python
W_merged = (m / ||W + ΔW||) * (W + ΔW)
```

#### Issue 4b: Unmerge Not Possible

**Problem**: DoRA scaling is non-linear; cannot recover original weights.

**Fix**: Disabled unmerge for AdaDoRA with clear error message.

### Commits

```
920db0f fix: address AdaDoRA implementation issues
41d6a5e fix: AdaDoRA implementation issues
2922c80 FEAT Add DoRA adaptive mechanism to AdaLoraLayer and SVDLinear classes
```

---

## Usage Guide

### Basic Configuration

```python
from peft import AdaLoraConfig, get_peft_model

config = AdaLoraConfig(
    init_r=12,
    target_r=8,
    tinit=200,
    tfinal=1000,
    deltaT=10,
    beta1=0.85,
    beta2=0.85,
    total_step=10000,
    target_modules=["q_proj", "v_proj"],
    use_dora_adaptive=True,  # Enable AdaDoRA
)

model = get_peft_model(model, config)
```

### Training Loop

```python
for step, batch in enumerate(dataloader):
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    # AdaLoRA rank allocation (still works with AdaDoRA)
    model.base_model.update_and_allocate(step)

    optimizer.step()
    optimizer.zero_grad()
```

### Inference with Merged Weights

```python
# Merge for faster inference
model.merge_adapter()

# Run inference
outputs = model(**inputs)

# Note: unmerge() is NOT supported for AdaDoRA
```

---

## Limitations

### Current Restrictions

| Restriction | Reason |
|-------------|--------|
| No `fan_in_fan_out=True` | Conv1D-style weight layout not implemented |
| Single adapter only | No principled multi-adapter magnitude composition |
| No unmerge | Non-linear transformation not reversible |
| `nn.Linear` only | Magnitude init requires standard linear layer |

### Performance Characteristics

| Aspect | AdaLoRA | AdaDoRA |
|--------|---------|---------|
| Forward FLOPs | O(batch × seq × d × r) | O(batch × seq × d × r) + norm computation |
| Parameters | B, A, E | B, A, E, m (+d per layer) |
| Memory | Low | Low + magnitude storage |
| Merge support | Yes | Yes (one-way) |
| Unmerge support | Yes | No |

### When to Use AdaDoRA

**Good for:**
- Low-rank settings (r ≤ 8) where magnitude-direction separation helps
- Tasks where training stability is important
- Single-adapter fine-tuning scenarios

**Not ideal for:**
- Multi-adapter switching/composition
- GPT-2 style models with Conv1D layers
- Scenarios requiring frequent merge/unmerge cycles

---

## References

1. **AdaLoRA**: Zhang et al., "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning" (ICLR 2023)
2. **DoRA**: Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation" (2024) - [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)
3. **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)
