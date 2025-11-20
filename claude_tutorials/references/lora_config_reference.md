# LoraConfig Complete Parameter Reference

## Table of Contents
- [Core Parameters](#core-parameters)
- [Targeting Parameters](#targeting-parameters)
- [Training Parameters](#training-parameters)
- [Advanced Initialization](#advanced-initialization)
- [Architectural Modifications](#architectural-modifications)
- [Runtime Configuration](#runtime-configuration)

---

## Overview

This reference documents all 30+ configuration parameters in `LoraConfig`, explaining their conceptual purpose and direct impact on model architecture and training.

**Source:** `src/peft/tuners/lora/config.py:250-799`

---

## Core Parameters

These parameters define the fundamental LoRA architecture.

| Parameter | Type | Default | Impact on Architecture | Code Location |
|-----------|------|---------|----------------------|---------------|
| **r** | `int` | `8` | **Rank of adaptation matrices**<br>- Determines size of A and B matrices<br>- A: (r × in_features), B: (out_features × r)<br>- Trainable params: r × (in_features + out_features)<br>- Higher r = more capacity, more params | `layer.py:195-196` |
| **lora_alpha** | `int` | `8` | **Scaling factor for adapter output**<br>- scaling = alpha / r (or alpha / √r if rsLoRA)<br>- Controls magnitude of adapter contribution<br>- Typical: alpha = r or alpha = 2×r<br>- Does NOT affect parameter count | `layer.py:200-202` |
| **lora_dropout** | `float` | `0.0` | **Dropout probability for LoRA layers**<br>- Applied to input before A matrix<br>- Adds nn.Dropout(p) or nn.Identity() module<br>- Regularization during training<br>- No effect during inference | `layer.py:187-192` |

### Detailed Examples

#### Rank (r) Impact

```python
# Example: GPT-2 attention layer (768 → 2304)

# r=4 (low rank)
A: (4, 768) = 3,072 params
B: (2304, 4) = 9,216 params
Total: 12,288 trainable params

# r=8 (default)
A: (8, 768) = 6,144 params
B: (2304, 8) = 18,432 params
Total: 24,576 trainable params

# r=16 (high rank)
A: (16, 768) = 12,288 params
B: (2304, 16) = 36,864 params
Total: 49,152 trainable params

# r=64 (very high rank)
A: (64, 768) = 49,152 params
B: (2304, 64) = 147,456 params
Total: 196,608 trainable params (8x more than r=8!)
```

#### Alpha/Scaling Impact

```python
# Alpha controls output magnitude, NOT parameter count

# r=8, alpha=8 (scaling = 1.0)
output = base_output + adapter_output * 1.0

# r=8, alpha=16 (scaling = 2.0) - TYPICAL
output = base_output + adapter_output * 2.0

# r=8, alpha=32 (scaling = 4.0)
output = base_output + adapter_output * 4.0

# rsLoRA: r=8, alpha=16 (scaling = 16/√8 = 5.66)
output = base_output + adapter_output * 5.66
```

---

## Targeting Parameters

These control which layers get adapted.

| Parameter | Type | Default | Impact on Architecture | Code Location |
|-----------|------|---------|----------------------|---------------|
| **target_modules** | `List[str]` or `str` | `None` | **Which layers to adapt**<br>- List: Exact or suffix matching<br>- String: Regex pattern<br>- `"all-linear"`: All linear layers<br>- Determines total adapter parameter count | `tuners_utils.py:368-383` |
| **exclude_modules** | `List[str]` or `str` | `None` | **Layers to exclude from adaptation**<br>- Applied after target_modules<br>- Reduces adapter parameter count<br>- Useful to skip output layers | `tuners_utils.py:368-383` |
| **target_parameters** | `List[str]` | `None` | **Target nn.Parameter instead of modules**<br>- For MoE models using nn.Parameter<br>- Different injection mechanism (ParamWrapper)<br>- Mutually exclusive per adapter | `lora/model.py:173-183` |
| **modules_to_save** | `List[str]` | `None` | **Additional modules to train (non-LoRA)**<br>- E.g., classification heads, embeddings<br>- Saved alongside adapter weights<br>- Full parameter training for these modules | `other.py` ModulesToSaveWrapper |

### Targeting Examples

```python
# Example 1: Attention-only adaptation
config = LoraConfig(
    target_modules=["q_proj", "v_proj"],
    # Only adapts query and value projections
    # 2 matrices per layer
)

# Example 2: All attention projections
config = LoraConfig(
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    # 4 matrices per layer
    # 2x more adapter params than Example 1
)

# Example 3: Regex pattern
config = LoraConfig(
    target_modules=".*attn.*(q|k|v)_proj",
    # Matches all attention q/k/v projections
)

# Example 4: All linear layers except output
config = LoraConfig(
    target_modules="all-linear",
    exclude_modules=["lm_head"],
    # Adapts ALL linear layers except language model head
)

# Example 5: With classification head
config = LoraConfig(
    target_modules=["q_proj", "v_proj"],
    modules_to_save=["classifier"],
    # LoRA on q/v, full training on classifier
)
```

---

## Training Parameters

Control training behavior and optimization.

| Parameter | Type | Default | Impact on Architecture | Code Location |
|-----------|------|---------|----------------------|---------------|
| **bias** | `Literal["none", "all", "lora_only"]` | `"none"` | **Which biases to train**<br>- `"none"`: No biases (typical)<br>- `"all"`: All model biases trainable<br>- `"lora_only"`: Only biases of adapted layers<br>- Increases saved adapter size if not "none" | `save_and_load.py:98-111` |
| **use_rslora** | `bool` | `False` | **Use Rank-Stabilized LoRA scaling**<br>- Changes scaling from α/r to α/√r<br>- Better performance at low ranks<br>- No change to architecture, only forward pass | `layer.py:199-202` |
| **lora_bias** | `bool` | `False` | **Add bias term to LoRA B matrix**<br>- lora_B becomes nn.Linear(r, out, bias=True)<br>- Adds out_features trainable params per layer<br>- For extracted LoRA from full fine-tuning | `layer.py:196-197` |
| **use_dora** | `bool` | `False` | **Enable Weight-Decomposed LoRA (DoRA)**<br>- Adds magnitude vector per layer<br>- Adds out_features params per adapted layer<br>- Separates magnitude and direction<br>- Better low-rank performance | `layer.py:206`, `dora.py` |

### Training Parameter Impact

```python
# Standard LoRA (r=8, 768→2304)
Trainable: 24,576 params (A + B)

# With bias="lora_only"
Trainable: 24,576 + 2,304 = 26,880 params

# With lora_bias=True
Trainable: 24,576 + 2,304 = 26,880 params
# (B gets bias term)

# With use_dora=True
Trainable: 24,576 + 2,304 = 26,880 params
# (magnitude vector added)

# With all three:
Trainable: 24,576 + 2,304 + 2,304 + 2,304 = 31,488 params
# (28% increase over standard)
```

---

## Advanced Initialization

Control how adapter weights are initialized.

| Parameter | Type | Default | Impact on Architecture | Code Location |
|-----------|------|---------|----------------------|---------------|
| **init_lora_weights** | `bool` or `str` | `True` | **Weight initialization method**<br>- `True`: B=0, A=Kaiming (default, no-op)<br>- `"gaussian"`: Gaussian with std=1/r<br>- `"eva"`: Data-driven SVD initialization<br>- `"pissa"`: Principal Singular values init<br>- `"olora"`: QR decomposition init<br>- `"loftq"`: Quantization-aware init<br>- `"corda"`: Context-oriented init<br>- `"orthogonal"`: Orthogonal init | `layer.py:209-233` |
| **loftq_config** | `LoftQConfig` | `{}` | **LoftQ quantization configuration**<br>- `loftq_bits`: Quantization bits (4)<br>- `loftq_iter`: Alternating iterations (1)<br>- Quantizes base model during init<br>- Better performance with quantized models | `config.py:54-70` |
| **eva_config** | `EvaConfig` | `None` | **EVA initialization configuration**<br>- `rho`: Max rank factor (2.0)<br>- `tau`: Similarity threshold (0.99)<br>- `use_label_mask`: Mask ignored tokens<br>- Data-driven rank redistribution | `config.py:122-171` |
| **corda_config** | `CordaConfig` | `None` | **CorDA initialization configuration**<br>- `corda_method`: "ipm" or "kpm"<br>- `cache_file`, `covariance_file`: Caching<br>- Context-oriented decomposition<br>- Faster convergence than PiSSA | `config.py:173-247` |

### Initialization Impact

```python
# Standard initialization (init_lora_weights=True)
A: Kaiming uniform ~U(-√(5/r), √(5/r))
B: Zeros
Result: ΔW = B @ A = 0 (no-op before training)

# Gaussian initialization
A: Normal(0, 1/r)
B: Zeros
Result: ΔW = 0, but A has different distribution

# EVA initialization (data-driven)
- Computes SVD of layer activations on calibration data
- Redistributes ranks based on explained variance
- Some layers get r=16, others r=4 (adaptive)
- Better utilization of parameter budget

# PiSSA initialization
- SVD of original weights: W = U S V^T
- A = √S V^T, B = U √S
- Modifies base weights: W_residual = W - B @ A
- Faster convergence, needs base weight conversion

# OLoRA initialization
- QR decomposition of W
- A initialized from R, B from Q
- Modifies base weights
- Orthogonal structure
```

---

## Architectural Modifications

Advanced features that change model structure.

| Parameter | Type | Default | Impact on Architecture | Code Location |
|-----------|------|---------|----------------------|---------------|
| **layers_to_transform** | `List[int]` or `int` | `None` | **Specific layer indices to adapt**<br>- Only adapt layers at specified indices<br>- E.g., [0, 1, 2] adapts first 3 layers<br>- Reduces adapter parameter count<br>- Must specify layers_pattern | `config.py:461-467` |
| **layers_pattern** | `List[str]` or `str` | `None` | **ModuleList name to target**<br>- E.g., "layers", "h", "blocks"<br>- Required with layers_to_transform<br>- Identifies which ModuleList to index | `config.py:468-475` |
| **rank_pattern** | `dict` | `{}` | **Per-layer rank override**<br>- Maps layer name to rank<br>- E.g., {'^model.layers.0': 16, '^model.layers.1': 8}<br>- Allows layer-specific parameter allocation<br>- Heterogeneous architecture | `config.py:476-484` |
| **alpha_pattern** | `dict` | `{}` | **Per-layer alpha override**<br>- Maps layer name to alpha<br>- Allows layer-specific scaling<br>- Independent from rank_pattern | `config.py:485-493` |
| **layer_replication** | `List[Tuple[int, int]]` | `None` | **Expand model by repeating layers**<br>- E.g., [[0, 4], [2, 5]] creates [0,1,2,3,2,3,4]<br>- Base weights shared, adapters separate<br>- Expands model depth without duplicating weights | `config.py:615-633`, `tuners_utils.py` replicate_layers |

### Architectural Modification Examples

```python
# Example 1: Adapt only first 3 layers
config = LoraConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    layers_to_transform=[0, 1, 2],
    layers_pattern="layers",
    # Only layers 0, 1, 2 get adapters
    # Saves 93% of adapter params (for 48-layer model)
)

# Example 2: Heterogeneous ranks
config = LoraConfig(
    r=8,  # default rank
    target_modules=["q_proj", "v_proj"],
    rank_pattern={
        "^model.layers.0": 16,  # First layer: rank 16
        "^model.layers.1": 16,
        "^model.layers.[2-5]": 12,  # Layers 2-5: rank 12
        # All others use r=8
    },
    # Allocate more capacity to early layers
)

# Example 3: Layer replication (depth expansion)
config = LoraConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    layer_replication=[[0, 12], [6, 18]],
    # Original: 12 layers [0-11]
    # New: layers [0-11] + [6-17] = 24 layers
    # Base weights shared, each layer has own adapter
)
```

---

## Runtime Configuration

Parameters that don't get saved with the adapter.

| Parameter | Type | Default | Impact on Architecture | Code Location |
|-----------|------|---------|----------------------|---------------|
| **runtime_config** | `LoraRuntimeConfig` | `LoraRuntimeConfig()` | **Runtime-only settings**<br>- `ephemeral_gpu_offload`: Temporary GPU transfers<br>- Used for CPU/GPU hybrid training<br>- NOT saved in adapter_config.json | `config.py:27-50` |

---

## Specialized Parameters

Advanced or experimental features.

| Parameter | Type | Default | Impact on Architecture | Code Location |
|-----------|------|---------|----------------------|---------------|
| **fan_in_fan_out** | `bool` | `False` | **Handle transposed weight layout**<br>- For Conv1D layers (GPT-2 style)<br>- Transposes weights before/after computation<br>- No change to param count | `layer.py:620`, forward pass |
| **alora_invocation_tokens** | `List[int]` | `None` | **Enable Activated LoRA (aLoRA)**<br>- Token IDs that trigger adapter activation<br>- Selective adapter application<br>- Speeds up inference with shared KV cache | `config.py:574-590`, `variants.py` |
| **use_qalora** | `bool` | `False` | **Quantization-Aware LoRA (QALoRA)**<br>- GPTQ-specific optimization<br>- Better performance for quantized models<br>- `qalora_group_size`: Pooling factor | `config.py:591-613` |
| **arrow_config** | `ArrowConfig` | `None` | **Arrow routing configuration**<br>- `top_k`: Number of adapters to combine (3)<br>- `router_temperature`: Softmax temperature (1.0)<br>- `use_gks`: Enable GenKnowSub (False)<br>- Dynamic adapter selection/combination | `config.py:72-120`, `arrow.py` |
| **trainable_token_indices** | `List[int]` or `dict` | `None` | **Selective token fine-tuning**<br>- Specify which token embeddings to train<br>- Space-efficient vocabulary adaptation<br>- Alternative to full embedding training | `config.py:520-533` |
| **megatron_config** | `dict` | `None` | **Megatron-LM integration**<br>- TransformerConfig for Megatron<br>- Parallel LoRA for ColumnParallel/RowParallel<br>- Large-scale distributed training | `config.py:494-509` |
| **ensure_weight_tying** | `bool` | `False` | **Tie adapters on tied layers**<br>- For models with tied embeddings<br>- Ensures consistency across tied weights<br>- Applies to modules_to_save | `config.py:666-676` |

---

## Configuration Examples

### Minimal Configuration (Defaults)

```python
config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
)
# Results in:
# - Rank 8 adapters
# - Scaling factor 1.0
# - No dropout
# - No bias training
# - Standard Kaiming init (B=0)
# - Adapts only q_proj and v_proj
```

### Aggressive Adaptation (Maximum Capacity)

```python
config = LoraConfig(
    task_type="CAUSAL_LM",
    r=64,  # High rank
    lora_alpha=128,  # 2x scaling
    target_modules="all-linear",  # All linear layers
    exclude_modules=["lm_head"],
    lora_dropout=0.1,
    bias="all",  # Train all biases
    use_dora=True,  # DoRA decomposition
    use_rslora=True,  # Rank-stabilized scaling
)
# Results in:
# - 8x more adapter params than default (r=64 vs r=8)
# - All linear layers adapted except output
# - Additional bias and magnitude parameters
# - Scaling = 128/√64 = 16.0 (very high)
```

### Memory-Efficient Configuration

```python
config = LoraConfig(
    task_type="CAUSAL_LM",
    r=4,  # Low rank
    lora_alpha=8,  # 2x scaling
    target_modules=["q_proj", "v_proj"],  # Minimal targets
    layers_to_transform=[0, 1, 2, 3, 4],  # First 5 layers only
    layers_pattern="layers",
    lora_dropout=0.05,  # Light regularization
)
# Results in:
# - 2x fewer params than default (r=4 vs r=8)
# - Only first 5 layers adapted
# - 90% reduction in adapter size for 48-layer model
# - ~500 KB adapter size vs ~3 MB
```

### Data-Driven Initialization

```python
from peft import EvaConfig

config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    init_lora_weights="eva",
    eva_config=EvaConfig(
        rho=2.0,  # Allow ranks up to 16
        tau=0.99,  # Similarity threshold
        use_label_mask=True,  # Mask ignored tokens
    ),
)
# Requires calibration dataset during initialization
# Automatically redistributes ranks based on layer importance
# Some layers may get r=4, others r=16
```

---

## Summary Table: Parameter Impact

| Category | Parameters | Primary Impact | Parameter Count Impact |
|----------|-----------|----------------|------------------------|
| **Core** | r, lora_alpha, lora_dropout | Adaptation capacity and magnitude | r: linear scaling |
| **Targeting** | target_modules, exclude_modules | Which layers adapted | Linear with # targets |
| **Training** | bias, use_rslora, use_dora | Training behavior, convergence | Small increase |
| **Initialization** | init_lora_weights, *_config | Initial state, convergence speed | None (init only) |
| **Architectural** | layers_to_transform, rank_pattern | Model structure, capacity allocation | Can reduce/increase |
| **Specialized** | alora_invocation_tokens, arrow_config | Advanced features | Variable |

---

## Common Configurations for Different Scenarios

### 1. Quick Prototyping (Fast Training)

```python
LoraConfig(r=8, target_modules=["q_proj", "v_proj"], lora_alpha=16)
```
- **Params**: ~1M (for 7B model)
- **Training time**: Fast
- **Use case**: Testing, prototyping

### 2. Production Quality (Balanced)

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
)
```
- **Params**: ~8M (for 7B model)
- **Training time**: Moderate
- **Use case**: Production models

### 3. Maximum Performance (Heavy)

```python
LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules="all-linear",
    use_dora=True,
    use_rslora=True,
    lora_dropout=0.1,
)
```
- **Params**: ~50M (for 7B model)
- **Training time**: Slow
- **Use case**: Maximum accuracy needed

---

**Total Parameters Documented**: 30+ configuration options with detailed architectural impact analysis

See also:
- `../annotated_code/01_get_peft_model_annotated.py` for parameter usage
- `../guides/01_model_injection_deep_dive.md` for how target_modules affects injection
