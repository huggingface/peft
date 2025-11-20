# Model Injection: How PEFT Modifies Base Model Architecture

## Table of Contents
- [Overview](#overview)
- [Conceptual Understanding](#conceptual-understanding)
- [Step-by-Step Injection Process](#step-by-step-injection-process)
- [Code Mapping](#code-mapping)
- [Layer Replacement Examples](#layer-replacement-examples)
- [Key Data Structures](#key-data-structures)

---

## Overview

**What is Model Injection?**

Model injection is the process by which PEFT **modifies a base model's architecture in-place** by replacing or wrapping specific layers with adapter-enhanced versions. For LoRA, this means replacing `nn.Linear` layers with `LoraLinear` layers that add low-rank adaptation matrices.

**Why "Injection" Not "Addition"?**

The term "injection" is used because:
1. Adapters are **injected into existing layers**, not added as separate modules
2. The base model structure is **modified in-place**
3. Original layers are **wrapped** (not duplicated)
4. The process is **reversible** (can unload adapters)

**Critical Insight:**

```python
# BEFORE injection:
model.transformer.h[0].attn.c_attn  # nn.Linear(768, 2304)

# AFTER injection:
model.transformer.h[0].attn.c_attn  # Linear(base_layer=nn.Linear(768, 2304), lora_A=..., lora_B=...)
```

The layer at the same path is now a different class, but maintains the same interface!

---

## Conceptual Understanding

### The Three Stages of Injection

```
┌─────────────────┐
│  Base Model     │  Stage 1: User calls get_peft_model()
│  (Frozen)       │           Creates PeftModel wrapper
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Target Layer   │  Stage 2: Identify which layers to adapt
│  Identification │           Based on config.target_modules
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Layer          │  Stage 3: Replace layers with adapter versions
│  Replacement    │           nn.Linear → LoraLinear
└─────────────────┘
```

### Key Principles

1. **In-Place Modification**: The base model object is modified directly
2. **Structural Preservation**: Module hierarchy remains identical
3. **Weight Sharing**: Base weights are wrapped, not copied
4. **Selective Targeting**: Only specific modules are adapted
5. **Reversibility**: Adapters can be unloaded to restore original model

---

## Step-by-Step Injection Process

### Complete Flow Diagram

```
User Code
    |
    v
get_peft_model(model, config, adapter_name="default")
    |
    ├─> 1. Create PeftModel wrapper
    |      File: src/peft/mapping_func.py:30-128
    |      Code: return PeftModel(model, peft_config, adapter_name)
    |
    v
PeftModel.__init__(model, peft_config, adapter_name)
    |
    ├─> 2. Create tuner instance (e.g., LoraModel)
    |      File: src/peft/peft_model.py:125-128
    |      Code: cls = PEFT_TYPE_TO_TUNER_MAPPING[peft_config.peft_type]
    |            self.base_model = cls(model, {adapter_name: peft_config}, adapter_name)
    |
    v
LoraModel.__init__(model, peft_config, adapter_name)
    |                                          [Inherits from BaseTuner]
    ├─> 3. Call BaseTuner.__init__()
    |      File: src/peft/tuners/tuners_utils.py:263-295
    |
    v
BaseTuner.__init__()
    |
    ├─> 4. Call inject_adapter()
    |      File: src/peft/tuners/tuners_utils.py:295
    |      Code: self.inject_adapter(self.model, adapter_name)
    |
    v
BaseTuner.inject_adapter(model, adapter_name)
    |
    ├─> 5. Prepare config (infer target_modules if needed)
    |      File: src/peft/tuners/tuners_utils.py (in inject_adapter)
    |      Code: peft_config = self._prepare_adapter_config(peft_config, model_config)
    |
    ├─> 6. Iterate through all model layers
    |      Code: for key, target in model.named_modules():
    |
    ├─> 7. Check if layer should be adapted
    |      Code: if self._check_target_module_exists(peft_config, key):
    |
    ├─> 8. Create and replace layer
    |      Code: self._create_and_replace(peft_config, adapter_name, target, ...)
    |
    v
LoraModel._create_and_replace(config, adapter_name, target, ...)
    |
    ├─> 9. Determine adapter type (based on quantization, layer type)
    |      File: src/peft/tuners/lora/model.py:284-348
    |      Code: new_module = self._create_new_module(lora_config, adapter_name, target, ...)
    |
    ├─> 10. Replace old module with new adapter-enhanced module
    |       File: src/peft/tuners/lora/model.py:255-282
    |       Code: setattr(parent, child_name, new_module)
    |
    v
Result: Model with injected adapters!
```

---

## Code Mapping

### Critical File Paths and Functions

Here are the 8 most important locations in the injection pipeline:

| **Step** | **File Path** | **Function/Class** | **Lines** | **Purpose** |
|----------|---------------|-------------------|----------|-----------|
| 1 | `src/peft/mapping_func.py` | `get_peft_model()` | 30-128 | Entry point, creates PeftModel |
| 2 | `src/peft/peft_model.py` | `PeftModel.__init__()` | 103-144 | Instantiates tuner class |
| 3 | `src/peft/tuners/tuners_utils.py` | `BaseTuner.__init__()` | 263-298 | Base initialization, calls inject |
| 4 | `src/peft/tuners/tuners_utils.py` | `BaseTuner.inject_adapter()` | ~650-850 | Main injection loop |
| 5 | `src/peft/tuners/tuners_utils.py` | `BaseTuner._check_target_module_exists()` | 368-383 | Validates target modules |
| 6 | `src/peft/tuners/lora/model.py` | `LoraModel._create_and_replace()` | 159-253 | LoRA-specific replacement |
| 7 | `src/peft/tuners/lora/model.py` | `LoraModel._create_new_module()` | 284-348 | Creates LoRA layer instance |
| 8 | `src/peft/tuners/lora/layer.py` | `Linear.__init__()` | 599-635 | Initializes LoRA Linear layer |

### Detailed Code Examples

#### 1. Entry Point: `get_peft_model()`

**Location:** `src/peft/mapping_func.py:122-128`

```python
return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](
    model,
    peft_config,
    adapter_name=adapter_name,
    autocast_adapter_dtype=autocast_adapter_dtype,
    low_cpu_mem_usage=low_cpu_mem_usage,
)
```

**What happens:**
- Routes to task-specific PeftModel subclass (e.g., `PeftModelForCausalLM`)
- Passes base model and config to PeftModel constructor

#### 2. PeftModel Initialization

**Location:** `src/peft/peft_model.py:125-128`

```python
cls = PEFT_TYPE_TO_TUNER_MAPPING[peft_config.peft_type]
with ctx():
    self.base_model = cls(model, {adapter_name: peft_config}, adapter_name)
```

**What happens:**
- Looks up tuner class (LoRA → `LoraModel`, IA3 → `IA3Model`, etc.)
- Instantiates tuner, passing base model and config
- Tuner's `__init__` will trigger injection

#### 3. Tuner Initialization and Injection Trigger

**Location:** `src/peft/tuners/tuners_utils.py:292-295`

```python
self.active_adapter: str | list[str] = adapter_name
self._pre_injection_hook(self.model, self.peft_config[adapter_name], adapter_name)
if peft_config != PeftType.XLORA:
    self.inject_adapter(self.model, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
```

**What happens:**
- Sets active adapter name
- Calls pre-injection hook (for LoRA, handles layer replication)
- **Triggers `inject_adapter()`** - the main injection function

#### 4. Main Injection Loop

**Location:** `src/peft/tuners/tuners_utils.py` (in `inject_adapter()`)

**Simplified pseudo-code:**

```python
def inject_adapter(self, model, adapter_name):
    # Prepare configuration
    peft_config = self._prepare_adapter_config(peft_config, model_config)

    # Iterate through all model layers
    for key, target in model.named_modules():
        # Check if this layer should be adapted
        if not self._check_target_module_exists(peft_config, key):
            continue

        # Get parent module
        parent, target_name = get_parent_and_name(key)

        # Create and replace with adapter version
        self._create_and_replace(
            peft_config,
            adapter_name,
            target,
            target_name,
            parent,
            current_key=key,
        )

    # Mark only adapters as trainable
    self._mark_only_adapters_as_trainable(model)
```

#### 5. Target Module Checking

**Location:** `src/peft/tuners/tuners_utils.py:368-383`

```python
@staticmethod
def _check_target_module_exists(peft_config: PeftConfig, key: str):
    target_modules = peft_config.target_modules
    if isinstance(target_modules, str):
        # Regex matching
        return re.fullmatch(target_modules, key)
    else:
        # Exact or suffix matching
        return any(key.endswith(target_key) for target_key in target_modules)
```

**What it does:**
- Checks if `key` (e.g., `"transformer.h.0.attn.c_attn"`) matches `target_modules`
- Supports regex patterns or exact/suffix matching
- Returns `True` if layer should be adapted

#### 6. Layer Creation and Replacement (LoRA)

**Location:** `src/peft/tuners/lora/model.py:159-253`

```python
def _create_and_replace(self, lora_config, adapter_name, target, target_name, parent, current_key):
    # ... validation code ...

    # Determine rank (r) and alpha for this specific layer
    r = lora_config.rank_pattern.get(current_key, lora_config.r)
    alpha = lora_config.alpha_pattern.get(current_key, lora_config.lora_alpha)

    # Build kwargs for new module
    kwargs = {
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": lora_config.lora_dropout,
        "fan_in_fan_out": lora_config.fan_in_fan_out,
        "init_lora_weights": lora_config.init_lora_weights,
        # ... more config ...
    }

    # Check if target already has LoRA adapter
    if isinstance(target, LoraLayer) and not isinstance(target, AdaLoraLayer):
        # Just add new adapter to existing LoRA layer
        target.update_layer(adapter_name, r, lora_alpha=alpha, ...)
    else:
        # Create new LoRA-enhanced module
        new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)

        # Replace old module with new one
        self._replace_module(parent, target_name, new_module, target)
```

**What happens:**
1. Extracts layer-specific rank and alpha (allows per-layer customization)
2. Checks if layer already has LoRA (multi-adapter support)
3. Either adds new adapter to existing LoRA layer, or creates new LoRA layer
4. Replaces module in parent's children

#### 7. Module Replacement

**Location:** `src/peft/tuners/lora/model.py:255-282`

```python
def _replace_module(self, parent, child_name, new_module, child):
    # Set new module as attribute of parent
    setattr(parent, child_name, new_module)

    # Unwrap base layer if needed
    if hasattr(child, "base_layer"):
        child = child.base_layer

    # Move adapter parameters to same device as base layer
    for name, module in new_module.named_modules():
        if (self.prefix in name) or ("ranknum" in name):
            # Determine device from base layer weights
            if hasattr(child, "qweight"):
                weight = child.qweight
            elif hasattr(child, "weight"):
                weight = child.weight
            # ... more cases ...

            # Move adapter module to correct device
            module.to(weight.device)
```

**Critical insight:**
- Uses `setattr(parent, child_name, new_module)` to replace layer
- This modifies the model **in-place**
- Old layer is not deleted (garbage collected), but new layer takes its place
- Adapter parameters moved to same device as base weights

---

## Layer Replacement Examples

### Example 1: GPT-2 Attention Layer

**Before Injection:**

```python
model.transformer.h[0].attn.c_attn
# Type: transformers.pytorch_utils.Conv1D
# Params: weight (2304, 768), bias (2304,)
# Total: 1,771,776 + 2,304 = 1,774,080 parameters (all frozen after PEFT)
```

**After Injection:**

```python
model.transformer.h[0].attn.c_attn
# Type: peft.tuners.lora.layer.Linear
# Structure:
#   base_layer: Conv1D(weight=(2304, 768), bias=(2304,))  [FROZEN]
#   lora_A: nn.Linear(768, 8, bias=False)  [TRAINABLE]
#   lora_B: nn.Linear(8, 2304, bias=False)  [TRAINABLE]
#   lora_dropout: nn.Dropout(p=0.1)
#   scaling: 2.0 (= lora_alpha / r = 16 / 8)
# Trainable: 6,144 + 18,432 = 24,576 parameters (1.39% of original)
```

**Forward Pass Computation:**

```python
# Old (pre-PEFT):
output = base_layer(x)  # Conv1D: output = x @ W.T + b

# New (with LoRA):
output = base_layer(x) + lora_B(lora_A(lora_dropout(x))) * scaling
       = W x + b + (B A) dropout(x) * (alpha/r)
```

### Example 2: BERT Query Projection

**Before Injection:**

```python
model.encoder.layer[0].attention.self.query
# Type: torch.nn.Linear
# Params: weight (768, 768), bias (768,)
# Total: 589,824 + 768 = 590,592 parameters
```

**After Injection:**

```python
model.encoder.layer[0].attention.self.query
# Type: peft.tuners.lora.layer.Linear
# Structure:
#   base_layer: nn.Linear(768, 768)  [FROZEN]
#   lora_A: nn.Linear(768, 16, bias=False)  [TRAINABLE]
#   lora_B: nn.Linear(16, 768, bias=False)  [TRAINABLE]
#   scaling: 2.0 (if alpha=32, r=16)
# Trainable: 12,288 + 12,288 = 24,576 parameters (4.16% of original)
```

---

## Key Data Structures

### 1. ModuleDict for Multi-Adapter Support

```python
# In LoraLayer (src/peft/tuners/lora/layer.py:99-122)
class LoraLayer:
    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}  # {adapter_name: rank}
        self.lora_alpha = {}  # {adapter_name: alpha}
        self.scaling = {}  # {adapter_name: scaling_factor}
        self.lora_dropout = nn.ModuleDict({})  # {adapter_name: Dropout}
        self.lora_A = nn.ModuleDict({})  # {adapter_name: nn.Linear}
        self.lora_B = nn.ModuleDict({})  # {adapter_name: nn.Linear}
        # ...
```

**Why dictionaries?**
- Allows multiple adapters on same layer
- Each adapter has independent A/B matrices
- Can switch between adapters at runtime

**Example multi-adapter:**

```python
# After adding two adapters:
layer.lora_A = {
    "task_A": nn.Linear(768, 8),
    "task_B": nn.Linear(768, 16),
}
layer.lora_B = {
    "task_A": nn.Linear(8, 768),
    "task_B": nn.Linear(16, 768),
}
```

### 2. Active Adapters List

```python
# In BaseTuner (src/peft/tuners/tuners_utils.py)
self.active_adapter: str | list[str] = adapter_name
```

**Controls which adapters are used in forward pass:**

```python
# Single adapter active:
model.active_adapters = ["task_A"]

# Multiple adapters active (merged forward):
model.active_adapters = ["task_A", "task_B"]
```

### 3. Target Modules Configuration

```python
# In LoraConfig:
target_modules: Union[list[str], str]

# Examples:
target_modules = ["q_proj", "v_proj"]  # Exact matching
target_modules = ".*attn.*"  # Regex pattern
target_modules = "all-linear"  # Special: all linear layers
```

---

## Summary

**Model injection is a three-stage process:**

1. **Identification**: Determine which layers to adapt based on `target_modules`
2. **Creation**: Create adapter-enhanced layer wrapping the original layer
3. **Replacement**: Replace original layer with adapter version using `setattr()`

**Key characteristics:**
- **In-place**: Original model object is modified
- **Selective**: Only specified layers are adapted
- **Reversible**: Can unload adapters with `model.unload()`
- **Efficient**: Base weights shared (wrapped), not copied
- **Multi-adapter**: Supports multiple adapters on same model

**Critical files to understand:**
1. `src/peft/mapping_func.py` - Entry point
2. `src/peft/peft_model.py` - PeftModel wrapper
3. `src/peft/tuners/tuners_utils.py` - BaseTuner injection logic
4. `src/peft/tuners/lora/model.py` - LoRA-specific replacement
5. `src/peft/tuners/lora/layer.py` - LoRA layer implementation

---

**Next Steps:**
- See `02_adapter_saving_deep_dive.md` for how PEFT saves only adapter weights
- See `../annotated_code/02_lora_forward_pass_annotated.py` for forward pass details
- See `../references/lora_config_reference.md` for all configuration options
