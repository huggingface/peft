# Adapter Saving: How PEFT Checkpoints Only Adapter Weights

## Table of Contents
- [Overview](#overview)
- [Why Save Only Adapter Weights?](#why-save-only-adapter-weights)
- [The Filtering Mechanism](#the-filtering-mechanism)
- [Code Deep Dive](#code-deep-dive)
- [File Format and Structure](#file-format-and-structure)
- [Loading Saved Adapters](#loading-saved-adapters)

---

## Overview

**The Core Idea:**

When you save a PEFT model, you save **ONLY the adapter parameters**, not the entire model. This means:

```python
# Full model size: ~500 MB
# Saved adapter size: ~3 MB
# Reduction: 167x smaller!
```

**Why this matters:**
1. **Massive storage savings**: Share hundreds of adapters for one base model
2. **Fast deployment**: Download/upload adapters in seconds
3. **Easy sharing**: Host adapters on HuggingFace Hub
4. **Multi-task learning**: Switch between tasks by swapping adapters

---

## Why Save Only Adapter Weights?

### The Problem with Full Model Saving

**Traditional fine-tuning:**

```python
# Save full model
model.save_pretrained("my_finetuned_model")
# Creates: pytorch_model.bin (~500 MB for GPT-2)

# To use different task
model2.save_pretrained("my_finetuned_model_task2")
# Creates: pytorch_model.bin (~500 MB again!)

# Total storage: 1 GB for 2 tasks
```

**PEFT approach:**

```python
# Save base model once
base_model.save_pretrained("gpt2_base")
# Creates: pytorch_model.bin (~500 MB)

# Save adapter 1
peft_model.save_pretrained("adapter_task1")
# Creates: adapter_model.safetensors (~3 MB)

# Save adapter 2
peft_model.save_pretrained("adapter_task2")
# Creates: adapter_model.safetensors (~3 MB)

# Total storage: ~506 MB for 2 tasks (2x smaller!)
# For 100 tasks: ~800 MB instead of 50 GB!
```

### Mathematical Insight

**Base model parameters:**
- GPT-2: 124M parameters × 4 bytes = 496 MB
- Frozen during training

**LoRA adapter parameters (r=8):**
- Per layer: (768 × 8) + (8 × 768) = 12,288 params
- 48 layers × 2 attention projections = 96 adapted layers
- Total: ~1.2M parameters × 4 bytes = 4.8 MB
- Trainable during fine-tuning

**Key insight:** Since base model is frozen, we only need to save the 1% that changed!

---

## The Filtering Mechanism

### Three-Step Filtering Process

```
Full Model State Dict
        |
        v
    [Filter 1]
    Keep only keys containing adapter markers
    (e.g., "lora_", "ia3_", "dora_")
        |
        v
    [Filter 2]
    Filter by specific adapter name
    (for multi-adapter models)
        |
        v
    [Filter 3]
    Remove adapter name from keys
    (for portability)
        |
        v
    Adapter-Only State Dict
```

### Visual Example

**Full Model State Dict:**

```python
{
    # Base model weights (FROZEN - not saved)
    'transformer.h.0.attn.c_attn.weight': Tensor([2304, 768]),  # 1.77M params
    'transformer.h.0.attn.c_attn.bias': Tensor([2304]),  # 2.3K params

    # LoRA adapter weights (TRAINABLE - saved)
    'transformer.h.0.attn.c_attn.lora_A.default.weight': Tensor([8, 768]),  # 6K params
    'transformer.h.0.attn.c_attn.lora_B.default.weight': Tensor([2304, 8]),  # 18K params

    # Base model weights (FROZEN - not saved)
    'transformer.h.0.attn.c_proj.weight': Tensor([768, 768]),  # 590K params
    'transformer.h.0.attn.c_proj.bias': Tensor([768]),  # 768 params

    # LoRA adapter weights (TRAINABLE - saved)
    'transformer.h.0.attn.c_proj.lora_A.default.weight': Tensor([8, 768]),  # 6K params
    'transformer.h.0.attn.c_proj.lora_B.default.weight': Tensor([768, 8]),  # 6K params

    # ... (many more layers)
}
```

**After Filter 1 (Keep only "lora_" keys):**

```python
{
    'transformer.h.0.attn.c_attn.lora_A.default.weight': Tensor([8, 768]),
    'transformer.h.0.attn.c_attn.lora_B.default.weight': Tensor([2304, 8]),
    'transformer.h.0.attn.c_proj.lora_A.default.weight': Tensor([8, 768]),
    'transformer.h.0.attn.c_proj.lora_B.default.weight': Tensor([768, 8]),
    # ...
}
```

**After Filter 2 (Keep only adapter_name="default"):**

```python
# If there were multiple adapters ("default", "task_A"), only "default" remains
{
    'transformer.h.0.attn.c_attn.lora_A.default.weight': Tensor([8, 768]),
    'transformer.h.0.attn.c_attn.lora_B.default.weight': Tensor([2304, 8]),
    # ...
}
```

**After Filter 3 (Remove adapter name):**

```python
{
    # ".default" removed for portability!
    'transformer.h.0.attn.c_attn.lora_A.weight': Tensor([8, 768]),
    'transformer.h.0.attn.c_attn.lora_B.weight': Tensor([2304, 8]),
    'transformer.h.0.attn.c_proj.lora_A.weight': Tensor([8, 768]),
    'transformer.h.0.attn.c_proj.lora_B.weight': Tensor([768, 8]),
    # ...
}
```

**Final saved state dict:** Only adapter parameters, adapter name stripped!

---

## Code Deep Dive

### Critical Functions

| **Function** | **File** | **Line** | **Purpose** |
|--------------|----------|----------|-------------|
| `get_peft_model_state_dict()` | `src/peft/utils/save_and_load.py` | 56-84 | Extracts adapter-only state dict |
| `PeftModel.save_pretrained()` | `src/peft/peft_model.py` | 189-386 | Saves adapter + config to disk |
| `load_peft_weights()` | `src/peft/utils/save_and_load.py` | ~450+ | Loads saved adapter weights |
| `PeftModel.from_pretrained()` | `src/peft/peft_model.py` | 388-500 | Loads adapter from checkpoint |

### Step-by-Step Code Breakdown

#### Step 1: User Calls `save_pretrained()`

**Location:** `src/peft/peft_model.py:189-198`

```python
peft_model.save_pretrained(
    save_directory="./my_adapter",
    safe_serialization=True,  # Use safetensors format
    selected_adapters=None,  # Save all adapters
    save_embedding_layers="auto",
)
```

#### Step 2: Extract Adapter Weights

**Location:** `src/peft/peft_model.py:292-297`

```python
for adapter_name in selected_adapters:
    peft_config = self.peft_config[adapter_name]

    # THIS IS THE KEY FUNCTION!
    output_state_dict = get_peft_model_state_dict(
        self,
        state_dict=kwargs.get("state_dict", None),
        adapter_name=adapter_name,
        save_embedding_layers=save_embedding_layers,
    )
```

**What `get_peft_model_state_dict()` does:**

**Location:** `src/peft/utils/save_and_load.py:56-132`

```python
def get_peft_model_state_dict(model, state_dict=None, adapter_name="default", ...):
    # Get full model state dict if not provided
    if state_dict is None:
        state_dict = model.state_dict()  # Contains base + adapter weights

    config = model.peft_config[adapter_name]

    # LoRA/AdaLoRA specific filtering
    if config.peft_type in (PeftType.LORA, PeftType.ADALORA):
        bias = config.bias

        if bias == "none":
            # Only LoRA matrices, no biases
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k}

        elif bias == "all":
            # LoRA matrices + all biases
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k}

        elif bias == "lora_only":
            # LoRA matrices + biases of adapted layers only
            to_return = {}
            for k in state_dict:
                if "lora_" in k:
                    to_return[k] = state_dict[k]
                    # Also include bias of this layer
                    bias_name = k.split("lora_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]

        # Filter by adapter name
        to_return = {
            k: v for k, v in to_return.items()
            if (("lora_" in k and adapter_name in k) or ("bias" in k))
        }

        # Handle DoRA refactoring (lora_magnitude_vector)
        if config.use_dora:
            # Remove ".weight" suffix from magnitude vectors
            # for backward compatibility
            # ...

    return to_return
```

**Key points:**
1. Starts with full `model.state_dict()` (base + adapters)
2. Filters to keep only keys containing `"lora_"`
3. Further filters by `adapter_name` for multi-adapter models
4. Handles bias according to config
5. Returns adapter-only dict

#### Step 3: Handle Tensor Aliasing (Safetensors)

**Location:** `src/peft/peft_model.py:301-322`

```python
if safe_serialization:
    # Safetensors doesn't allow tensor aliasing (shared storage)
    # Find all tensors sharing storage
    ptrs = collections.defaultdict(list)
    for name, tensor in output_state_dict.items():
        if isinstance(tensor, torch.Tensor):
            ptrs[id_tensor_storage(tensor)].append(name)

    # Identify shared tensors
    shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}

    # Clone shared tensors to avoid aliasing
    for _, names in shared_ptrs.items():
        for shared_tensor_name in names[1:]:
            output_state_dict[shared_tensor_name] = output_state_dict[shared_tensor_name].clone()
```

**Why needed?**
- Safetensors format doesn't support tensor aliasing
- Some parameters may share underlying storage
- Cloning ensures each parameter has unique storage

#### Step 4: Save to Disk

**Location:** `src/peft/peft_model.py:330-343`

```python
if safe_serialization:
    # Save as safetensors (recommended)
    safe_save_file(
        output_state_dict,
        os.path.join(output_dir, "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )
else:
    # Save as PyTorch .bin
    torch.save(
        output_state_dict,
        os.path.join(output_dir, "adapter_model.bin")
    )
```

#### Step 5: Save Configuration

**Location:** `src/peft/peft_model.py:384`

```python
peft_config.save_pretrained(output_dir)
# Saves adapter_config.json
```

---

## File Format and Structure

### Saved Files

```
my_adapter/
├── adapter_config.json       # Configuration
└── adapter_model.safetensors # Weights (or adapter_model.bin)
```

### `adapter_config.json`

**Location:** Created by `LoraConfig.save_pretrained()`

```json
{
    "peft_type": "LORA",
    "auto_mapping": null,
    "base_model_name_or_path": "gpt2",
    "revision": null,
    "task_type": "CAUSAL_LM",
    "inference_mode": true,

    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["c_attn", "c_proj"],
    "lora_dropout": 0.1,
    "fan_in_fan_out": false,
    "bias": "none",
    "use_rslora": false,
    "use_dora": false,
    "modules_to_save": null,
    "init_lora_weights": true,
    "layers_to_transform": null,
    "layers_pattern": null
}
```

### `adapter_model.safetensors`

**Binary format containing:**

```
Metadata:
    format: "pt"

Tensors:
    transformer.h.0.attn.c_attn.lora_A.weight: [8, 768] float32
    transformer.h.0.attn.c_attn.lora_B.weight: [2304, 8] float32
    transformer.h.0.attn.c_proj.lora_A.weight: [8, 768] float32
    transformer.h.0.attn.c_proj.lora_B.weight: [768, 8] float32
    ... (for all 48 layers)

Total size: ~3 MB
```

**Safetensors advantages:**
- Faster loading than pickle/PyTorch .bin
- Safer (no arbitrary code execution)
- Better cross-platform support
- Lazy loading support

---

## Loading Saved Adapters

### Loading Process

```python
# Step 1: Load base model (once)
base_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Step 2: Load adapter
peft_model = PeftModel.from_pretrained(
    base_model,
    "my_adapter",  # Path to saved adapter
    adapter_name="my_task"
)
```

### What Happens Internally

**Location:** `src/peft/peft_model.py:388-500`

```python
@classmethod
def from_pretrained(cls, model, model_id, adapter_name="default", ...):
    # 1. Load config
    config = PeftConfig.from_pretrained(model_id)
    # Reads adapter_config.json

    # 2. Create PeftModel with empty adapters
    peft_model = cls(model, config, adapter_name)
    # Injects LoRA layers with random initialization

    # 3. Load adapter weights
    adapters_weights = load_peft_weights(model_id)
    # Reads adapter_model.safetensors

    # 4. Set loaded weights
    set_peft_model_state_dict(peft_model, adapters_weights, adapter_name)
    # Copies loaded weights into adapter layers

    return peft_model
```

### Multi-Adapter Loading

```python
# Load base model
base_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Load first adapter
peft_model = PeftModel.from_pretrained(base_model, "adapter_task_A", adapter_name="task_A")

# Load second adapter (into same model!)
peft_model.load_adapter("adapter_task_B", adapter_name="task_B")

# Now model has both adapters!
# peft_model.lora_A = {"task_A": ..., "task_B": ...}

# Switch between adapters
peft_model.set_adapter("task_A")  # Use task_A
peft_model.set_adapter("task_B")  # Use task_B
```

---

## Summary

### Key Takeaways

1. **Filtering Mechanism:**
   - Extract only adapter parameters from full state dict
   - Filter by adapter type markers (`"lora_"`, `"ia3_"`, etc.)
   - Filter by adapter name for multi-adapter models
   - Strip adapter name for portability

2. **Storage Efficiency:**
   - GPT-2 full model: ~500 MB
   - LoRA adapter (r=8): ~3 MB
   - **167x reduction!**

3. **Two Files Saved:**
   - `adapter_config.json`: Configuration
   - `adapter_model.safetensors`: Weights

4. **Loading:**
   - Load base model once
   - Load adapters as needed
   - Switch between adapters without reloading base

5. **Multi-Task Applications:**
   - 100 tasks with full fine-tuning: 50 GB
   - 100 tasks with PEFT: ~800 MB
   - **62x reduction!**

### Critical Code Locations

| **Operation** | **File** | **Function** | **Line Range** |
|---------------|----------|--------------|----------------|
| Extract adapter weights | `src/peft/utils/save_and_load.py` | `get_peft_model_state_dict()` | 56-132 |
| Save to disk | `src/peft/peft_model.py` | `save_pretrained()` | 189-386 |
| Load from disk | `src/peft/peft_model.py` | `from_pretrained()` | 388-500 |
| Load additional adapter | `src/peft/peft_model.py` | `load_adapter()` | ~1290+ |

---

**Next Steps:**
- See `01_model_injection_deep_dive.md` for how adapters are injected
- See `../annotated_code/03_adapter_saving_annotated.py` for detailed code annotations
- See `../references/lora_config_reference.md` for all configuration options
