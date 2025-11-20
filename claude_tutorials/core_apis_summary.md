# Core APIs and Implementation Files Summary

## Overview

This document lists the 8 most critical user-facing APIs and 8 key adapter implementation files identified during the PEFT architecture analysis.

---

## Phase 1.1: 8 Most Critical User-Facing APIs

These are the functions and classes that users interact with directly when using PEFT.

### 1. `get_peft_model()`
**Location**: `src/peft/mapping_func.py:30-128`

**What**: Main entry point for applying PEFT to a model

**Signature**:
```python
def get_peft_model(
    model: PreTrainedModel,
    peft_config: PeftConfig,
    adapter_name: str = "default",
    mixed: bool = False,
    autocast_adapter_dtype: bool = True,
    revision: Optional[str] = None,
    low_cpu_mem_usage: bool = False,
) -> PeftModel | PeftMixedModel
```

**Usage**:
```python
from peft import get_peft_model, LoraConfig

config = LoraConfig(r=8, target_modules=["q_proj", "v_proj"])
peft_model = get_peft_model(model, config)
```

**See Also**: `annotated_code/01_get_peft_model_annotated.py`

---

### 2. `LoraConfig`
**Location**: `src/peft/tuners/lora/config.py:250-799`

**What**: Configuration class for LoRA adapters

**Key Parameters**:
```python
@dataclass
class LoraConfig(PeftConfig):
    r: int = 8                                    # Rank
    lora_alpha: int = 8                          # Scaling factor
    target_modules: Optional[Union[List[str], str]] = None
    lora_dropout: float = 0.0
    bias: Literal["none", "all", "lora_only"] = "none"
    use_rslora: bool = False
    use_dora: bool = False
    init_lora_weights: Union[bool, str] = True
    # ... 25+ more parameters
```

**Usage**:
```python
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)
```

**See Also**: `references/lora_config_reference.md`

---

### 3. `PeftModel`
**Location**: `src/peft/peft_model.py:71-1500+`

**What**: Wrapper class that adds adapter functionality to base models

**Key Methods**:
- `__init__()`: Initialize PEFT model with adapters
- `save_pretrained()`: Save adapter weights to disk
- `from_pretrained()`: Load saved adapters
- `load_adapter()`: Load additional adapters
- `set_adapter()`: Switch between adapters
- `merge_adapter()`: Merge adapter weights into base model
- `unload()`: Remove adapters and restore base model

**Usage**:
```python
# Saving
peft_model.save_pretrained("my_adapter")

# Loading
from peft import PeftModel
peft_model = PeftModel.from_pretrained(base_model, "my_adapter")

# Multi-adapter
peft_model.load_adapter("adapter_2", adapter_name="task_b")
peft_model.set_adapter("task_b")
```

**See Also**: `guides/02_adapter_saving_deep_dive.md`

---

### 4. `PeftModel.save_pretrained()`
**Location**: `src/peft/peft_model.py:189-386`

**What**: Saves adapter weights and configuration to directory

**Signature**:
```python
def save_pretrained(
    self,
    save_directory: str,
    safe_serialization: bool = True,
    selected_adapters: Optional[list[str]] = None,
    save_embedding_layers: Union[str, bool] = "auto",
    is_main_process: bool = True,
    path_initial_model_for_weight_conversion: Optional[str] = None,
    **kwargs: Any,
) -> None
```

**What It Saves**:
- `adapter_config.json`: Configuration
- `adapter_model.safetensors`: Adapter weights (or `.bin`)

**Usage**:
```python
peft_model.save_pretrained(
    "./my_adapter",
    safe_serialization=True,  # Use safetensors
)
```

**See Also**: `annotated_code/03_adapter_saving_annotated.py`

---

### 5. `PeftModel.from_pretrained()`
**Location**: `src/peft/peft_model.py:388-500+`

**What**: Loads saved adapter into a base model

**Signature**:
```python
@classmethod
def from_pretrained(
    cls,
    model: torch.nn.Module,
    model_id: Union[str, os.PathLike],
    adapter_name: str = "default",
    is_trainable: bool = False,
    config: Optional[PeftConfig] = None,
    autocast_adapter_dtype: bool = True,
    low_cpu_mem_usage: bool = False,
    **kwargs: Any,
)
```

**Usage**:
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("gpt2")
peft_model = PeftModel.from_pretrained(base_model, "./my_adapter")
```

---

### 6. `get_peft_model_state_dict()`
**Location**: `src/peft/utils/save_and_load.py:56-228`

**What**: Extracts only adapter parameters from full model state dict

**Signature**:
```python
def get_peft_model_state_dict(
    model,
    state_dict=None,
    adapter_name="default",
    unwrap_compiled=False,
    save_embedding_layers="auto"
)
```

**Usage**:
```python
from peft.utils import get_peft_model_state_dict

# Get adapter-only state dict (for manual saving)
adapter_state_dict = get_peft_model_state_dict(peft_model)
# Returns only keys with "lora_", "ia3_", etc.
```

**See Also**: `annotated_code/03_adapter_saving_annotated.py`

---

### 7. `inject_adapter_in_model()`
**Location**: `src/peft/mapping.py:47-93`

**What**: Lower-level API for in-place adapter injection without PeftModel wrapper

**Signature**:
```python
def inject_adapter_in_model(
    peft_config: PeftConfig,
    model: torch.nn.Module,
    adapter_name: str = "default",
    low_cpu_mem_usage: bool = False,
    state_dict: Optional[dict[str, torch.Tensor]] = None,
) -> torch.nn.Module
```

**Usage**:
```python
from peft import inject_adapter_in_model, LoraConfig

config = LoraConfig(r=8, target_modules=["q_proj"])
model = inject_adapter_in_model(config, model)
# Returns the mutated model directly (not wrapped in PeftModel)
```

**When to use**: Advanced use cases where you don't want PeftModel wrapper

---

### 8. `AutoPeftModel` / `AutoPeftModelForCausalLM`
**Location**: `src/peft/auto.py:27-185`

**What**: Auto classes for automatically loading correct PeftModel type

**Signature**:
```python
class AutoPeftModel:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        **kwargs: Any,
    )
```

**Usage**:
```python
from peft import AutoPeftModelForCausalLM

# Automatically loads base model + adapter
peft_model = AutoPeftModelForCausalLM.from_pretrained("./my_adapter")
# Infers base model from config and loads both
```

**Advantage**: One-line loading (doesn't require separate base model loading)

---

## Phase 1.2: 8 Key Adapter Implementation Files

These files contain the core logic for adapter injection and computation.

### 1. `src/peft/tuners/lora/layer.py`
**Lines**: 2,292 total

**What**: LoRA layer implementations (Linear, Conv2d, Embedding, etc.)

**Key Classes**:
- `LoraLayer` (base class): Lines 93-587
- `Linear` (most important): Lines 599-825
  - `forward()`: Lines 779-820 ⭐ **THE CORE COMPUTATION**
- `Embedding`: Lines 827-1040
- `Conv2d`: Lines 1042-1300+

**Critical Function**: `Linear.forward()` at line 779
```python
def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    # ...
    result = self.base_layer(x)  # W x + b
    # ...
    result = result + lora_B(lora_A(dropout(x))) * scaling
    # ...
    return result
```

**See Also**: `annotated_code/02_lora_forward_pass_annotated.py`

---

### 2. `src/peft/tuners/lora/model.py`
**Lines**: 850 total

**What**: LoraModel class that handles adapter injection

**Key Class**: `LoraModel(BaseTuner)`: Lines 69-850

**Key Methods**:
- `_prepare_model()`: Line 146 - Pre-injection model preparation
- `_create_and_replace()`: Lines 159-253 - Creates and replaces layers
- `_create_new_module()`: Lines 284-348 - Dispatcher for layer types
- `_replace_module()`: Lines 255-282 - Performs actual replacement
- `add_weighted_adapter()`: Lines 555-691 - Merge multiple adapters

**See Also**: `guides/01_model_injection_deep_dive.md`

---

### 3. `src/peft/tuners/lora/config.py`
**Lines**: 799 total

**What**: Configuration classes for LoRA

**Key Classes**:
- `LoraRuntimeConfig`: Lines 27-50
- `LoftQConfig`: Lines 54-70
- `ArrowConfig`: Lines 72-120
- `EvaConfig`: Lines 122-171
- `CordaConfig`: Lines 173-247
- `LoraConfig`: Lines 250-799 ⭐ **MAIN CONFIG**

**See Also**: `references/lora_config_reference.md`

---

### 4. `src/peft/tuners/tuners_utils.py`
**Lines**: 2,015 total

**What**: Base classes for all tuners

**Key Classes**:
- `BaseTunerLayer`: Base class for adapter layers
- `BaseTuner`: Lines 213-850+ - Base injection logic

**Key Methods**:
- `__init__()`: Lines 263-298 - Triggers injection
- `inject_adapter()`: ~Lines 650-850 - **MAIN INJECTION LOOP**
- `_check_target_module_exists()`: Lines 368-383 - Target validation
- `_prepare_adapter_config()`: Lines 325-351 - Config preparation
- `_create_and_replace()`: Abstract method (implemented by subclasses)

**See Also**: `guides/01_model_injection_deep_dive.md`

---

### 5. `src/peft/utils/save_and_load.py`
**Lines**: Extensive (multiple functions)

**What**: Utilities for saving and loading adapter weights

**Key Functions**:
- `get_peft_model_state_dict()`: Lines 56-228 - Extract adapter weights
- `set_peft_model_state_dict()`: Loads adapter weights
- `load_peft_weights()`: Loads from safetensors/bin
- `get_embedding_layer_name()`: Lines 48-53
- `has_valid_embedding_base_layer()`: Lines 43-45

**See Also**: `annotated_code/03_adapter_saving_annotated.py`

---

### 6. `src/peft/peft_model.py`
**Lines**: 3,330 total (largest file)

**What**: PeftModel wrapper and task-specific subclasses

**Key Classes**:
- `PeftModel`: Lines 71-1500+ - Base wrapper
- `PeftModelForCausalLM`: Causal language modeling
- `PeftModelForSeq2SeqLM`: Encoder-decoder models
- `PeftModelForSequenceClassification`: Classification
- `PeftModelForTokenClassification`: Token-level tasks
- `PeftModelForQuestionAnswering`: QA tasks

**Key Methods in PeftModel**:
- `__init__()`: Lines 103-144
- `save_pretrained()`: Lines 189-386
- `from_pretrained()`: Lines 388-500+
- `load_adapter()`: ~Line 1290
- `set_adapter()`: Sets active adapter
- `merge_and_unload()`: Merges adapters into base

---

### 7. `src/peft/mapping.py`
**Lines**: 93 total

**What**: Core mapping functions and registries

**Key Components**:
- `PEFT_TYPE_TO_CONFIG_MAPPING`: Maps PeftType to Config class
- `PEFT_TYPE_TO_TUNER_MAPPING`: Maps PeftType to Tuner class
- `PEFT_TYPE_TO_PREFIX_MAPPING`: Maps PeftType to prefix ("lora_", etc.)
- `inject_adapter_in_model()`: Lines 47-93 - Alternative injection

**Usage**: Used internally by `get_peft_model()` for routing

---

### 8. `src/peft/mapping_func.py`
**Lines**: 129 total

**What**: Contains the main `get_peft_model()` function

**See**: Entry #1 in User-Facing APIs section above

**See Also**: `annotated_code/01_get_peft_model_annotated.py`

---

## Quick Reference: File Paths

### User-Facing APIs
```
src/peft/
├── mapping_func.py         # get_peft_model()
├── peft_model.py          # PeftModel class
├── auto.py                # AutoPeftModel
└── mapping.py             # inject_adapter_in_model()
```

### LoRA Implementation
```
src/peft/tuners/lora/
├── config.py              # LoraConfig
├── model.py               # LoraModel (injection logic)
├── layer.py               # LoraLinear, LoraEmbedding (forward pass)
├── bnb.py                 # Quantized variants
├── gptq.py
├── awq.py
└── variants.py            # DoRA, aLoRA, Arrow
```

### Infrastructure
```
src/peft/
├── tuners/
│   └── tuners_utils.py    # BaseTuner (injection base class)
└── utils/
    ├── save_and_load.py   # get_peft_model_state_dict()
    └── other.py           # Utilities
```

---

## Summary Statistics

**User-Facing APIs**: 8
- 3 entry points (get_peft_model, inject_adapter_in_model, AutoPeftModel)
- 1 configuration class (LoraConfig)
- 1 wrapper class (PeftModel)
- 3 save/load functions

**Implementation Files**: 8
- 3 LoRA-specific (config, model, layer)
- 2 base infrastructure (tuners_utils, mapping)
- 1 wrapper (peft_model)
- 1 save/load utilities
- 1 entry point (mapping_func)

**Total Lines in Key Files**: ~10,000+ lines
**Most Important File**: `src/peft/tuners/lora/layer.py` (especially `forward()` at line 779)

---

**For detailed explanations, see**:
- [README.md](README.md) - Main tutorial navigation
- [annotated_code/](annotated_code/) - Line-by-line code annotations
- [guides/](guides/) - Conceptual deep dives
- [references/](references/) - Complete parameter documentation
