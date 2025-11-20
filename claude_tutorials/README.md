# PEFT Adapter Architecture and Data Flow Tutorial

## Overview

This comprehensive tutorial deconstructs the mechanisms for adapter injection, configuration, and weight saving for LoRA and other PEFT methods. It provides heavily annotated code, detailed guides, and complete reference documentation.

**Created**: 2025
**Total Content**: ~5,000+ lines of annotated documentation
**Target Audience**: Developers wanting deep understanding of PEFT internals

---

## Quick Navigation

### 📚 Annotated Code (Deep Dives)
Start here for line-by-line explanations of critical functions:

1. **[get_peft_model() Entry Point](annotated_code/01_get_peft_model_annotated.py)**
   - The main user-facing function
   - Parameter validation and routing logic
   - ~250 lines with extensive annotations

2. **[LoRA Forward Pass](annotated_code/02_lora_forward_pass_annotated.py)** ⭐
   - **THE MOST IMPORTANT FILE**
   - Line-by-line breakdown of `result = W x + B(A x) * scaling`
   - Shows exactly where W, A, B matrices are accessed
   - Mathematical derivations and dimension examples
   - ~400 lines with computation breakdowns

3. **[Adapter Saving Mechanism](annotated_code/03_adapter_saving_annotated.py)**
   - How PEFT saves only adapter weights (not base model)
   - State dict filtering pipeline
   - Size reduction examples (500 MB → 3 MB)
   - ~350 lines with filtering logic

### 📖 Conceptual Guides
Read these for high-level understanding:

1. **[Model Injection Deep Dive](guides/01_model_injection_deep_dive.md)**
   - Complete injection pipeline from `get_peft_model()` to replaced layers
   - Step-by-step flow diagrams
   - Code mapping to 8 critical file locations
   - Layer replacement examples (nn.Linear → LoraLinear)
   - ~600 lines

2. **[Adapter Saving Deep Dive](guides/02_adapter_saving_deep_dive.md)**
   - How PEFT achieves 167x storage reduction
   - Three-step filtering mechanism
   - File format and structure
   - Loading process walkthrough
   - ~500 lines

### 📋 Reference Documentation

1. **[LoraConfig Complete Reference](references/lora_config_reference.md)**
   - All 30+ configuration parameters documented
   - Architectural impact analysis for each parameter
   - Parameter count calculations
   - Common configuration recipes
   - ~800 lines

2. **[Core APIs Summary](core_apis_summary.md)**
   - 8 most critical user-facing APIs
   - 8 key adapter implementation files
   - Quick reference with file locations

---

## Learning Paths

### Path 1: Quick Understanding (30 minutes)
**Goal**: Understand the big picture

1. Read [Core APIs Summary](core_apis_summary.md) (5 min)
2. Read [Model Injection Deep Dive](guides/01_model_injection_deep_dive.md) - focus on diagrams (10 min)
3. Read [LoRA Forward Pass](annotated_code/02_lora_forward_pass_annotated.py) - focus on "Key Takeaways" (10 min)
4. Skim [LoraConfig Reference](references/lora_config_reference.md) - focus on "Core Parameters" (5 min)

### Path 2: Implementation Deep Dive (2 hours)
**Goal**: Understand implementation details for contribution

1. Read all annotated code files in order (45 min)
2. Read both conceptual guides (45 min)
3. Review LoraConfig reference for parameter details (30 min)

### Path 3: Architecture Mastery (4+ hours)
**Goal**: Complete mastery for architecture development

1. Read all materials in this tutorial
2. Follow code paths in actual PEFT source files
3. Trace execution with debugger
4. Modify and experiment with configurations

---

## Key Concepts Covered

### 1. Model Injection
**What**: How adapters are inserted into a base model's architecture

**Key Insights**:
- Models are modified **in-place**, not copied
- `nn.Linear` layers replaced with `LoraLinear` layers
- Base weights wrapped (not duplicated)
- Process is reversible

**Files**:
- Guide: `guides/01_model_injection_deep_dive.md`
- Code: `annotated_code/01_get_peft_model_annotated.py`

### 2. LoRA Computation
**What**: How `W x + B(A x) * scaling` is computed

**Key Insights**:
- Base weights (W) stay frozen
- Adapter matrices (A, B) are trainable
- Computation is `result + lora_B(lora_A(dropout(x))) * scaling`
- Efficient: never compute `B @ A` explicitly

**Files**:
- Code: `annotated_code/02_lora_forward_pass_annotated.py` ⭐

### 3. Adapter Saving
**What**: How PEFT saves only 1-2% of model size

**Key Insights**:
- State dict filtered to keep only adapter parameters
- Adapter names stripped for portability
- 167x storage reduction for GPT-2
- Enables sharing hundreds of adapters

**Files**:
- Guide: `guides/02_adapter_saving_deep_dive.md`
- Code: `annotated_code/03_adapter_saving_annotated.py`

### 4. Configuration
**What**: 30+ parameters that control adapter behavior

**Key Insights**:
- `r` (rank): Primary capacity control
- `lora_alpha`: Scaling factor (doesn't affect param count)
- `target_modules`: Which layers to adapt
- Heterogeneous configs: different ranks per layer

**Files**:
- Reference: `references/lora_config_reference.md`

---

## Code Mapping: Critical Files

### User-Facing Entry Points

| File | Function/Class | Lines | Purpose |
|------|---------------|-------|---------|
| `src/peft/mapping_func.py` | `get_peft_model()` | 30-128 | Main entry point |
| `src/peft/peft_model.py` | `PeftModel` class | 71-1500+ | Model wrapper |
| `src/peft/peft_model.py` | `save_pretrained()` | 189-386 | Save adapters |
| `src/peft/peft_model.py` | `from_pretrained()` | 388-500 | Load adapters |

### Adapter Implementation (LoRA)

| File | Function/Class | Lines | Purpose |
|------|---------------|-------|---------|
| `src/peft/tuners/lora/config.py` | `LoraConfig` | 250-799 | Configuration |
| `src/peft/tuners/lora/model.py` | `LoraModel` | 69-850 | Injection logic |
| `src/peft/tuners/lora/layer.py` | `Linear` class | 599-825 | Layer implementation |
| `src/peft/tuners/lora/layer.py` | `forward()` method | 779-820 | **Core computation** |

### Infrastructure

| File | Function/Class | Lines | Purpose |
|------|---------------|-------|---------|
| `src/peft/tuners/tuners_utils.py` | `BaseTuner` | 213-850 | Base injection logic |
| `src/peft/utils/save_and_load.py` | `get_peft_model_state_dict()` | 56-228 | Extract adapter weights |
| `src/peft/mapping.py` | `inject_adapter_in_model()` | 47-93 | Alternative entry point |

---

## Examples and Calculations

### Parameter Count Examples

**GPT-2 (124M parameters)**
```
Base model: 124M params (frozen)

LoRA r=8:
- Per attention layer: 24,576 params
- 48 layers × 2 projections = ~1.2M params
- Percentage: 0.97% trainable

LoRA r=64:
- Per attention layer: 196,608 params
- 48 layers × 2 projections = ~9.4M params
- Percentage: 7.6% trainable
```

### Storage Examples

**Adapter Size Comparison**
```
Full fine-tuned GPT-2: 496 MB
LoRA adapter (r=8): 3 MB (167x smaller)
LoRA adapter (r=4): 1.5 MB (331x smaller)
LoRA adapter (r=64): 24 MB (21x smaller)

100 task-specific models:
- Full fine-tuning: 49.6 GB
- LoRA adapters: 300 MB (165x smaller!)
```

---

## Frequently Asked Questions

### Q: Where exactly are the LoRA matrices stored?
**A**: See `annotated_code/02_lora_forward_pass_annotated.py` lines 280-340
- Matrix A: `self.lora_A[adapter_name].weight` (shape: r × in_features)
- Matrix B: `self.lora_B[adapter_name].weight` (shape: out_features × r)
- Base W: `self.base_layer.weight` (frozen)

### Q: How does PEFT save only adapter weights?
**A**: See `annotated_code/03_adapter_saving_annotated.py` and `guides/02_adapter_saving_deep_dive.md`
- Filters `model.state_dict()` to keep only keys with "lora_"
- Further filters by adapter name
- Strips adapter name for portability
- Saves to `adapter_model.safetensors`

### Q: What's the difference between `r` and `lora_alpha`?
**A**: See `references/lora_config_reference.md` - Core Parameters section
- `r`: Rank, controls parameter count (higher = more params)
- `lora_alpha`: Scaling factor, controls output magnitude (doesn't affect param count)
- Typical: alpha = r or alpha = 2×r

### Q: Can I use different ranks for different layers?
**A**: Yes! See `references/lora_config_reference.md` - Architectural Modifications
- Use `rank_pattern` parameter
- Example: `{"^model.layers.0": 16, "^model.layers.1": 8}`

### Q: How is the forward pass computed?
**A**: See `annotated_code/02_lora_forward_pass_annotated.py` - The Core LoRA Computation
```python
result = base_layer(x) + lora_B(lora_A(dropout(x))) * scaling
```
- First: `base_layer(x)` computes `W x + b`
- Then: `lora_A(dropout(x))` projects to rank-r space
- Then: `lora_B(...)` projects back to output space
- Finally: Scale and add to base output

---

## Prerequisites

**To understand this tutorial, you should know**:
- Python and PyTorch basics
- Neural network fundamentals (linear layers, forward pass)
- Basic linear algebra (matrix multiplication)
- Git and Python package structure

**Optional but helpful**:
- Transformers library familiarity
- Fine-tuning concepts
- Low-rank matrix approximation

---

## Contributing

Found an error or want to add content?

1. Open an issue describing the problem/addition
2. Fork the repository
3. Make changes following the annotation style
4. Submit a pull request

**Annotation Style Guidelines**:
- Use `===` separators for major sections
- Use `---` for minor sections
- Include "WHAT/HOW/WHY" for every major operation
- Provide concrete examples with dimensions
- Reference source code line numbers
- Add mathematical derivations where relevant

---

## Acknowledgments

This tutorial is based on the PEFT library by HuggingFace:
- Repository: https://github.com/huggingface/peft
- Paper: LoRA (Hu et al., 2021) - https://arxiv.org/abs/2106.09685

---

## License

This tutorial follows the same Apache 2.0 license as the PEFT library.

---

## Changelog

**2025-01-XX**: Initial release
- 3 annotated code files
- 2 conceptual guides
- 1 complete configuration reference
- 5,000+ lines of documentation

---

## Quick Start Example

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

# 1. Load base model
model = AutoModelForCausalLM.from_pretrained("gpt2")
print(f"Base model params: {model.num_parameters():,}")

# 2. Create LoRA config
config = LoraConfig(
    r=8,                                    # Rank
    lora_alpha=16,                         # Scaling factor
    target_modules=["c_attn", "c_proj"],  # Which layers
    lora_dropout=0.1,                      # Dropout
    task_type="CAUSAL_LM"
)

# 3. Get PEFT model (injection happens here!)
peft_model = get_peft_model(model, config)
print(f"Trainable params: {peft_model.num_parameters(only_trainable=True):,}")

# 4. Train your model
# ... training code ...

# 5. Save adapter (only ~3 MB!)
peft_model.save_pretrained("my_adapter")

# 6. Load adapter later
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
model = PeftModel.from_pretrained(base_model, "my_adapter")
```

**To understand what happens in steps 3-6, read**:
- Step 3: `guides/01_model_injection_deep_dive.md`
- Step 4: `annotated_code/02_lora_forward_pass_annotated.py`
- Step 5-6: `guides/02_adapter_saving_deep_dive.md`

---

**Happy Learning! 🚀**
