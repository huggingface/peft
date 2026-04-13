# KappaTuneSelector

**Automatic LoRA target module selection based on matrix condition numbers (κ)**

`KappaTuneSelector` implements the condition-number-based target selection strategy from the [KappaTune paper](https://arxiv.org/abs/2506.16289).  
It scans every `nn.Linear` module, computes its matrix condition number **κ = σ_max / σ_min**, and selects the most isotropic layers (lowest κ). These layers are the most flexible for LoRA adaptation and help mitigate catastrophic forgetting on downstream datasets.

The selector fully supports **4-bit and int8 quantized models** (bitsandbytes).

## Quick one-liner (recommended)

```python
from peft.utils.target_selection import find_kappa_target_modules

target_modules = find_kappa_target_modules(model, top_p=0.2)
```

## API reference

::: peft.utils.target_selection.KappaTuneSelector
    options:
      heading_level: 3

::: peft.utils.target_selection.find_kappa_target_modules
