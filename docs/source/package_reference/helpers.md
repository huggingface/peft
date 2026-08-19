<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Helper methods

A collection of helper functions for PEFT.

## Checking if a model is a PEFT model

[[autodoc]] helpers.check_if_peft_model
    - all

## Temporarily Rescaling Adapter Scale in LoraLayer Modules

[[autodoc]] helpers.rescale_adapter_scale
    - all

## Context manager to disable input dtype casting in the `forward` method of LoRA layers

[[autodoc]] helpers.disable_input_dtype_casting
    - all

## Context manager to enable DoRA caching (faster at inference time but requires more memory)

[[autodoc]] helpers.DoraCaching
    - all

## KappaTune target selection

`KappaTuneSelector` and `find_kappa_target_modules` implement a general target selection process from the [KappaTune paper](https://arxiv.org/abs/2506.16289). 

The method identifies modules with higher flexibility (higher output differential entropy) and lower specialization (lower sensitivity to specific input directions).

These properties make the selected modules good candidates for mitigating catastrophic forgetting in any adaptation method that adds trainable parameters, including LoRA, DoRA, LoHa, AdaLoRA, and even direct fine-tuning of the original weights.

[[autodoc]] helpers.KappaTuneSelector
    - all

[[autodoc]] helpers.find_kappa_target_modules
    - all

## Emulator model

`get_emulator_model` constructs a lightweight *emulator* of a model by replacing `nn.Linear` layers with low-rank
SVD factorizations, following the [EMLoC paper](https://arxiv.org/abs/2506.12015) (Lin et al., NeurIPS 2025). Each
linear layer is replaced by two sequential linear layers whose weights are derived from a truncated SVD of the
original weight. When calibration data is provided, the SVD is made *activation-aware* to preserve task-relevant
directions.

The `rank` argument controls the compression:

- `int`: use a fixed rank for all layers.
- `float`: interpreted as an energy threshold in `(0, 1]` — for each layer, the smallest rank `k` is chosen such
  that the top-`k` singular values account for at least that fraction of the total squared singular values (same
  logic as [`peft.tuners.lora.conversion`]).

Higher ranks yield closer approximations to the original model at the cost of more parameters.

```python
>>> from peft import get_emulator_model
>>> from transformers import AutoModelForCausalLM
>>>
>>> model = AutoModelForCausalLM.from_pretrained("gpt2")
>>> # Using a fixed rank
>>> emulator = get_emulator_model(model, rank=4)
>>> # Using an energy threshold (activation-aware, with calibration data)
>>> emulator = get_emulator_model(model, rank=0.95, data_loader=loader)
```

[[autodoc]] helpers.get_emulator_model
    - all
