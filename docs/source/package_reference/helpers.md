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
