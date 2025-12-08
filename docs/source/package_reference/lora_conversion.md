<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# LoRA conversion

Functions that allow to convert non-LoRA PEFT models to LoRA models.

## Convert a non-LoRA model to a LoRA model, return the `LoraConfig` and `state_dict`

[[autodoc]] tuners.lora.conversion.convert_to_lora
    - all

## Convert a non-LoRA model to a LoRA model, save the adapter checkpoint and config at the given path

[[autodoc]] tuners.lora.conversion.save_as_lora
    - all
