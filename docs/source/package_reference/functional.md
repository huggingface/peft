<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Functions for PEFT integration

A collection of functions that could be useful for non-PeftModel models, e.g. transformers or diffusers integration

The functions provided here can be considered "public API" of PEFT and hence are safe to be used by packages that provide PEFT integrations.

## Cast the adapter weight dtypes
[[autodoc]] functional.cast_adapter_dtype
    - all

## Delete the PEFT adapter from model
[[autodoc]] functional.delete_adapter
    - all

## Get the state dict of the PEFT adapter
[[autodoc]] functional.get_peft_model_state_dict
    - all

## Inject a PEFT adapter into the model based on a PEFT config
[[autodoc]] functional.inject_adapter_in_model
    - all

## Set the active PEFT adapter(s) of the model
[[autodoc]] functional.set_adapter
    - all

## Set the `requires_grad` attribute of the specified adapters
[[autodoc]] functional.set_requires_grad
    - all

## Load the weights of the PEFT state dict into the model
[[autodoc]] functional.set_peft_model_state_dict
    - all
