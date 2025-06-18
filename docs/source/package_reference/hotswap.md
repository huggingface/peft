<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Hotswapping adapters

The idea of hotswapping an adapter is the following: We can already load multiple adapters, e.g. two LoRAs, at the same time. But sometimes, we want to load one LoRA and then replace its weights in-place with the LoRA weights of another adapter. This is now possible the `hotswap_adapter` function.

In general, this should be faster than deleting one adapter and loading the adapter in its place, which would be the how to achieve the same final outcome without hotswapping. Another advantage of hotswapping is that it prevents re-compilation in case the PEFT model is already compiled using `torch.compile`. This can save quite a lot of time.

## Example without `torch.compile`

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from peft.utils.hotswap import hotswap_adapter

model_id = ...
inputs = ...
device = ...
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# load lora 0
model = PeftModel.from_pretrained(model, <path-adapter-0>)
with torch.inference_mode():
    output_adapter_0 = model(inputs)

# replace the "default" lora adapter with the new one
hotswap_adapter(model, <path-adapter-1>, adapter_name="default", torch_device=device)
with torch.inference_mode():
    output_adapter_1 = model(inputs).logits
```

## Example with `torch.compile`

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from peft.utils.hotswap import hotswap_adapter, prepare_model_for_compiled_hotswap

model_id = ...
inputs = ...
device = ...
max_rank = ...  # maximum rank among all LoRA adapters that will be used
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# load lora 0
model = PeftModel.from_pretrained(model, <path-adapter-0>)
# Prepare the model to allow hotswapping even if ranks/scalings of 2nd adapter differ.
# You can skip this step if all ranks and scalings are identical.
prepare_model_for_compiled_hotswap(model, target_rank=max_rank)
model = torch.compile(model)
with torch.inference_mode():
    output_adapter_0 = model(inputs)

# replace the "default" lora adapter with the new one
hotswap_adapter(model, <path-adapter-1>, adapter_name="default", torch_device=device)
with torch.inference_mode():
    output_adapter_1 = model(inputs).logits
```

## Caveats

Hotswapping works with transformers models and diffusers models. However, there are some caveats:

- Right now, only LoRA is properly supported.
- It only works for the same PEFT method, so no swapping LoRA and LoHa, for example.
- The adapter that is being swapped in must target the same layers as the previous adapter or a subset of those layers. It cannot target new layers. Therefore, if possible, start with the adapter that targets most layers.

[[autodoc]] utils.hotswap.hotswap_adapter
    - all

[[autodoc]] utils.hotswap.hotswap_adapter_from_state_dict
    - all
