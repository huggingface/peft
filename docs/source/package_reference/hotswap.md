<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Hotswapping adapters

The idea of hotswapping an adapter is the following: We can already load multiple adapters, e.g. two LoRAs, at the same time. But sometimes, we want to load one LoRA and then replace its weights in-place with the LoRA weights of another adapter. This is now possible the `hotswap_adapter` function.

In general, this should be faster than deleting one adapter and loading the adapter in its place, which would be the how to achieve the same final outcome without hotswapping. Another advantage of hotswapping is that it prevents re-compilation in case the PEFT model is already compiled using `torch.compile`. This can save quite a lot of time.

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
model = torch.compile(model)  # optionally compile the model
with torch.inference_mode():
    output_adapter_0 = model(inputs)

# replace the "default" lora adapter with the new one
hotswap_adapter(model, <path-adapter-1>, adapter_name="default", torch_device=device)
with torch.inference_mode():
    output_adapter_1 = model(inputs).logits
```

Hotswapping works with transformers models and diffusers models. However, there are some caveats:

- It only works for the same PEFT method, so no swapping LoRA and LoHa, for example.
- Right now, only LoRA is properly supported.
- The adapters must be compatible (e.g. same LoRA alpha, same target modules).
- If you use `torch.compile` and want to avoid recompilation, the LoRA rank must be the same.

[[autodoc]] utils.hotswap.hotswap_adapter
    - all

[[autodoc]] utils.hotswap.hotswap_adapter_from_state_dict
    - all
