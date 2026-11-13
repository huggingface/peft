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

Note that if you want to hotswap weights that were added through `target_parameters`, i.e. that directly target an `nn.Parameter`, re-compilation and/or graph breaks cannot be prevented. Therefore, it is recommended to avoid using `target_parameters` together with compiled models and hotswapping.

## Starting with a dummy adapter

If none of your adapters targets every layer needed by the others, use [`helpers.create_dummy_lora_config`] to prepare a dummy adapter. Its initial contribution is zero, so you can serve base-model behavior before loading trained adapter weights. The helper only inspects configurations; it does not load the base model or adapter weights.

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftConfig, get_peft_model
from peft.helpers import create_dummy_lora_config
from peft.utils.hotswap import hotswap_adapter, prepare_model_for_compiled_hotswap

adapter_paths = ["path-to-adapter0", "path-to-adapter1"]
configs = [PeftConfig.from_pretrained(path) for path in adapter_paths]
dummy_config = create_dummy_lora_config(configs)
print("New combined LoraConfig:")
print(dummy_config.to_dict())

base_model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model = get_peft_model(base_model, dummy_config).eval()
# Ranks are already large enough; preparation also makes scalings safe to change after compilation.
prepare_model_for_compiled_hotswap(model)
model = torch.compile(model)

with torch.inference_mode():
    base_output = model(**inputs)

for path in adapter_paths:
    hotswap_adapter(model, path, adapter_name="default", torch_device=device)
    with torch.inference_mode():
        adapted_output = model(**inputs)
```

The dummy configuration takes the union of `target_modules` and the maximum of all `r` and `rank_pattern` values. It removes `exclude_modules`, `layers_to_transform`, and `layers_pattern` to cover all adapters. This can reserve more adapter memory than an exact union of the targeted layers. Modules absent from an incoming adapter are zeroed during hotswap. Keep the dummy adapter active during initial inference; disabling it changes the forward path and may trigger recompilation when it is enabled again.

The helper requires nonempty, explicit target-module lists or sets. String targets, including regexes and `"all-linear"`, and automatically inferred targets are unsupported. It supports standard LoRA and rsLoRA, with compatible hotswap settings (including matching dropout, rsLoRA, and alpha-pattern settings). Variants, extra saved modules, trained biases/tokens, direct parameter targets, and base-weight-changing initializations are rejected. All adapters must use the same base weights. Configuration checks cannot verify the model's actual layer types; the existing restrictions of compiled hotswapping still apply. For grouped convolutions, the maximum rank may need to be increased to a multiple of all targeted group counts before creating the model.

From a PEFT checkout, you can also save the configuration using:

```bash
python scripts/create-dummy-lora-for-hotswap.py path-to-adapter0 path-to-adapter1 --output-dir dummy-lora
```

Adapter arguments accept local directories or Hugging Face Hub repository IDs. The default output directory is `dummy-lora`. The script writes only `adapter_config.json`, not an adapter checkpoint. Load it with `LoraConfig.from_pretrained("dummy-lora")`, pass it to `get_peft_model`, and prepare the model as above. `PeftModel.from_pretrained` cannot load this directory because it has no adapter weights. If you need the dummy weights as well, call `model.save_pretrained`.

## Caveats

Hotswapping works with transformers models and diffusers models. However, there are some caveats:

- Right now, only LoRA is properly supported.
- It only works for the same PEFT method, so no swapping LoRA and LoHa, for example.
- The adapter that is being swapped in must target the same layers as the previous adapter or a subset of those layers. It cannot target new layers. Therefore, if possible, start with the adapter that targets most layers.
- Creation of a dummy LoRA config requires `target_parameters` to be a list of strings, regexes are not supported. A few other parameters of the `LoraConfig` are also not supported; if they are found, an error is raised. When targeting only some layer indices with `layers_to_transform` and `layers_pattern`, no attempt is made to find the smallest possible overlap, instead the dummy LoRA will be over-provisioned. The same is true for modules excluded via `exclude_modules`.

## API

[[autodoc]] utils.hotswap.hotswap_adapter
    - all

[[autodoc]] utils.hotswap.hotswap_adapter_from_state_dict
    - all
