<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Adapter injection

With PEFT, you can inject trainable adapters into any `torch` module which allows you to use adapter methods without relying on the modeling classes in PEFT. This works for all adapters except for those based on prompt learning (e.g. prefix tuning or p-tuning).

Check the table below to see when you should inject adapters.

| Pros | Cons |
|---|---|
| the model is modified inplace, keeping all the original attributes and methods | manually write the `from_pretrained` and `save_pretrained` utility functions from Hugging Face to save and load adapters |
| works for any `torch` module and modality | doesn't work with any of the utility methods provided by `PeftModel` such as disabling and merging adapters |

## Creating a new PEFT model

To perform the adapter injection, use the [`inject_adapter_in_model`] method. This method takes 3 arguments, the PEFT config, the model, and an optional adapter name. You can also attach multiple adapters to the model if you call [`inject_adapter_in_model`] multiple times with different adapter names.

For example, to inject LoRA adapters into the `linear` submodule of the `DummyModel` module:

```python
import torch
from peft import inject_adapter_in_model, LoraConfig

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 10)
        self.linear = torch.nn.Linear(10, 10)
        self.lm_head = torch.nn.Linear(10, 10)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.linear(x)
        x = self.lm_head(x)
        return x


lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    target_modules=["linear"],
)

model = DummyModel()
model = inject_adapter_in_model(lora_config, model)

dummy_inputs = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]])
dummy_outputs = model(dummy_inputs)
```

Print the model to see that the adapters have been correctly injected.

```bash
DummyModel(
  (embedding): Embedding(10, 10)
  (linear): Linear(
    in_features=10, out_features=10, bias=True
    (lora_dropout): ModuleDict(
      (default): Dropout(p=0.1, inplace=False)
    )
    (lora_A): ModuleDict(
      (default): Linear(in_features=10, out_features=64, bias=False)
    )
    (lora_B): ModuleDict(
      (default): Linear(in_features=64, out_features=10, bias=False)
    )
    (lora_embedding_A): ParameterDict()
    (lora_embedding_B): ParameterDict()
  )
  (lm_head): Linear(in_features=10, out_features=10, bias=True)
)
```

### Injection based on a `state_dict`

Sometimes, it is possible that there is a PEFT adapter checkpoint but the corresponding PEFT config is not known for whatever reason. To inject the PEFT layers for this checkpoint, you would usually have to reverse-engineer the corresponding PEFT config, most notably the `target_modules` argument, based on the `state_dict` from the checkpoint. This can be cumbersome and error prone. To avoid this, it is also possible to call [`inject_adapter_in_model`] and pass the loaded `state_dict` as an argument:

```python
from safetensors.torch import load_file

model = ...
state_dict = load_file(<path-to-safetensors-file>)
lora_config = LoraConfig(...)
model = inject_adapter_in_model(lora_config, model, state_dict=state_dict)
```

In this case, PEFT will use the `state_dict` as reference for which layers to target instead of using the PEFT config. As a user, you don't have to set the exact `target_modules` of the PEFT config for this to work. However, you should still pass a PEFT config of the right type, in this example `LoraConfig`, you can leave the `target_modules` as `None`.

Be aware that this still only creates the uninitialized PEFT layers, the values from the `state_dict` are not used to populate the model weights. To populate the weights, proceed with calling [`set_peft_model_state_dict`] as described below.

⚠️ Note that if there is a mismatch between what is configured in the PEFT config and what is found in the `state_dict`, PEFT will warn you about this. You can ignore the warning if you know that the PEFT config is not correctly specified.

> [!WARNING]
> If the original PEFT adapters was using `target_parameters` instead of `target_modules`, injecting from a `state_dict` will not work correctly. In this case, it is mandatory to use the correct PEFT config for injection.

## Saving the model

To only save the adapter, use the [`get_peft_model_state_dict`] function:

```python
from peft import get_peft_model_state_dict

peft_state_dict = get_peft_model_state_dict(model)
print(peft_state_dict)
```

Otherwise, `model.state_dict()` returns the full state dict of the model.

## Loading the model

After loading the saved `state_dict`, it can be applied using the [`set_peft_model_state_dict`] function:

```python
from peft import set_peft_model_state_dict

model = DummyModel()
model = inject_adapter_in_model(lora_config, model)
outcome = set_peft_model_state_dict(model, peft_state_dict)
# check that there were no wrong keys
print(outcome.unexpected_keys)
```

If injecting the adapter is slow or you need to load a large number of adapters, you may use an optimization that allows to create an "empty" adapter on meta device and only fills the weights with real weights when the [`set_peft_model_state_dict`] is called. To do this, pass `low_cpu_mem_usage=True` to both [`inject_adapter_in_model`] and [`set_peft_model_state_dict`].

```python
model = DummyModel()
model = inject_adapter_in_model(lora_config, model, low_cpu_mem_usage=True)

print(model.linear.lora_A["default"].weight.device.type == "meta")  # should be True
set_peft_model_state_dict(model, peft_state_dict, low_cpu_mem_usage=True)
print(model.linear.lora_A["default"].weight.device.type == "cpu")  # should be True
```
