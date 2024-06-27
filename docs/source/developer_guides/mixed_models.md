<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Mixed adapter types

Normally, it isn't possible to mix different adapter types in 🤗 PEFT. You can create a PEFT model with two different LoRA adapters (which can have different config options), but it is not possible to combine a LoRA and LoHa adapter. With [`PeftMixedModel`] however, this works as long as the adapter types are compatible. The main purpose of allowing mixed adapter types is to combine trained adapters for inference. While it is possible to train a mixed adapter model, this has not been tested and is not recommended.

To load different adapter types into a PEFT model, use [`PeftMixedModel`] instead of [`PeftModel`]:

```py
from peft import PeftMixedModel

base_model = ...  # load the base model, e.g. from transformers
# load first adapter, which will be called "default"
peft_model = PeftMixedModel.from_pretrained(base_model, <path_to_adapter1>)
peft_model.load_adapter(<path_to_adapter2>, adapter_name="other")
peft_model.set_adapter(["default", "other"])
```

The [`~PeftMixedModel.set_adapter`] method is necessary to activate both adapters, otherwise only the first adapter would be active. You can keep adding more adapters by calling [`~PeftModel.add_adapter`] repeatedly.

[`PeftMixedModel`] does not support saving and loading mixed adapters. The adapters should already be trained, and loading the model requires a script to be run each time.

## Tips

- Not all adapter types can be combined. See [`peft.tuners.mixed.COMPATIBLE_TUNER_TYPES`](https://github.com/huggingface/peft/blob/1c1c7fdaa6e6abaa53939b865dee1eded82ad032/src/peft/tuners/mixed/model.py#L35) for a list of compatible types. An error will be raised if you try to combine incompatible adapter types.
- It is possible to mix multiple adapters of the same type which can be useful for combining adapters with very different configs.
- If you want to combine a lot of different adapters, the most performant way to do it is to consecutively add the same adapter types. For example, add LoRA1, LoRA2, LoHa1, LoHa2 in this order, instead of LoRA1, LoHa1, LoRA2, and LoHa2. While the order can affect the output, there is no inherently *best* order, so it is best to choose the fastest one.
