<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Working with mixed adapter types

Normally, it is not possible to mix different adapter types in 🤗 PEFT. For example, even though it is possible to create a PEFT model that has two different LoRA adapters (that can have different config options), it is not possible to combine a LoRA adapter with a LoHa adapter. However, by using a mixed model, this works as long as the adapter types are compatible.

## Loading different adapter types into a PEFT model

To load different adapter types into a PEFT model, proceed the same as if you were loading two adapters of the same type, but use `PeftMixedModel` instead of `PeftModel`:

```py
from peft import PeftMixedModel

base_model = ...  # load the base model, e.g. from transformers
# load first adapter, which will be called "default"
peft_model = PeftMixedModel.from_pretrained(base_model, <path_to_adapter1>)
peft_model.load_adapter(<path_to_adapter2>, adapter_name="other")
peft_model.set_adapter(["default", "other"])
```

The last line is necessary if you want to activate both adapters, otherwise, only the first adapter would be active. Of course, you can add more different adapters by calling `add_adapter` repeatedly.

Currently, the main purpose of mixed adapter types is to combine trained adapters for inference. Although it is technically also possible to train a mixed adapter model, this has not been tested and is not recommended.

## Tips

- Not all adapter types can be combined. See `peft.tuners.mixed.COMPATIBLE_TUNER_TYPES` for a list of compatible types. An error will be raised if you are trying to combine incompatible adapter types.
- It is possible to mix multiple adapters of the same type. This can be useful to combine adapters with very different configs.
- If you want to combine a lot of different adapters, it is most performant to add the same types of adapters consecutively. E.g., add LoRA1, LoRA2, LoHa1, LoHa2 in this order, instead of LoRA1, LoHa1, LoRA2, LoHa2. The order will make a difference for the outcome in most cases, but since no order is better a priori, it is best to choose the order that is most performant.
