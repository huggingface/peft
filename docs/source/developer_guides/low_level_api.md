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

With PEFT, you can inject trainable adapters into any `torch` module which allows you to use adapter methods without relying on the modeling classes in PEFT. Currently, PEFT supports injecting [LoRA](../conceptual_guides/adapter#low-rank-adaptation-lora), [AdaLoRA](../conceptual_guides/adapter#adaptive-low-rank-adaptation-adalora), and [IA3](../conceptual_guides/ia3) into models because for these adapters, inplace modification of the model is sufficient for finetuning it.

Check the table below to see when you should inject adapters.

| Pros | Cons |
|---|---|
| the model is modified inplace, keeping all the original attributes and methods | manually write the `from_pretrained` and `save_pretrained` utility functions from Hugging Face to save and load adapters |
| works for any `torch` module and modality | doesn't work with any of the utility methods provided by `PeftModel` such as disabling and merging adapters |

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

To only save the adapter, use the [`get_peft_model_state_dict`] function:

```python
from peft import get_peft_model_state_dict

peft_state_dict = get_peft_model_state_dict(model)
print(peft_state_dict)
```

Otherwise, `model.state_dict()` returns the full state dict of the model.
