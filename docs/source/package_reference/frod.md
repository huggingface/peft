<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# FRoD: Full-Rank Efficient Fine-Tuning with Rotational Degrees

FRoD is a parameter-efficient fine-tuning method that combines a shared full-rank basis with sparse learnable
rotational degrees. The adapter update is expressed through fixed projection tensors and trainable coefficients, which
allows FRoD to apply full-rank updates while keeping the number of trained parameters small.

Paper: [Full-Rank Efficient Fine-Tuning with Rotational Degrees](https://doi.org/10.1609/aaai.v40i31.39813).

When saving the adapter parameters, it is possible to avoid storing the projection tensors by setting
`save_projection=False` on the `FrodConfig`. In that case, the projections are restored from the base model weights and
the fixed random seed from `projection_prng_key`. This reduces checkpoint size, but the default is
`save_projection=True` to make checkpoint loading independent of regeneration details.

Compared to LoRA, FRoD can express a full-rank update in each adapted linear layer while training only the diagonal
coefficients and a sparse set of off-diagonal rotation coefficients. This can be useful when a low-rank update is too
restrictive. The trade-off is that FRoD computes fixed projection tensors from the base weights during adapter
injection, which makes setup more expensive and the implementation less broadly supported than LoRA.

Projection initialization can be slow on large models because FRoD runs matrix decompositions over the target module
categories before injecting the adapters. A progress bar is shown by default and can be disabled with
`FrodConfig(progressbar=False)`.

For memory-constrained training, `runtime_offload_base_weight=True` can move target base weights to CPU during active
FRoD forward passes that do not use dropout. This is opt-in because PEFT methods usually keep all base parameters on
the accelerator after loading and forward passes.

FRoD currently has the following constraint:

- Only `nn.Linear` and `transformers.pytorch_utils.Conv1D` layers are supported.

## Quickstart

```python
from transformers import AutoModelForSequenceClassification

from peft import FrodConfig, TaskType, get_peft_model

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=2)

peft_config = FrodConfig(
    task_type=TaskType.SEQ_CLS,
    target_modules=["query", "value"],
    modules_to_save=["classifier"],
    sparse_rate=0.02,
    frod_dropout=0.0,
    runtime_offload_base_weight=True,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

## FrodConfig

[[autodoc]] tuners.frod.config.FrodConfig

## FrodModel

[[autodoc]] tuners.frod.model.FrodModel
