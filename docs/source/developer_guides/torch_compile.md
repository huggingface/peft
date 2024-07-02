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

# torch.compile

In PEFT, [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) works for some but not all features. The reason why it won't always work is because PEFT is highly dynamic in certain places (loading and switching between multiple adapters, for instance), which can cause trouble for `torch.compile`. In other places, `torch.compile` may work, but won't be as fast as expected because of graph breaks.

If you don't see an error, it doesn't necessarily mean that `torch.compile` worked correctly. It might give you an output, but the output is incorrect. This guide describes what works with `torch.compile` and what doesn't.

> [!TIP]
> Unless indicated otherwise, the default `torch.compile` settings were used.

## Training and inference with `torch.compile`

These features **work** with `torch.compile`. Everything listed below was tested with a causal LM:

- Training with `Trainer` from 🤗 transformers
- Training with a custom PyTorch loop
- Inference
- Generation

The following adapters were tested successfully:

- AdaLoRA
- BOFT
- IA³
- Layer Norm Tuning
- LoHa
- LoRA
- LoRA + DoRA
- OFT
- VeRA
- HRA

The following adapters **don't work** correctly for training or inference when using `torch.compile`:

- LoKr
- LoRA targeting embedding layers

## Advanced PEFT features with `torch.compile`

Below are some of the more advanced PEFT features that **work**. They were all tested with LoRA.

- `modules_to_save` (i.e. `config = LoraConfig(..., modules_to_save=...)`)
- Merging adapters (one or multiple)
- Merging multiple adapters into one adapter (i.e. calling `model.add_weighted_adapter(...)`)

Generally, we can expect that if a feature works correctly with LoRA and is also supported by other adapter types, it should also work for that adapter type.

The more advanced PEFT features below **don't work** in conjunction with `torch.compile`. Tests were run with LoRA:

- Using PEFT adapters with quantization (bitsandbytes)
- Inference with multiple adapters
- Unloading (i.e. calling `model.merge_and_unload()`)
- Disabling adapters (i.e. using `with model.disable_adapter()`)
- Mixed adapter batches (i.e. calling `model(batch, adapter_names=["__base__", "default", "other", ...])`)

## Test cases

All the use cases listed above are tested inside of [`peft/tests/test_torch_compile.py`](https://github.com/huggingface/peft/blob/main/tests/test_torch_compile.py). If you want to check in more detail how we tested a certain feature, please go to that file and check the test that corresponds to your use case.

> [!TIP]
> If you have another use case where you know that `torch.compile` does or does not work with PEFT, please contribute by letting us know or by opening a PR to add this use case to the covered test cases.
