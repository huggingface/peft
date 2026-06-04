<!--Copyright 2026-present The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Memory Efficient Training

🤗 PEFT provides you with methods for parameter efficient fine-tuning but that doesn't mean that your training process is memory efficient. This guide is a collection of tips that you can use to improve memory efficiency of your training process. This guide is mostly an overview page that will link you to the respective other guides and offer some tips for specific situations.

## Choosing the right method

Not every PEFT method is built equally and some formulations are easier to build in a memory efficient manner. If you are on a memory budget it makes sense to check out the [PEFT method comparison suite](https://huggingface.co/spaces/peft-internal-testing/PEFT-method-comparison) and filter for **maximum** accelerator memory usage. Average accelerator memory usage can be fairly equal across methods but not every method scales equally with activations and sequence length and is more prone to memory spikes than others.

## Chunked NLL loss

Using [`NLLLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html) is very common when training language models (or classification tasks, for that matter) but it is usually computed in one go, meaning you will allocate a matrix of size `batch × sequence × vocabulary`. With particularly long sequences or vocabularies this can get expensive fast.

When using [TRL] you can either use the [Liger kernel integration](https://huggingface.co/docs/trl/liger_kernel_integration) or use [Chunked NLLLoss](https://huggingface.co/docs/trl/v1.5.1/en/reducing_memory_usage#chunked-cross-entropy-for-reducing-peak-memory-usage). The latter will split the sequence in chunks of size 256 to keep the maximum memory consumption constant.

![NLL vs. Chunked NLL comparison](https://private-user-images.githubusercontent.com/45557362/585095978-246efe41-a013-4597-8b0b-b38b83c71bb5.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODA1NjkzMDUsIm5iZiI6MTc4MDU2OTAwNSwicGF0aCI6Ii80NTU1NzM2Mi81ODUwOTU5NzgtMjQ2ZWZlNDEtYTAxMy00NTk3LThiMGItYjM4YjgzYzcxYmI1LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA2MDQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNjA0VDEwMzAwNVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTRkMGQyZDMwMzkyMTk5MTE2MTBjM2U2OWM3OTUwZjZiMWYzOTJiOWY1OWYwM2RhZjFiYTVmZmE1OGE2ZmVhYmUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.nfeb11_ztlkQbvDwRGOl7vgbvskx_TgdyKeVE4CoYPg)

In case the default chunk size is not optimal for your setting, look in the [original TRL PR](https://github.com/huggingface/trl/pull/5575) for more information on how to tune the chunk size.

## Quantization

Quantization is one of the best ways to reduce memory consumption *of the base model* and will, depending on the employed quantization, also reduce activation memory. Since the PEFT methods will only take up a small portion of the total number of parameters, PEFT defaults to use a higher precision than the base model. This can also have the effect that adapters can mitigate some of the quality loss incured by quantization methods. Read the [PEFT quantization guide](quantization).

## Compilation

The models we train are composed of operations like matrix multiplications, sums and assignments where each operation produces a new result and, subsequently, needs to take up memory. If those intermediate results are not needed we can fuse these operations and save up on memory. This is just one of many optimizations that `torch.compile` can do for you, so check out the [PEFT torch.compile guide](torch_compile).

## Gradient Checkpointing

You can trade memory with computation by only saving every nth gradient between layers and computing the rest on the fly. Check out the [gradient checkpointing](https://huggingface.co/docs/transformers/grad_checkpointing) documentation of Transformers to learn more.

