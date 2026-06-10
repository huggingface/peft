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


# Parameter efficient fine-tuning methods

PEFT methods train as few parameters as possible while aiming for performance comparable to full fine-tuning. Fewer trainable parameters are less expressive, so the same performance isn't guaranteed. In exchange you use less memory, often less compute, and gain features like fast hot-swapping between expert adapters and less forgetting of prior knowledge.

Giving general advice for training large models is hard but for generative
models, especially language models, you can follow these steps:

1. use prompting (e.g. few-shot examples in the prompt) to see if the model is
   already capable of the task. If the model solves your problem, great! You can
   now use [Prompt-based methods](#prompt-based-methods) to learn the prompt and
   save precious tokens.
2. If prompt-based methods are not sufficient you can use [layer tuning](#layer-tuning)
   and [adapter methods](#adapter-methods). These methods are generally
   more expressive than prompt-based methods and get closer to full-finetuning.
3. Make sure to measure retention of already learnt knowledge since each
   fine-tuning step is potentially unlearning past knowledge.

The [PEFT method comparison suite](https://huggingface.co/spaces/peft-internal-testing/PEFT-method-comparison) aims to give a rough overview of (most) implemented methods on selected benchmarks and models.


## Prompt-based methods

Prompting primes a frozen pretrained model for a specific downstream task by including a text prompt that describes the task or even demonstrates an example of the task. With prompting, you can avoid fully training a separate model for each downstream task, and use the same frozen pretrained model instead. This is a lot easier because you can use the same model for several different tasks, and it is significantly more efficient to train and store a smaller set of prompt parameters than to train all the model's parameters.

There are two categories of prompting methods:

- hard prompts are manually handcrafted text prompts with discrete input tokens; the downside is that it requires a lot of effort to create a good prompt
- soft prompts are learnable tensors concatenated with the input embeddings that can be optimized to a dataset; the downside is that they aren't human readable because you aren't matching these "virtual tokens" to the embeddings of a real word

The PEFT library supports several types of prompting methods (p-tuning, prefix tuning, prompt tuning, ...), explore the table of contents for a full listing of soft prompt methods.
If you're interested in applying these methods to other tasks and use cases, take a look at our [notebook collection](https://huggingface.co/spaces/PEFT/soft-prompting)!

> [!TIP]
> Some familiarity with the general process of training a causal language model would be really helpful and allow you to focus on the soft prompting methods. If you're new, we recommend taking a look at the [Causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling) guide first from the Transformers documentation. When you're ready, come back and see how easy it is to drop PEFT into your training!

## Layer Tuning

Layer Tuning categorizes methods that target one type of layer or one aspect of a layer specifically, for example [LayerNorm Tuning](../package_reference/layernorm_tuning) targets only [`LayerNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) layers and [TrainableTokens](../package_reference/trainable_tokens) only targets specific tokens in the embedding matrix. This contrasts prompt-based methods which work with the model input or adapter methods which extend the existing weights and are generally more independent of the layer type, targeting linear or convolutional layers.

## Adapter methods

Adapter methods can be seen as ways of adding relatively small, trainable matrices to existing models for fine-tuning. The goal is to introduce few trainable parameters to steer the big model in the direction of the task that needs fine-tuning to save on resources, such as memory or compute.

A popular way to realize adapters is to insert smaller trainable matrices that are a low-rank decomposition of the adapted weight's layout to save on memory. There are several different ways to express the weight matrix as a low-rank decomposition, but [Low-Rank Adaptation (LoRA)](../package_reference/lora) is the most common method. The PEFT library supports several other variations of this formulation - some are direct variants of LoRA and are documented under LoRA, some are different enough to count as their own methods, such as [Low-Rank Hadamard Product (LoHa)](../package_reference/loha), [Low-Rank Kronecker Product (LoKr)](../package_reference/lokr), and [Adaptive Low-Rank Adaptation (AdaLoRA)](../package_reference/adalora). If you're interested in applying these methods to other tasks and use cases like semantic segmentation, token classification, take a look at our [notebook collection](https://huggingface.co/collections/PEFT/notebooks-6573b28b33e5a4bf5b157fc1)!

> [!TIP]
> LoRA is one of the most popular PEFT methods and a good starting point if you're just getting started with PEFT. It was originally developed for large language models but it is a tremendously popular training method for diffusion models because of its efficiency and effectiveness.

Low-rank adapters are only one possible adapter formulation, PEFT implements many other types of adapters as well. For example, Orthogonal Fine-Tuning methods ([OFT](../package_reference/oft), [BOFT](../package_reference/boft), ...) use orthogonal decompositions of the adapter weights to achieve small size. Methods like [MiSS](../package_reference/miss) shard matrices and share these shards to save on memory. [IA3](../package_reference/ia3) introduces learned vectors that rescale the key, value, and feed-forward activations.
