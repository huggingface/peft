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

# Adapters

Adapter-based methods add extra trainable parameters after the attention and fully-connected layers of a frozen pretrained model to reduce memory-usage and speed up training. The adapters are typically small, but demonstrate comparable performance to a fully finetuned model. These methods enable training larger models with fewer resources.

This guide will give you a brief overview of adapter methods supported by PEFT (if you're interested in learning more details about a specific method, take a look at the linked paper).

## Llama-Adapter

[Llama-Adapter](https://hf.co/papers/2303.16199) is a method for adapting Llama into a instruction-following model. To help adapt the model for instruction-following, the adapter is trained with a 52K instruction-output dataset.

A set of of learnable adaption prompts are prefixed to the input instruction tokens. These are inserted into the upper layers of the model because it is better to learn with the higher-level semantics of the pretrained model. The instruction-output tokens prefixed to the input guide the adaption prompt to generate a contextual response.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/llama-adapter.png"/>
</div>
<small><a href="https://hf.co/papers/2303.16199">LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention</a>.</small>

To avoid adding noise to the tokens, the adapter uses zero-initialized attention. On top of this, the adapter adds a learnable gating factor (initialized with zeros) to progressively add information to the model during training. This prevents overwhelming the model's pretrained knowledge with the newly learned instructions.
