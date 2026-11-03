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

# Llama-Adapter

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/llama-adapter.png"/>
</div>
<small><a href="https://hf.co/papers/2303.16199">LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention</a></small>

[Llama-Adapter](https://hf.co/papers/2303.16199) is a PEFT method specifically designed for turning Llama into an instruction-following model. The Llama model is frozen and only a set of adaptation prompts prefixed to the input instruction tokens are learned. Since randomly initialized modules inserted into the model can cause the model to lose some of its existing knowledge, Llama-Adapter uses zero-initialized attention with zero gating to progressively add the instructional prompts to the model.

The abstract from the paper is:

*We present LLaMA-Adapter, a lightweight adaption method to efficiently fine-tune LLaMA into an instruction-following model. Using 52K self-instruct demonstrations, LLaMA-Adapter only introduces 1.2M learnable parameters upon the frozen LLaMA 7B model, and costs less than one hour for fine-tuning on 8 A100 GPUs. Specifically, we adopt a set of learnable adaption prompts, and prepend them to the input text tokens at higher transformer layers. Then, a zero-init attention mechanism with zero gating is proposed, which adaptively injects the new instructional cues into LLaMA, while effectively preserves its pre-trained knowledge. With efficient training, LLaMA-Adapter generates high-quality responses, comparable to Alpaca with fully fine-tuned 7B parameters. Furthermore, our approach can be simply extended to multi-modal input, e.g., images, for image-conditioned LLaMA, which achieves superior reasoning capacity on ScienceQA. We release our code at https://github.com/ZrrSkywalker/LLaMA-Adapter*.

## When to use Llama-Adapter

Llama-Adapter is a good choice when:

- You want a very small checkpoint. Only the adaption prompts and their gates are trained and saved, which amounts to `adapter_layers * (adapter_len * hidden_size + 1)` parameters.
- You are instruction-tuning a supported model (Llama, Mistral, or GPT-2), the use case the paper focuses on.
- You want to keep several task-specific adapters for the same base model and switch between them with `set_adapter`.

## When not to use Llama-Adapter

Llama-Adapter may not be the best choice when:

- Your model is not a Llama, Mistral, or GPT-2 model. Other architectures raise a `ValueError`. Consider [LoRA](./lora) instead.
- You need zero inference overhead. The adapter cannot be merged into the base weights (`merge_and_unload()` is not supported), so each adapted layer attends over `adapter_len` additional tokens at inference time.
- You want several adapters to be active at the same time. Only one adapter can be active, and adaption prompts cannot be stacked on the same attention module.
- You want to pass `inference_mode=True` to `set_adapter`, which is not supported. Set `inference_mode=True` on the config instead, or load the adapter with [`~PeftModel.from_pretrained`], which loads it frozen unless `is_trainable=True` is passed.

## Benchmark overview

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=ADAPTION_PROMPT"
	frameborder="0"
	width="850"
	height="1000"
></iframe>

# API

## AdaptionPromptConfig

[[autodoc]] tuners.adaption_prompt.config.AdaptionPromptConfig

## AdaptionPromptModel

[[autodoc]] tuners.adaption_prompt.model.AdaptionPromptModel
