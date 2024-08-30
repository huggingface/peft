<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# VB-LoRA: Extreme Parameter Efficient Fine-Tuning with Vector Banks

[VB-LoRA](https://arxiv.org/abs/2405.15179) is a parameter-efficient fine-tuning method that introduces a 'divide-and-share' paradigm. It achieves significantly higher parameter efficiency than LoRA by further decomposing low-rank matrices across differnt modules (such as K, Q, V, and FFN) and layers into sub-vectors, which are then shared globally through a vector bank.

The abstract from the paper is:

*As the adoption of large language models increases and the need for per-user or per-task model customization grows, the parameter-efficient fine-tuning (PEFT) methods, such as low-rank adaptation (LoRA) and its variants, incur substantial storage and transmission costs. To further reduce stored parameters, we introduce a "divide-and-share" paradigm that breaks the barriers of low-rank decomposition across matrix dimensions, modules and layers by sharing parameters globally via a vector bank. As an instantiation of the paradigm to LoRA, our proposed VB-LoRA composites all the low-rank matrices of LoRA from a shared vector bank with a differentiable top-k admixture module. VB-LoRA achieves extreme parameter efficiency while maintaining comparable or better performance compared to state-of-the-art PEFT methods. Extensive experiments demonstrate the effectiveness of VB-LoRA on natural language understanding, natural language generation, and instruction tuning tasks. When fine-tuning the Llama2-13B model, VB-LoRA only uses 0.4% of LoRA's stored parameters, yet achieves superior results.*

VB-LoRA utilizes a sparse top-k module to learn the sharing machanism. When saving adapter parameters, you can either save only the top-k weights and their indices by setting `save_only_topk_weights = True` in `VBLoRAConfig`, or save all the trainable logits by setting it to `False`. Enabling `save_only_topk_weights = True` significantly reduces storage space; for instance, in Llama2-7B, the storage file size decreases from 308MB to 2.5MB. Note that models saved with this setting are intended for merging or inference only and cannot be used to resume training.

## VBLoRAConfig

[[autodoc]] tuners.vblora.config.VBLoRAConfig

## VBLoRAModel

[[autodoc]] tuners.vblora.model.VBLoRAModel

