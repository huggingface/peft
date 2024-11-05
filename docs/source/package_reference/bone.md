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

# Bone

Block Affine ([Bone](https://huggingface.co/papers/2409.15371)) is a new PEFT technology different from the LoRA series, reducing training resources and requiring no complex initialization. Bone not only enhances the utilization of original weight information but also emphasizes the internal connections between weights, leading to faster convergence and better data fitting.

The abstract from the paper is:

Low-Rank Adaptation (LoRA) has achieved remarkable training results by freezing the original weights and training only low-rank matrices, establishing itself as the predominant fine-tuning method for LLMs. In pursuit of performance closer to full-parameter training, a series of LoRA variants have emerged, such as LoRA+, PISSA, Olora, and LoRA-GA. However, these improvements complicate the initial setup of model training and increase initialization time. More importantly, they overlook the internal interactions of the original weight information. To address these issues, we introduce a novel theory, ``Weight Guide'' aimed at continuously guiding trainable matrices through the original weights during training to enhance the utilization of weight information. Based on this theory, we designed a new PEFT technique called Bone (Block Affine), which not only enhances the utilization of original weight information but also emphasizes the internal connections between weights, leading to faster convergence and better data fitting. Experimental comparisons across two different LLM architectures (LLaMA2, RWKV6) and various parameter scales demonstrate that the Bone structure can achieve rapid convergence and superior data fitting without the need for complex initialization. For example, when fine-tuning LLaMA2-7B on the MetaMathQA dataset and validating on GSM8k and math benchmarks, Bone achieved fine-tuning scores of 49.36 and 8.8, respectively, outperforming PISSA by 5.84\% and 1.96\%.

## BoneConfig

[[autodoc]] tuners.bone.config.BoneConfig

## BoneModel

[[autodoc]] tuners.bone.model.BoneModel