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

Block-Affine Adaptation ([Bone](https://huggingface.co/papers/2409.15371)) This work, inspired by GQA and MQA, leverages the sparsity of LLM weights to design the Block-Affine Adaptation (Bone) structure. In Bone, the original weights are divided into multiple sub-spaces, all of which share a single low-rank matrix initialized to zero for updates. Extensive exper-iments demonstrate that Bone consistently outperforms LoRA and its variants across various tasks, while also offering superior computational efficiency. 

The abstract from the paper is:

Low-Rank Adaptation (LoRA) has achieved remarkable training results by freezing the original weights and training only low-rank matrices, establishing itself as the predominant fine-tuning method for LLMs. In pursuit of performance closer to full-parameter training, a series of LoRA variants have emerged, such as LoRA+, PISSA, Olora, and LoRA-GA. This paper introduces a novel PEFT technique distinct from LoRA, called Block-Affine Adaptation (Bone). By dividing the original weights into multiple subspaces that share a single matrix for weight updates, Bone simplifies the process by requiring the trainable matrix to be initialized to zero, eliminating the need for complex initialization as in some LoRA variants. Compared to LoRA, Bone significantly reduces memory usage and achieves faster computation. Evaluation of both NLU and NLG tasks demonstrates that Bone substantially outperforms LoRA and its variants. Inspired by Pissa, we further proposed the 'Weight Guide' theory to better utilize the information from the original weights. By integrating 'Weight Guide' with Bone, we developed a new structure called Block-Affine Transformation (Bat), and ablation experiments confirmed the effectiveness of 'Weight Guide'.

## BoneConfig

[[autodoc]] tuners.bone.config.BoneConfig

## BoneModel

[[autodoc]] tuners.bone.model.BoneModel