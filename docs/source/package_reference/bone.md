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

DiSHA: Dimension-Sharding Adaptation ([DiSHA](https://huggingface.co/papers/2409.15371)) We introduce Dimension-Sharding Adaptation (DiSHA), which expands the PEFT design space to unlock lower intrinsic ranks and faster convergence by default. Building on DiSHA, we propose an efficient algorithm called Block-Affine Adaptation (Bone) structure and a non-linear update method called Block Affine Transformation Adaptation (BAT).


The abstract from the paper is:

Low-Rank Adaptation (LoRA) leverages the low intrinsic rank of weight updates in Large Language Models (LLMs), establishing a Parameter-Efficient Fine-Tuning (PEFT) paradigm. However, LoRA suffers from slow convergence. We introduce Dimension-Sharding Adaptation (DiSHA), which expands the PEFT design space to unlock lower intrinsic ranks and faster convergence by default. Within DiSHA's design space, we propose Block Affine Adaptation (Bone), a computationally efficient structure that delivers both high performance and efficiency. While certain DiSHA configurations may result in colinear updates to weight shards, we address this with Block Affine Transformation Adaptation (BAT), a nonlinear variant of DiSHA. BAT introduces nonlinearity by combining trainable matrices with original weight shards in a nonlinear manner, inducing nonlinearity in matrix updates without introducing additional parameters. Empirical results show that Bone, under the DiSHA framework, consistently outperforms LoRA variants in both NLG and NLU tasks, with significantly improved computational efficiency. Further analysis demonstrates that BAT enhances model capabilities by leveraging its nonlinear design.


## BoneConfig

[[autodoc]] tuners.bone.config.BoneConfig

## BoneModel

[[autodoc]] tuners.bone.model.BoneModel