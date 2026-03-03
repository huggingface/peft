<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Uni-LoRA: One Vector is All You Nees

## Overview

[Uni-LoRA](https://arxiv.org/abs/2506.00799) is a parameter-efficient fine-tuning method that reconstructs LoRA parameters from a low-dimensional subspace instead of learning full low-rank matrices for each layer. It uses a shared and fixed isometric projection to map the vector learned in the subspace to the LoRA parameter space of each layer, thereby achieving global parameter sharing with very low computational overhead. This design reduces the number of trainable parameters to a single global vector while maintaining competitive performance.

The abstract from the paper is:

*Low-Rank Adaptation (LoRA) has become the de facto parameter-efficient fine-tuning (PEFT) method for large language models (LLMs) by constraining weight updates to low-rank matrices. Recent works such as Tied-LoRA, VeRA, and VB-LoRA push efficiency further by introducing additional constraints to reduce the trainable parameter space. In this paper, we show that the parameter space reduction strategies employed by these LoRA variants can be formulated within a unified framework, Uni-LoRA, where the LoRA parameter space, flattened as a high-dimensional vector space \(\mathbb{R}^D\), can be reconstructed through a projection from a subspace \(\mathbb{R}^d\), with \(d \ll D\). We demonstrate that the fundamental difference among various LoRA methods lies in the choice of the projection matrix \(P \in \mathbb{R}^{D \times d}\). Most existing LoRA variants rely on layer-wise or structure-specific projections that limit cross-layer parameter sharing, thereby compromising parameter efficiency. In light of this, we introduce an efficient and theoretically grounded projection matrix that is isometric, enabling global parameter sharing and reducing computation overhead. Furthermore, under the unified view of Uni-LoRA, this design requires only a single trainable vector to reconstruct LoRA parameters for the entire LLM — making Uni-LoRA both a unified framework and a “one-vector-only” solution. Extensive experiments on GLUE, mathematical reasoning, and instruction tuning benchmarks demonstrate that Uni-LoRA achieves state-of-the-art parameter efficiency while outperforming or matching prior approaches in predictive performance.*

## Usage Tips

- **Single-vector parameterization**: Uni-LoRA only learns one global vector `theta_d`, reducing both memory and storage footprint. This is especially useful when scaling to larger models or maintaining many per-user adapters.

- **Global projection**: Uni-LoRA applies an isometric projection shared across layers, allowing cross-layer parameter reuse.

- **Compatibility**: Uni-LoRA is compatible with standard LoRA target modules and integrates seamlessly into existing PEFT workflows.

## UniLoraConfig

[[autodoc]] tuners.unilora.config.UniLoraConfig

## UniLoraModel

[[autodoc]] tuners.unilora.model.UniLoraModel