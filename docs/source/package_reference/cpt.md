<!-- Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Context-Aware Prompt Tuning (CPT)

[Context-aware Prompt Tuning: Advancing In-Context Learning with Adversarial Methods (CPT)](https://huggingface.co/papers/2410.17222) combines In-Context Learning (ICL) with Prompt Tuning (PT) and adversarial optimization to improve few-shot learning by refining context embeddings. CPT optimizes only context tokens, which minimizes overfitting and enhances performance on classification tasks.

The abstract from the paper is:

*Traditional fine-tuning is effective but computationally intensive, as it requires updating billions of parameters. CPT, inspired by ICL, PT, and adversarial attacks, refines context embeddings in a parameter-efficient manner. By optimizing context tokens and applying a controlled gradient descent, CPT achieves superior accuracy across various few-shot classification tasks, showing significant improvement over existing methods such as LoRA, PT, and ICL.*

## CPTConfig

[[autodoc]] tuners.cpt.config.CPTConfig

## CPTEmbedding

[[autodoc]] tuners.cpt.model.CPTEmbedding

