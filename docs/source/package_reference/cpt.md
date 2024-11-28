<!-- Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0


⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Context-Aware Prompt Tuning (CPT)

[Context-aware Prompt Tuning: Advancing In-Context Learning with Adversarial Methods (CPT)](https://huggingface.co/papers/2410.17222) combines In-Context Learning (ICL) with Prompt Tuning (PT) and adversarial optimization to improve few-shot learning by refining context embeddings. CPT optimizes only context tokens, which minimizes overfitting and enhances performance on classification tasks.

The abstract from the paper is:

*Traditional fine-tuning is effective but computationally intensive, as it requires updating billions of parameters. CPT, inspired by ICL, PT, and adversarial attacks, refines context embeddings in a parameter-efficient manner. By optimizing context tokens and applying a controlled gradient descent, CPT achieves superior accuracy across various few-shot classification tasks, showing significant improvement over existing methods such as LoRA, PT, and ICL.*

CPT is designed for few-shot scenarios, as concatenating more examples increases memory usage due to the self-attention mechanism and additional loss terms. For larger datasets, users can limit the number of context examples and use the remaining samples solely for optimization to manage memory efficiently

Take a look at [Example](https://github.com/huggingface/peft/blob/main/examples/cpt_finetuning/README.md) for a step-by-step guide on how to train a model with CPT.


## CPTConfig

[[autodoc]] tuners.cpt.config.CPTConfig

## CPTEmbedding

[[autodoc]] tuners.cpt.model.CPTEmbedding

