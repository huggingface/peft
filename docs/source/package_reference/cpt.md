<!-- Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0


⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Context-aware Prompt Tuning: Advancing In-Context Learning with Adversarial Methods

[CPT](https://huggingface.co/papers/2410.17222) combines In-Context Learning (ICL), Prompt Tuning (PT), and adversarial optimization to improve few-shot learning by refining context embeddings. CPT updates the context tokens by optimizing both the context and the training examples, encapsulating them into a novel loss design that minimizes overfitting, enables more effective optimization, and drives significant improvements in classification tasks.

[//]: # ([CPT]&#40;https://huggingface.co/papers/2410.17222&#41; for the paper)

The abstract from the paper is:

> Large Language Models (LLMs) can perform few-shot learning using either optimization-based approaches or In-Context Learning (ICL). Optimization-based methods often suffer from overfitting, as they require updating a large number of parameters with limited data. In contrast, ICL avoids overfitting but typically underperforms compared to optimization-based methods and is highly sensitive to the selection, order, and format of demonstration examples. To overcome these challenges, we introduce Context-aware Prompt Tuning (CPT), a method inspired by ICL, Prompt Tuning (PT), and adversarial attacks. CPT builds on the ICL strategy of concatenating examples before the input, extending it by incorporating PT-like learning to refine the context embedding through iterative optimization, extracting deeper insights from the training examples. Our approach carefully modifies specific context tokens, considering the unique structure of the examples within the context. In addition to updating the context with PT-like optimization, CPT draws inspiration from adversarial attacks, adjusting the input based on the labels present in the context while preserving the inherent value of the user-provided data. To ensure robustness and stability during optimization, we employ a projected gradient descent algorithm, constraining token embeddings to remain close to their original values and safeguarding the quality of the context. Our method has demonstrated superior accuracy across multiple classification tasks using various LLM models, outperforming existing baselines and effectively addressing the overfitting challenge in few-shot learning.


Take a look at [Example](https://github.com/huggingface/peft/blob/main/examples/cpt_finetuning/README.md) for a step-by-step guide on how to train a model with CPT.


## CPTConfig

[[autodoc]] tuners.cpt.config.CPTConfig

## CPTEmbedding

[[autodoc]] tuners.cpt.model.CPTEmbedding

