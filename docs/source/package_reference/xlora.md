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

# X-LoRA

Mixture of LoRA Experts ([X-LoRA](https://huggingface.co/papers/2402.07148)) is a PEFT method enabling sparse or dense mixture of LoRA experts based on a high granularity (token, layer, sequence) scalings matrix. This leverages frozen LoRA adapters and a frozen base model to drastically reduces the number of parameters that need to be fine-tuned.

A unique aspect of X-LoRA is its versatility: it can be applied to any `transformers` base model with LoRA adapters. This means that, despite the mixture of experts strategy, no changes to the model code must be made.

The below graphic demonstrates how the scalings change for different prompts for each token. This highlights the activation of different adapters as the generation progresses and the sequence creates new context.

![Token-by-token scalings](https://github.com/EricLBuehler/xlora/raw/master/res/token_by_token_scalings.gif)

The abstract from the paper is:

*We report a mixture of expert strategy to create fine-tuned large language models using a deep layer-wise token-level approach based on low-rank adaptation (LoRA). Starting with a set of pre-trained LoRA adapters, our gating strategy uses the hidden states to dynamically mix adapted layers, allowing the resulting X-LoRA model to draw upon different capabilities and create never-before-used deep layer-wise combinations to solve tasks. The design is inspired by the biological principles of universality and diversity, where neural network building blocks are reused in different hierarchical manifestations. Hence, the X-LoRA model can be easily implemented for any existing large language model (LLM) without a need for modifications of the underlying structure. We develop a tailored X-LoRA model that offers scientific capabilities including forward/inverse analysis tasks and enhanced reasoning capability, focused on biomaterial analysis, protein mechanics and design. The impact of this work include access to readily expandable and adaptable models with strong domain knowledge and the capability to integrate across areas of knowledge. Featuring experts in biology, mathematics, reasoning, bio-inspired materials, mechanics and materials, chemistry, protein biophysics, mechanics and quantum-mechanics based molecular properties, we conduct a series of physics-focused case studies. We examine knowledge recall, protein mechanics forward/inverse tasks, protein design, adversarial agentic modeling including ontological knowledge graph construction, as well as molecular design. The model is capable not only of making quantitative predictions of nanomechanical properties of proteins or quantum mechanical molecular properties, but also reasons over the results and correctly predicts likely mechanisms that explain distinct molecular behaviors.*.

Please cite X-LoRA as:
```bibtex
@article{10.1063/5.0203126,
    author = {Buehler, Eric L. and Buehler, Markus J.},
    title = "{X-LoRA: Mixture of low-rank adapter experts, a flexible framework for large language models with applications in protein mechanics and molecular design}",
    journal = {APL Machine Learning},
    volume = {2},
    number = {2},
    pages = {026119},
    year = {2024},
    month = {05},
    abstract = "{We report a mixture of expert strategy to create fine-tuned large language models using a deep layer-wise token-level approach based on low-rank adaptation (LoRA). Starting with a set of pre-trained LoRA adapters, our gating strategy uses the hidden states to dynamically mix adapted layers, allowing the resulting X-LoRA model to draw upon different capabilities and create never-before-used deep layer-wise combinations to solve tasks. The design is inspired by the biological principles of universality and diversity, where neural network building blocks are reused in different hierarchical manifestations. Hence, the X-LoRA model can be easily implemented for any existing large language model without a need for modifications of the underlying structure. We develop a tailored X-LoRA model that offers scientific capabilities, including forward/inverse analysis tasks and enhanced reasoning capability, focused on biomaterial analysis, protein mechanics, and design. The impact of this work includes access to readily expandable and adaptable models with strong domain knowledge and the capability to integrate across areas of knowledge. Featuring experts in biology, mathematics, reasoning, bio-inspired materials, mechanics and materials, chemistry, protein biophysics, mechanics, and quantum-mechanics based molecular properties, we conduct a series of physics-focused case studies. We examine knowledge recall, protein mechanics forward/inverse tasks, protein design, adversarial agentic modeling including ontological knowledge graph construction, and molecular design. The model is capable not only of making quantitative predictions of nanomechanical properties of proteins or quantum mechanical molecular properties but also reasoning over the results and correctly predicting likely mechanisms that explain distinct molecular behaviors.}",
    issn = {2770-9019},
    doi = {10.1063/5.0203126},
    url = {https://doi.org/10.1063/5.0203126},
    eprint = {https://pubs.aip.org/aip/aml/article-pdf/doi/10.1063/5.0203126/19964043/026119\_1\_5.0203126.pdf},
}
```

## XLoraConfig

[[autodoc]] tuners.xlora.config.XLoraConfig

## XLoraModel

[[autodoc]] tuners.xlora.model.XLoraModel
