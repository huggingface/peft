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

# WeightLoRA

Weight LoRA is a less complex, but important, PEFT method that adds a weight $w_i$ to each LoRA adapter (here i -- adapter number). This is done in order to perform, in addition to the classical optimisation over all LoRAs $A_1, B_1, ..., A_n, B_n$, an alternative optimisation over a vector of weights $w := (w_1, ..., w_n)^T \in R^n$ with a wide variety of constraints. In our research paper, we consider two approaches: 1) the vector $w$ must be in simplex $\Delta_{n-1}$, and 2) the vector $w$ has only $K$ non-zero coordinates. Both of these methods solve the problem of finding the most important LoRA adapters in the model and concentrating training on them while disabling the rest.

The abstract from the paper is:

The widespread utilization of language models in modern applications is inconceivable without Parameter-Efficient Fine-Tuning techniques, such as low-rank adaptation (LoRA), which adds trainable adapters to selected layers. Although LoRA may obtain accurate solutions, it requires significant memory to train large models and intuition on which layers to add adapters. In this paper, we propose a novel method, WeightLoRA, which overcomes this issue by adaptive selection of the most critical LoRA heads throughout the optimization process. As a result, we can significantly reduce the number of trainable parameters while maintaining the capability to obtain consistent or even superior metric values. Finally, we conduct experiments for the series of competitive benchmarks and DeBERTa and BART models, comparing our approach with the most popular LoRA modifications. The experimental results demonstrate the efficacy of WeightLoRA and the superior performance of WeightLoRA+ in comparison to the baselines in nearly all cases.

## WeightLoraConfig

[[autodoc]] tuners.weight_lora.config.WeightLoraConfig

## WeightLoraModel

[[autodoc]] tuners.weight_lora.model.WeightLoraModel