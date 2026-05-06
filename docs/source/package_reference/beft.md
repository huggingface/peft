<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# BEFT: Bias-Efficient Fine-Tuning of Language Models in Low-Data Regimes

[BEFT](https://arxiv.org/abs/2509.15974) is a parameter efficient fine-tuning algorithm (PEFT) that only fine-tunes the added bias terms of value projections from pretrained transformer models. BEFT demonstrates that fine-tuning the added bias terms of value projections from pretrained transformers generally leads to a higher downstream performance in low-data regimes than fine-tuning the added bias terms of query/key projections.

BEFT currently has the following tradeoffs:

Pros:
- BEFT requires far fewer parameters than LoRA, while maintaining competitive or superior performance across tasks in low-data regimes. 

Cons:
- In high-data regimes, BEFT may show limited effectiveness compared to LoRA and full-parameters fine-tuning.

If your use case belongs to the high-data regime, consider other PEFT methods such as LoRA.

The abstract from the paper is:

*Fine-tuning the bias terms of large language models (LLMs) has the potential to achieve unprecedented parameter efficiency while maintaining competitive performance, particularly in low-data regimes. However, the link between fine-tuning different bias terms (i.e., **b**<sub>q</sub>, **b**<sub>k</sub>, and **b**<sub>v</sub> in the query, key, or value projections) and downstream performance remains largely unclear to date. In this paper, we investigate the link between fine-tuning **b**<sub>q</sub>, **b**<sub>k</sub>, and **b**<sub>v</sub> with the performance of the downstream task. Our key finding is that directly fine-tuning **b**<sub>v</sub> generally leads to higher downstream performance in low-data regimes, in comparison to **b**<sub>q</sub> and **b**<sub>k</sub>. We extensively evaluate this unique property across a wide range of LLMs spanning encoder-only and decoder-only architectures up to 6.7B parameters (including bias-free LLMs). Our results provide strong evidence for the effectiveness of directly fine-tuning **b**<sub>v</sub> across various downstream tasks*.

## BeftConfig

[[autodoc]] tuners.beft.config.BeftConfig

## BeftModel

[[autodoc]] tuners.beft.model.BeftModel