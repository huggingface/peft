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

# Lily: Low-Rank Interconnected Adaptation across Layers

[Lily](https://huggingface.co/papers/2407.09946) is a parameter-efficient fine-tuning technique that introduces cross-layer weight sharing for adapter matrices. Instead of learning an independent AB pair per layer as in LoRA, Lily uses **locally shared A adapters** (each A is shared across a block of `stride_A` consecutive layers) and **globally shared B experts** (a small pool of `num_B` B adapters is shared across all layers). At each forward pass, a lightweight data-dependent router computes a softmax-weighted combination of the B experts to produce the effective B for that layer and input.

This sharing can reduce the total number of adapter matrices from `2N` (standard LoRA) to `N / stride_A + num_B`, freeing up the parameter budget to use a **much larger rank `r`** — typically `2×`–`4×` what you would use in LoRA. Higher rank and better interconnectivity increase the effective rank of the weight update `ΔW = A × combined_B`, leading to better adaptation performance.

Because the B combination is **data-dependent** (the router weights depend on the input activations at runtime), `merge` and `unmerge` are **not supported**. If weight merging is required for your deployment, consider other methods such as LoRA instead.

Lily currently has the following additional constraints:
- Only `nn.Linear` layers are supported.
- Quantized layers are not supported.

If these constraints don't work for your use case, consider other methods instead.

The abstract from the paper is:

> Low-rank adaptation (LoRA) is a widely used parameter-efficient fine-tuning (PEFT) method that learns weight updates ΔW = AB for pretrained weights W through low-rank adapters A and B. While LoRA ensures hardware efficiency, its low-rank weight updates limit adaptation performance. In this paper, we propose low-rank interconnected adaptation across layers (Lily), a novel PEFT method that introduces an interconnected framework with locally shared A and globally shared B experts. This structure eliminates redundant per-layer AB pairs, enabling higher-rank ΔW with equal or fewer parameters. To enhance expressiveness, we use data-dependent routers to determine A-B interconnections, preventing B experts from converging to the same behavior and improving representational power across domains. Experiments across modalities, architectures, and model sizes demonstrate Lily's superior performance and efficiency.

## LilyConfig

[[autodoc]] tuners.lily.config.LilyConfig

## LilyModel

[[autodoc]] tuners.lily.model.LilyModel