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

# RoAd

[RoAd](https://arxiv.org/pdf/2409.00119) is a parameter‑efficient fine‑tuning technique that adapts large language models by learning a small set of 2×2 rotation matrices (and optional scaling factors) applied to pairs of hidden dimensions, achieving competitive or superior performance with under 0.1% trainable parameters. Unlike LoRA’s batched low‑rank updates, RoAd’s sparse rotations reformulate to simple element‑wise operations, yielding significantly higher serving throughput when handling heterogeneous requests in the same batch. Moreover, RoAd integrates seamlessly into a distributed interchange intervention framework, enabling interpretable, composable task‑specific adaptations by combining orthogonal subspaces learned for different tasks.

Finetuning with RoAd typically requires higher learning rate compared to LoRA or similar methods, around 1e-3.

## RoadConfig

[[autodoc]] tuners.road.config.RoadConfig

## RoadModel

[[autodoc]] tuners.road.model.RoadModel
