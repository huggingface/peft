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

# DeLoRA: Decoupled Low-rank Adaptation
[DeLoRA](https://huggingface.co/papers/2503.18225) is a parameter-efficient fine-tuning technique that implicitly maintains a Frobenius boundary with respect to the pretrained weights by normalizing and scaling learnable low-rank matrices. This effectively decouples the learning of directions (BA term) and magnitude (boundary term) of the weight updates, avoiding catastrophic shifts in the adapted weights and enhancing robustness to hyperparameter choices.

Note:
- use a learning rate 10-100x larger than for standard LoRA variants (typical values from 1e-3/1e-2/..)
- ensure the initial boundary parameter lambda is not too small (typical values around 10/15/..). Setting different lambdas to different layers is possible

DeLoRA currently has the following constraints:
- Only nn.Linear layers are supported.
- Quantized layers are not supported.

If these constraints don't work for your use case, consider other methods instead.

The abstract from the paper is:

> Parameter-Efficient FineTuning (PEFT) methods have recently gained significant popularity thanks to the widespread availability of large-scale pretrained models. These methods allow for quick adaptation to downstream tasks with minimal computational cost. However, popular finetuning methods such as LoRA exhibit limited robustness when it comes to hyperparameter choices or extended training regimes, preventing optimal out-of-the-box performance. In contrast, bounded approaches, such as ETHER, provide greater robustness but are limited to extremely low-rank adaptations and fixed-strength transformations, reducing their adaptation expressive power. In this work, we propose Decoupled Low-rank Adaptation (DeLoRA), a novel finetuning method that normalizes and scales learnable low-rank matrices. By bounding the distance of the transformation, DeLoRA effectively decouples the angular learning from the adaptation strength, enhancing robustness without compromising performance. Through evaluations on subject-driven image generation, natural language understanding, and instruction tuning, we show that DeLoRA matches or surpasses performance of competing PEFT methods, while exhibiting stronger robustness. 

## DeloraConfig

[[autodoc]] tuners.delora.config.DeloraConfig

## DeloraModel

[[autodoc]] tuners.delora.model.DeloraModel
