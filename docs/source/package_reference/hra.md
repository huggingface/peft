<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Bridging The Gap between Low-rank and Orthogonal Adaptation via Householder Reflection Adaptation (HRA)

[HRA](https://huggingface.co/papers/2405.17484) is a simple but effective adapter-based fine-tuning method by leveraging Householder reflections. This method harnesses the advantages of both strategies, reducing parameters and computation costs while penalizing the loss of pre-training knowledge. It consistently achieves better performance with fewer trainable parameters and outperforms state-of-the-art adapters across different models, including large language models (LLMs) and conditional image generators.


The abstract from the paper is:

> While following different technical routes, both low-rank and orthogonal adaptation techniques can efficiently adapt large-scale pre-training models in specific tasks or domains based on a small piece of trainable parameters. In this study, we bridge the gap between these two techniques, proposing a simple but effective adaptation method based on Householder reflections. Given a pre-trained model, our method fine-tunes its layers by multiplying each frozen weight matrix with an orthogonal matrix constructed by a chain of learnable Householder reflections (HRs). This HR-based orthogonal fine-tuning is equivalent to an adaptive low-rank adaptation. Moreover, we show that the orthogonality of the reflection planes corresponding to the HRs impacts the model capacity and regularity. The analysis motivates us to regularize the orthogonality of the HRs, leading to different implementations of the proposed Householder reflection adaptation (HRA) method. Compared with state-of-the-art methods, HRA achieves superior performance with fewer learnable parameters when adapting large language models and conditional image generators. The code is available at [peft](https://github.com/huggingface/peft/tree/main/src/peft/tuners/hra) and [HRA](https://github.com/DaShenZi721/HRA).

## HRAConfig

[[autodoc]] tuners.hra.config.HRAConfig

## HRAModel

[[autodoc]] tuners.hra.model.HRAModel
