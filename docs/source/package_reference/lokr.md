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

# LoKr

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/lora.png"/>
</div>
<small><a href="https://hf.co/papers/2103.10385">Navigating Text-To-Image Customization: From LyCORIS Fine-Tuning to Model Evaluation</a></small>

Low-Rank Kronecker Product ([LoKr](https://hf.co/papers/2309.14859)), is a LoRA-variant method that approximates the large weight matrix with two low-rank matrices and combines them with the [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product). LoKr also provides an optional third low-rank matrix to provide better control during fine-tuning. By expresseing the weight update matrix as a decomposition of a Kronecker product, creating a block matrix, LoKr is able to preserve the rank of the original weight matrix. The size of the smaller matrices are determined by its *rank* or `r`. Another benefit of the Kronecker product is that it can be vectorized by stacking the matrix columns. This can speed up the process because you're avoiding fully reconstructing ∆W.

The abstract from the paper is:

*Text-to-image generative models have garnered immense attention for their ability to produce high-fidelity images from text prompts. Among these, Stable Diffusion distinguishes itself as a leading open-source model in this fast-growing field. However, the intricacies of fine-tuning these models pose multiple challenges from new methodology integration to systematic evaluation. Addressing these issues, this paper introduces LyCORIS [Lora beYond Conventional methods, Other Rank adaptation Implementations for Stable diffusion](https://github.com/KohakuBlueleaf/LyCORIS), an open-source library that offers a wide selection of fine-tuning methodologies for Stable Diffusion. Furthermore, we present a thorough framework for the systematic assessment of varied fine-tuning techniques. This framework employs a diverse suite of metrics and delves into multiple facets of fine-tuning, including hyperparameter adjustments and the evaluation with different prompt types across various concept categories. Through this comprehensive approach, our work provides essential insights into the nuanced effects of fine-tuning parameters, bridging the gap between state-of-the-art research and practical application.*

## Usage

```py
from peft import LoKrConfig, get_peft_model

config = LoKrConfig(
    r=16,
    alpha=16,
    target_modules=["query", "value"],
    module_dropout=0.1,
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
"trainable params: 116,069 || all params: 87,172,042 || trainable%: 0.13314934162033282"
```

## Benchmark overview

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=LOKR"
	frameborder="0"
	width="850"
	height="1000"
></iframe>

# API

## LoKrConfig

[[autodoc]] tuners.lokr.config.LoKrConfig

## LoKrModel

[[autodoc]] tuners.lokr.model.LoKrModel
