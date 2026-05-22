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

# LoHa

Low-Rank Hadamard Product ([LoHa](https://huggingface.co/papers/2108.06098)), is similar to LoRA except it approximates the large weight matrix with more low-rank matrices and combines them with the Hadamard product. This method is even more parameter-efficient than LoRA and achieves comparable performance. LoHa was originally proposed for federated learning (FedPara) but works well as a general-purpose PEFT method, and is especially popular for fine-tuning image generation models such as Stable Diffusion.

> **Note:** LoHa is part of the [LyCORIS](./adapter_utils) family of adapters. Its close relative [LoKr](./lokr) uses the Kronecker product instead of the Hadamard product. For more background on how LoHa works conceptually, see the [Adapters guide](../conceptual_guides/adapter#low-rank-hadamard-product-loha).

The abstract from the paper is:

*In this work, we propose a communication-efficient parameterization, FedPara, for federated learning (FL) to overcome the burdens on frequent model uploads and downloads. Our method re-parameterizes weight parameters of layers using low-rank weights followed by the Hadamard product. Compared to the conventional low-rank parameterization, our FedPara method is not restricted to low-rank constraints, and thereby it has a far larger capacity. This property enables to achieve comparable performance while requiring 3 to 10 times lower communication costs than the model with the original layers, which is not achievable by the traditional low-rank methods. The efficiency of our method can be further improved by combining with other efficient FL optimizers. In addition, we extend our method to a personalized FL application, pFedPara, which separates parameters into global and local ones. We show that pFedPara outperforms competing personalized FL methods with more than three times fewer parameters.*

## When to use LoHa

LoHa is a good choice when:

- You are fine-tuning **image generation models** (Stable Diffusion UNet or text encoder), where it is most widely used.
- You want **higher effective rank** than LoRA for the same number of trainable parameters, since the Hadamard product of two low-rank matrices spans a larger subspace than a single low-rank product.
- You want to **combine different PEFT methods** at inference time using [`PeftMixedModel`](./peft_model#peft.PeftMixedModel), for example LoHa together with LoKr.

LoHa supports linear and Conv2d layers. For tasks that additionally require embedding layer adaptation, consider [LoRA](./lora) instead.

## Quick start

```python
from diffusers import StableDiffusionPipeline
from peft import LoHaConfig, get_peft_model

config_unet = LoHaConfig(
    r=8,
    alpha=8,
    target_modules=[
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
        "proj_in",
        "proj_out",
    ],
    rank_dropout=0.0,
    module_dropout=0.0,
    use_effective_conv2d=True,
)

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.unet = get_peft_model(pipeline.unet, config_unet)
pipeline.unet.print_trainable_parameters()
```

## LoHaConfig

[[autodoc]] tuners.loha.config.LoHaConfig

## LoHaModel

[[autodoc]] tuners.loha.model.LoHaModel
