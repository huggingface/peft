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

# BOFT

[Orthogonal Butterfly (BOFT)](https://hf.co/papers/2311.06243) is a generic method designed for finetuning foundation models. It improves the parameter efficiency of the finetuning paradigm -- Orthogonal Finetuning (OFT), by taking inspiration from Cooley-Tukey fast Fourier transform, showing favorable results across finetuning different foundation models, including large vision transformers, large language models and text-to-image diffusion models.

The abstract from the paper is:

*Large foundation models are becoming ubiquitous, but training them from scratch is prohibitively expensive. Thus, efficiently adapting these powerful models to downstream tasks is increasingly important. In this paper, we study a principled finetuning paradigm -- Orthogonal Finetuning (OFT) -- for downstream task adaptation. Despite demonstrating good generalizability, OFT still uses a fairly large number of trainable parameters due to the high dimensionality of orthogonal matrices. To address this, we start by examining OFT from an information transmission perspective, and then identify a few key desiderata that enable better parameter-efficiency. Inspired by how the Cooley-Tukey fast Fourier transform algorithm enables efficient information transmission, we propose an efficient orthogonal parameterization using butterfly structures. We apply this parameterization to OFT, creating a novel parameter-efficient finetuning method, called Orthogonal Butterfly (BOFT). By subsuming OFT as a special case, BOFT introduces a generalized orthogonal finetuning framework. Finally, we conduct an extensive empirical study of adapting large vision transformers, large language models, and text-to-image diffusion models to various downstream tasks in vision and language*.

BOFT focuses on preserving a pretrained model's generative capabilities while being significantly more parameter-efficient than standard [OFT](./oft). Like OFT, BOFT maintains the same cosine similarity ([hyperspherical energy](https://huggingface.co/papers/1805.09298)) between all pairwise neurons in a layer by applying an orthogonal transformation to the pretrained weight matrix, ensuring the semantic relationships among neurons are preserved.

Instead of using a block-diagonal orthogonal matrix, BOFT factorizes the orthogonal transformation into a product of **sparse butterfly matrices** (originally introduced in the [Cooley–Tukey FFT](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm)). Unlike OFT's block-diagonal rotations, which only mix inputs within each block, the butterfly structure guarantees that every input can influence every output, producing a **dense connectivity** with just `O(d log d)` parameters. This factorization preserves expressivity while drastically reducing the parameter count compared to OFT (at the expense of computation time).

In practice, BOFT multiplies each pretrained weight matrix by a sequence of butterfly-structured orthogonal factors, enabling efficient and expressive neuron rotations. This makes BOFT well-suited for controllable generation and tasks where maintaining the pretrained model's subject representation is critical, while also scaling to larger models with lower memory and compute overhead.

BOFT can be applied to any subset of weight matrices in a neural network to reduce the number of trainable parameters. Given the target layers for injecting BOFT parameters, the number of trainable parameters can be determined based on the size of the weight matrices.

## Benchmark overview

<iframe
	src="https://hubnemo-peft-method-comparison-individual.hf.space/?highlight[type]=BOFT"
	frameborder="0"
	width="850"
	height="1000"
></iframe>

## Merge BOFT weights into the base model

Similar to LoRA, the weights learned by BOFT can be integrated into the pretrained weight matrices using the [`~BOFTModel.merge_and_unload()` function. This function merges the adapter weights with the base model which allows you to effectively use the newly merged model as a standalone model.

<div class="flex justify-center">
    <img src="https://raw.githubusercontent.com/wy1iu/butterfly-oft/main/assets/boft_merge.png"/>
</div>

This works because during training, the orthogonal weight matrix (R in the diagram above) and the pretrained weight matrices are separate. But once training is complete, these weights can actually be merged (multiplied) into a new weight matrix that is equivalent.

## BOFT Example Usage

For an example of the BOFT method application to various downstream tasks, please refer to the following guides:

Take a look at the following step-by-step guides on how to finetune a model with BOFT:
- [Dreambooth finetuning with BOFT](https://github.com/huggingface/peft/blob/main/examples/boft_dreambooth/boft_dreambooth.md)
- [Controllable generation finetuning with BOFT (ControlNet)](https://github.com/huggingface/peft/blob/main/examples/boft_controlnet/boft_controlnet.md)

For the task of image classification, one can initialize the BOFT config for a DinoV2 model as follows:

```py
import transformers
from transformers import AutoModelForSeq2SeqLM, BOFTConfig
from peft import BOFTConfig, get_peft_model

config = BOFTConfig(
    boft_block_size=4,
    boft_n_butterfly_factor=2,
    target_modules=["query", "value", "key", "output.dense", "mlp.fc1", "mlp.fc2"],
    boft_dropout=0.1,
    bias="boft_only",
    modules_to_save=["classifier"],
)

model = transformers.Dinov2ForImageClassification.from_pretrained(
    "facebook/dinov2-large",
    num_labels=100,
)

boft_model = get_peft_model(model, config)
```

# API

## BOFTConfig

[[autodoc]] tuners.boft.config.BOFTConfig

## BOFTModel

[[autodoc]] tuners.boft.model.BOFTModel
