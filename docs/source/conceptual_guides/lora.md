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

# Reparametrization

Reparametrization is a way to express a matrix in a different way without changing it's meaning. Often, this means expressing the matrix as a decomposition of a matrix product. This is the key idea behind reparametrization-based methods like Low-Rank Adaptation (LoRA), which represents the matrix of weight updates ∆W as the matrix product of two low-rank matrices. As a result, it is faster and easier to update the weights of these two smaller matrices instead of all of the pretrained model parameters.

This guide will provide a quick conceptual overview of the reparametrization methods supported by PEFT (if you're interested in learning more details about a specific method, take a look at the linked paper).

## Low-Rank Adaptation (LoRA)

As mentioned briefly earlier, [LoRA](https://hf.co/papers/2106.09685) is a technique that accelerates finetuning large models while consuming less memory.

LoRA represents the weight updates ∆W with two smaller matrices (called *update matrices*) through low-rank decomposition. These new matrices can be trained to adapt to the new data while keeping the overall number of parameters low. The original weight matrix remains frozen and doesn't receive any further updates. To produce the final results, the original and extra adapted weights are combined. You could also merge the adapter weights with the base model to eliminate inference latency.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/lora_diagram.png"/>
</div>

This approach has a number of advantages:

* LoRA makes finetuning more efficient by drastically reducing the number of trainable parameters.
* The original pretrained weights are kept frozen, which means you can have multiple lightweight and portable LoRA models for various downstream tasks built on top of them.
* LoRA is orthogonal to other parameter-efficient methods and can be combined with many of them.
* Performance of models finetuned using LoRA is comparable to the performance of fully finetuned models.

In principle, LoRA can be applied to any subset of weight matrices in a neural network to reduce the number of trainable parameters. However, for simplicity and further parameter efficiency, LoRA is typically only applied to the attention blocks in Transformer models. The resulting number of trainable parameters in a LoRA model depends on the size of the update matrices, which is determined mainly by the rank `r` and the shape of the original weight matrix.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/lora.png"/>
</div>
<small><a href="https://hf.co/papers/2103.10385">Navigating Text-To-Image Customization: From LyCORIS Fine-Tuning to Model Evaluation</a>.</small>

## Low-Rank Hadamard Product (LoHa)

Low-rank decomposition can impact performance because the weight updates are limited to the low-rank space. This can constrain a model's expressiveness. However, you don't necessarily want to use a larger rank because it increases the number of trainable parameters. To address this, [LoHa](https://huggingface.co/papers/2108.06098) uses the [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)) (element-wise product) instead of the matrix product. ∆W is represented by four smaller matrices instead of two - like in LoRA - and each pair of these low-rank matrices are combined with the Hadamard product. As a result, ∆W can have the same number of trainable parameters but a higher rank and expressivity.

## Low-Rank Kronecker Product (LoKr)

[LoKr](https://hf.co/papers/2309.14859) is very similar to LoRA and LoHa, but it replaces the matrix product with the [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product) instead. The Kronecker product decomposition creates a block matrix which preserves the rank of the original weight matrix. Another benefit of the Kronecker product is that it can be vectorized by stacking the matrix columns. This can speed up the process because you're avoiding fully reconstructing ∆W.

## Orthogonal Finetuning (OFT)

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/oft.png"/>
</div>
<small><a href="https://hf.co/papers/2306.07280">Controlling Text-to-Image Diffusion by Orthogonal Finetuning</a>.</small>

[OFT](https://hf.co/papers/2306.07280) is a method that primarily focuses on preserving a pretrained model's generative performance in the finetuned model. It tries to maintain the same cosine similarity (hyperspherical energy) between all pairwise neurons in a layer because this better captures the semantic information among neurons. OFT preserves this hyperspherical energy by learning an orthogonal transformation for neurons to keep the cosine similarity between them unchanged. In practice, this means taking the matrix product of an orthogonal matrix with the pretrained weight matrix. However, to be parameter-efficient, the orthogonal matrix is represented as a block-diagonal matrix with rank `r` blocks. Whereas LoRA reduces the number of trainable parameters with low-rank structures, OFT reduces the number of trainable parameters with a sparse block-diagonal matrix structure.

## Adaptive Low-Rank Adaptation (AdaLoRA)

[AdaLoRA](https://hf.co/papers/2303.10512) manages the parameter budget introduced from LoRA by allocating more parameters - in other words, a higher rank `r` - for important weight matrices that are better adapted for a task and pruning less important ones. The rank is controlled by a method similar to singular value decomposition (SVD). The ∆W is parameterized with two orthogonal matrices and a diagonal matrix which contains singular values. This parametrization method avoids iteratively applying SVD which is computationally expensive. Based on this method, the rank of ∆W is adjusted according to an importance score. ∆W is divided into triplets and each triplet is scored according to its contribution to model performance. Triplets with low importance scores are pruned and triplets with high importance scores are kept for finetuning.
