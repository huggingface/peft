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

# VeRA: Vector-based Random Matrix Adaptation

[VeRA](https://huggingface.co/papers/2310.11454) is a parameter-efficient fine-tuning technique that is similar to LoRA but requires even fewer extra parameters while promising similar or even better performance. As such, it is particularly useful when the parameter budget is very limited, e.g. when scaling to very large models. The reduction of the count of trainable parameters is achieved by sharing the same low-rank matrices across all layers, and only training two additional vectors per layer.

## How VeRA works

LoRA updates a frozen base model by learning two small, low-rank matrices for each adapted layer. The rank is the
inner dimension of these matrices, and it controls how much capacity the adapter has. Increasing the rank or adapting
more layers increases the number of trainable LoRA parameters because each layer receives its own pair of matrices.

VeRA keeps the same low-rank adaptation idea but changes which parameters are trained. Instead of learning separate
low-rank matrices for every layer, VeRA uses one frozen, randomly initialized pair of low-rank matrices, `A` and `B`,
that is shared across layers. Each adapted layer only learns two scaling vectors, commonly written as `d` and `b`,
which rescale the shared matrices to produce a layer-specific update.

This is why VeRA can use fewer trainable parameters than LoRA. In the simplified parameter count from the paper, LoRA
scales with `2 * L_tuned * d_model * r`, where `L_tuned` is the number of adapted layers, `d_model` is the model
dimension, and `r` is the rank. VeRA scales with `L_tuned * (d_model + r)` because the large low-rank matrices are
shared and frozen, while only the smaller per-layer vectors are trained.

When saving the adapter parameters, it's possible to eschew storing the low rank matrices by setting `save_projection=False` on the `VeraConfig`. In that case, these matrices will be restored based on the fixed random seed from the `projection_prng_key` argument. This cuts down on the size of the checkpoint, but we cannot guarantee reproducibility on all devices and for all future versions of PyTorch. If you want to ensure reproducibility, set `save_projection=True` (which is the default).

To handle different shapes of adapted layers, VeRA initializes shared A and B matrices with the largest required size for each dimension. During the forward pass, submatrices A and B for a given layer are sliced out from these shared matrices and used as described in the paper. For example, adapting two linear layers of shapes (100, 20) and (80, 50) will create A and B matrices of shapes (rank, 50) and (100, rank) respectively. Then, to adapt a layer of shape (100, 20), submatrices A and B of shapes (rank, 20) and (100, rank) will be extracted.

VeRA currently has the following constraint:

- Only `nn.Linear` layers are supported.

The abstract from the paper is:

> Low-rank adaptation (LoRA) is a popular method that reduces the number of trainable parameters when finetuning large language models, but still faces acute storage challenges when scaling to even larger models or deploying numerous per-user or per-task adapted models. In this work, we present Vector-based Random Matrix Adaptation (VeRA), which significantly reduces the number of trainable parameters compared to LoRA, yet maintains the same performance. It achieves this by using a single pair of low-rank matrices shared across all layers and learning small scaling vectors instead. We demonstrate its effectiveness on the GLUE and E2E benchmarks, image classification tasks, and show its application in instruction-tuning of 7B and 13B language models.

## Benchmark overview

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=VERA"
	frameborder="0"
	width="850"
	height="1000"
></iframe>

# API

## VeRAConfig

[[autodoc]] tuners.vera.config.VeraConfig

## VeRAModel

[[autodoc]] tuners.vera.model.VeraModel
