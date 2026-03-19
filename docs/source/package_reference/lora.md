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

# LoRA

Low-Rank Adaptation ([LoRA](https://huggingface.co/papers/2309.15223)) is a PEFT method that decomposes a large matrix into two smaller low-rank matrices in the attention layers. This drastically reduces the number of parameters that need to be fine-tuned.

The abstract from the paper is:

*We propose a neural language modeling system based on low-rank adaptation (LoRA) for speech recognition output rescoring. Although pretrained language models (LMs) like BERT have shown superior performance in second-pass rescoring, the high computational cost of scaling up the pretraining stage and adapting the pretrained models to specific domains limit their practical use in rescoring. Here we present a method based on low-rank decomposition to train a rescoring BERT model and adapt it to new domains using only a fraction (0.08%) of the pretrained parameters. These inserted matrices are optimized through a discriminative training objective along with a correlation-based regularization loss. The proposed low-rank adaptation Rescore-BERT (LoRB) architecture is evaluated on LibriSpeech and internal datasets with decreased training times by factors between 5.4 and 3.6.*.

## LoraConfig

[[autodoc]] tuners.lora.config.LoraConfig

## LoraModel

[[autodoc]] tuners.lora.model.LoraModel


## Utility

### ArrowConfig

[[autodoc]] tuners.lora.config.ArrowConfig

### LoftQ

[[autodoc]] utils.loftq_utils.replace_lora_weights_loftq


### Eva

#### EvaConfig

[[autodoc]] tuners.lora.config.EvaConfig

#### initialize_lora_eva_weights

[[autodoc]] tuners.lora.eva.initialize_lora_eva_weights

#### get_eva_state_dict

[[autodoc]] tuners.lora.eva.get_eva_state_dict


## Variants

### LoRA-GA

[LoRA-GA](https://hf.co/papers/2407.05000) (Low-Rank Adaptation with Gradient Approximation) improves upon standard LoRA by using gradient information during initialization to achieve faster convergence. Instead of random initialization, LoRA-GA performs SVD on estimated gradients to initialize adapter weights in a direction that aligns with full fine-tuning, resulting in 2-4x faster convergence with the same final performance.

The abstract from the paper is:

*Low-rank adaptation (LoRA) is a popular technique for parameter-efficient fine-tuning of large language models. However, LoRA's random initialization of adapter weights leads to slow convergence during the initial training phase. In this paper, we propose LoRA-GA (Low-Rank Adaptation with Gradient Approximation), a novel initialization method that leverages gradient information to initialize LoRA adapters. Specifically, we estimate gradients on a small set of training samples and perform singular value decomposition (SVD) to extract principal components. These components are used to initialize the adapter matrices, aligning the initial update direction with that of full fine-tuning. Our experiments across various tasks and model scales demonstrate that LoRA-GA achieves 2-4x faster convergence compared to standard LoRA while maintaining the same final performance. The method is orthogonal to existing LoRA variants and can be easily integrated with techniques like DoRA and LoRA+.*

#### Usage Tips

- **Gradient Estimation**: LoRA-GA requires a gradient estimation phase before model initialization. Use `preprocess_loraga()` with a `train_step` callback to compute gradients over a small number of training batches (typically 64-128 batches).


- **Initialization Strategies**: LoRA-GA supports four direction strategies (`direction`): `"ArBr"`, `"A2rBr"`, `"ArB2r"` (default), and `"random"`, and four scaling strategies (`scale`): `"stable"` (default), `"weight_svd"`, `"gd_scale"`, and `"unit"`. The default combination provides the best balance of convergence speed and stability.

- **Base Weight Modification**: Unlike standard LoRA, LoRA-GA modifies the base model weights during initialization by subtracting a scaled version of the low-rank approximation. This enables better alignment with full fine-tuning gradients. Since base weights are modified, use `save_pretrained()` with the `save_embedding_layers` argument or `save_mutated_as_lora` pattern to properly save the adapter.

- **Computational Overhead**: The gradient estimation adds a small overhead during initialization (typically 1-2 minutes for 64 batches), but this is quickly amortized by faster convergence during training.

- **Compatibility**: LoRA-GA requires full-precision weights and does not support quantized models. Can be combined with other LoRA variants like DoRA.

#### LoraGAConfig

[[autodoc]] tuners.lora.config.LoraGAConfig

#### Utilities

[[autodoc]] tuners.lora.loraga.estimate_gradients

[[autodoc]] tuners.lora.loraga.preprocess_loraga


## Intruder Dimension Reduction

[[autodoc]] tuners.lora.intruders.reduce_intruder_dimension
