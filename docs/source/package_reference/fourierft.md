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

# FourierFT: Discrete Fourier Transformation Fine-Tuning

[FourierFT](https://huggingface.co/papers/2405.03003) is a parameter-efficient fine-tuning technique that leverages Discrete Fourier Transform to compress the model's tunable weights. This method outperforms LoRA in the GLUE benchmark and common ViT classification tasks using much less parameters.

FourierFT currently has the following constraints:

- Only `nn.Linear` layers are supported.
- Quantized layers are not supported.

If these constraints don't work for your use case, consider other methods instead.

The abstract from the paper is:

> Low-rank adaptation (LoRA) has recently gained much interest in fine-tuning foundation models. It effectively reduces the number of trainable parameters by incorporating low-rank matrices A and B to represent the weight change, i.e., Delta W=BA. Despite LoRA's progress, it faces storage challenges when handling extensive customization adaptations or larger base models. In this work, we aim to further compress trainable parameters by enjoying the powerful expressiveness of the Fourier transform. Specifically, we introduce FourierFT, which treats Delta W as a matrix in the spatial domain and learns only a small fraction of its spectral coefficients. With the trained spectral coefficients, we implement the inverse discrete Fourier transform to recover Delta W. Empirically, our FourierFT method shows comparable or better performance with fewer parameters than LoRA on various tasks, including natural language understanding, natural language generation, instruction tuning, and image classification. For example, when performing instruction tuning on the LLaMA2-7B model, FourierFT surpasses LoRA with only 0.064M trainable parameters, compared to LoRA's 33.5M.

## FourierFTConfig

[[autodoc]] tuners.fourierft.config.FourierFTConfig

## FourierFTModel

[[autodoc]] tuners.fourierft.model.FourierFTModel
