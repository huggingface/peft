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

# WaveFT: Wavelet Fine-Tuning

[WaveFT](https://huggingface.co/papers/2505.12532) is a novel parameter-efficient fine-tuning (PEFT) method that introduces sparse updates in the **wavelet domain** of residual matrices. Unlike LoRA, which is constrained by discrete low-rank choices, WaveFT enables fine-grained control over the number of trainable parameters by directly learning a sparse set of coefficients in the transformed space. These coefficients are then mapped back to the weight domain via the Inverse Discrete Wavelet Transform (IDWT), producing high-rank updates without incurring inference overhead.

WaveFT currently has the following constraint:

- Only `nn.Linear` layers are supported.

The abstract from the paper is:

>Efficiently adapting large foundation models is critical, especially with tight compute and memory budgets. Parameter-Efficient Fine-Tuning (PEFT) methods such as LoRA offer limited granularity and effectiveness in few-parameter regimes. We propose Wavelet Fine-Tuning (WaveFT), a novel PEFT method that learns highly sparse updates in the wavelet domain of residual matrices. WaveFT allows precise control of trainable parameters, offering fine-grained capacity adjustment and excelling with remarkably low parameter count, potentially far fewer than LoRA’s minimum—ideal for extreme parameter-efficient scenarios. Evaluated on personalized text-to-image generation using Stable Diffusion XL as baseline, WaveFT significantly outperforms LoRA and other PEFT methods, especially at low parameter counts; achieving superior subject fidelity, prompt alignment, and image diversity.

## WaveFTConfig

[[autodoc]] tuners.waveft.config.WaveFTConfig

## WaveFTModel

[[autodoc]] tuners.waveft.model.WaveFTModel
