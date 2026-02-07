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

# TinyLoRA: Learning to Reason in 13 Parameters

[TinyLoRA](https://huggingface.co/papers/2602.04118) is an extremely parameter-efficient fine-tuning technique that enables fine-tuning with as few as 1-13 trainable parameters. It builds upon the [LoRA-XS](https://huggingface.co/papers/2405.17604) approach by using SVD decomposition of frozen weights and projecting a tiny trainable vector through fixed random tensors.

The key innovation of TinyLoRA is replacing the trainable low-rank matrix R with a weighted sum of fixed random projection matrices: `R = Σᵢ vᵢ Pᵢ`, where `v` is a tiny trainable vector and `Pᵢ` are fixed random matrices. This dramatically reduces the number of trainable parameters while maintaining competitive performance, especially when combined with reinforcement learning training methods like GRPO.

TinyLoRA supports weight tying through the `ntie` parameter, which controls how many modules share the same trainable vector `v`. With full tying across all target modules, you can achieve extreme parameter efficiency with just `u` trainable parameters for the entire model.

When saving the adapter parameters, it's possible to eschew storing the random projection matrices by setting `save_projection=False` on the `TinyLoraConfig`. In that case, these matrices will be restored based on the fixed random seed from the `projection_seed` argument. This cuts down on the size of the checkpoint, but we cannot guarantee reproducibility on all devices and for all future versions of PyTorch. If you want to ensure reproducibility, set `save_projection=True` (which is the default).

TinyLoRA currently has the following constraints:

- Only `nn.Linear`, `nn.Embedding`, and `transformers.pytorch_utils.Conv1D` layers are supported.

The abstract from the paper is:

> Recent research has shown that language models can learn to reason, often via reinforcement learning. Some work even trains low-rank parameterizations for reasoning, but conventional LoRA cannot scale below the model dimension. We question whether even rank=1 LoRA is necessary for learning to reason and propose TinyLoRA, a method for scaling low-rank adapters to sizes as small as one parameter. Within our new parameterization, we are able to train the 8B parameter size of Qwen2.5 to 91% accuracy on GSM8K with only 13 trained parameters in bf16 (26 total bytes). We find this trend holds in general: we are able to recover 90% of performance improvements while training 1000x fewer parameters across a suite of more difficult learning-to-reason benchmarks such as AIME, AMC, and MATH500. Notably, we are only able to achieve such strong performance with RL: models trained using SFT require 100-1000x larger updates to reach the same performance.

## TinyLoraConfig

[[autodoc]] tuners.tinylora.config.TinyLoraConfig

## TinyLoraModel

[[autodoc]] tuners.tinylora.model.TinyLoraModel
