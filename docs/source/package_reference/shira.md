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

# Sparse High Rank Adapters

Sparse High Rank Adapters or [SHiRA](https://arxiv.org/abs/2406.13175) is an alternate type of adapter and has been found to have significant advantages over the low rank adapters. Specifically, SHiRA achieves better accuracy than LoRA for a variety of vision and language tasks. It also offers simpler and higher quality multi-adapter fusion by significantly reducing concept loss, a common problem faced by low rank adapters. SHiRA directly finetunes a small number of the base model's parameters to finetune the model on any adaptation task.

SHiRA currently has the following constraint:

- Only `nn.Linear` layers are supported.

The abstract from the paper is:

> Low Rank Adaptation (LoRA) has gained massive attention in the recent generative AI research. One of the main advantages of LoRA is its ability to be fused with pretrained models, adding no overhead during inference. However, from a mobile deployment standpoint, we can either avoid inference overhead in the fused mode but lose the ability to switch adapters rapidly, or suffer significant (up to 30% higher) inference latency while enabling rapid switching in the unfused mode. LoRA also exhibits concept-loss when multiple adapters are used concurrently. In this paper, we propose Sparse High Rank Adapters (SHiRA), a new paradigm which incurs no inference overhead, enables rapid switching, and significantly reduces concept-loss. Specifically, SHiRA can be trained by directly tuning only 1-2% of the base model weights while leaving others unchanged. This results in a highly sparse adapter which can be switched directly in the fused mode. We further provide theoretical and empirical insights on how high sparsity in SHiRA can aid multi-adapter fusion by reducing concept loss. Our extensive experiments on LVMs and LLMs demonstrate that finetuning only a small fraction of the parameters in the base model significantly outperforms LoRA while enabling both rapid switching and multi-adapter fusion. Finally, we provide a latency- and memory-efficient SHiRA implementation based on Parameter-Efficient Finetuning (PEFT) Library which trains at nearly the same speed as LoRA while consuming up to 16% lower peak GPU memory, thus making SHiRA easy to adopt for practical use cases. To demonstrate rapid switching benefits during inference, we show that loading SHiRA on a base model can be 5x-16x faster than LoRA fusion on a CPU.

## ShiraConfig

[[autodoc]] tuners.shira.config.ShiraConfig

## ShiraModel

[[autodoc]] tuners.shira.model.ShiraModel
