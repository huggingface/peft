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

# C3A: Parameter-Efficient Fine-Tuning via Circular Convolution

[C3A](https://huggingface.co/papers/2407.19342) is a parameter-efficient fine-tuning technique that leverages Circular Convolution to achieve high rank adaptation within reasonable resource limits.

Note that you should use a much larger learning rate (LR) for C3A than for other methods. For example, a LR of 1e-1 for C3A is a good starting point. Besides, a much smaller weight decay should be used. You can refer to the `method_comparison` folder for more details.

For the `block_size`, it affects tunable parameters and performance. To start with, you can choose a $\mathrm{gcd}(d_1,d_2)$ near $\frac{\sqrt{d_1\times d_2}}{r}$, where $r$ is the rank for LoRA you would use for this task.

C3A currently has the following constraints:

- Only `nn.Linear` layers are supported.
- Quantized layers are not supported.
- The block size should be a common divisor of both the input and output sizes of target layers. 

If these constraints don't work for your use case, consider other methods instead.

The abstract from the paper is:

> Low-Rank Adaptation (LoRA) has gained popularity for fine-tuning large foundation models, leveraging low-rank matrices $\mathbf{A}$ and $\mathbf{B}$ to represent weight changes (i.e., $\Delta \mathbf{W} = \mathbf{B} \mathbf{A}$). This method reduces trainable parameters and mitigates heavy memory consumption associated with full delta matrices by sequentially multiplying $\mathbf{A}$ and $\mathbf{B}$ with the activation. Despite its success, the intrinsic low-rank characteristic may limit its performance. Although several variants have been proposed to address this issue, they often overlook the crucial computational and memory efficiency brought by LoRA. In this paper, we propose Circular Convolution Adaptation (C3A), which not only achieves high-rank adaptation with enhanced performance but also excels in both computational power and memory utilization. Extensive experiments demonstrate that C3A consistently outperforms LoRA and its variants across various fine-tuning tasks. 

## C3AConfig

[[autodoc]] tuners.c3a.config.C3AConfig

## C3AModel

[[autodoc]] tuners.c3a.model.C3AModel
