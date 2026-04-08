<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# AdaMSS

[AdaMSS](https://openreview.net/forum?id=8ZdWmpYxT0) (AdaMSS: Adaptive Multi-Subspace Approach for Parameter-Efficient Fine-Tuning) is a parameter-efficient fine-tuning method that decomposes weight matrices using SVD and clusters the decomposed space into multiple trainable subspaces. Each subspace learns independent low-rank updates while the original weights remain frozen. AdaMSS also supports Adaptive Subspace Allocation (ASA), which dynamically prunes less important subspaces during training based on gradient information.

The abstract from the paper is:

> We propose AdaMSS, an adaptive multi-subspace approach for parameter-efficient fine-tuning of large models. Unlike traditional parameterefficient fine-tuning methods that operate within a large single subspace of the network weights, AdaMSS leverages subspace segmentation to obtain multiple smaller subspaces and adaptively reduces the number of trainable parameters during training, ultimately updating only those associated with a small subset of subspaces most relevant to the target downstream task. By using the lowest-rank representation, AdaMSS achieves more compact expressiveness and finer tuning of the model parameters. Theoretical analyses demonstrate that AdaMSS has better generalization guarantee than LoRA, PiSSA, and other single-subspace low-rankbased methods. Extensive experiments across image classification, natural language understanding, and natural language generation tasks show that AdaMSS achieves comparable performance to full fine-tuning and outperforms other parameterefficient fine-tuning methods in most cases, all while requiring fewer trainable parameters. Notably, on the ViT-Large model, AdaMSS achieves 4.7% higher average accuracy than LoRA across seven tasks, using just 15.4% of the trainable parameters. On RoBERTa-Large, AdaMSS outperforms PiSSA by 7% in average accuracy across six tasks while reducing the number of trainable parameters by approximately 94.4%. These results demonstrate the effectiveness of AdaMSS in parameter-efficient fine-tuning. The code for AdaMSS is available at https: //github.com/jzheng20/AdaMSS.


AdaMSS currently has the following constraints:
- Only `nn.Linear` layers are supported.
- Requires scikit-learn for the KMeans clustering step.

If these constraints don't work for your use case, consider other methods instead.

## AdamssConfig

[[autodoc]] tuners.adamss.config.AdamssConfig

## AdamssModel

[[autodoc]] tuners.adamss.model.AdamssModel
