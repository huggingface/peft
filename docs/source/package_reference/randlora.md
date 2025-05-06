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

# RandLora: Full-rank parameter-efficient fine-tuning of large models 
[RandLora](https://huggingface.co/papers/2502.00987) is a parameter-efficient fine-tuning technique that is similar to [LoRA](https://huggingface.co/papers/2106.09685) and [VeRA](https://huggingface.co/papers/2310.11454) but performs full rank updates to improve performance. RandLora can be particulary usefull when adapting large model to hard tasks that require complex updates while preserving the parameter efficiency of LoRA. The full rank update of RandLora is achieved by linearly scaling random bases. The random bases are a collection of multiple low rank matrices such that the summation of their ranks if greater or equal to the full rank of the parameter matrices. The trainable parameters of RandLora are two diagonal matrices (vectors) that get multiplied with the right hand low rank random bases, in a similar way to VeRA's update. To maintain low memory usage, RandLora uses a custom function that prevents storing unnecessary bases in memory for backpropagation.

RandLora presents the noteworthy difference that contrary to other LoRA-like PEFT algorithm, increasing RandLora's random base ranks increases the amount of trainable parameters. Because number of bases x bases rank is constant in RandLora, reducing the rank will increase the number of random bases, hence the number of base-specific trainable diagonal bases.

Because reducing the rank of RandLora's random bases will increase their number, RandLora can become slower to train than LoRA for very small ranks where typically, ranks below 4 with result in a large training time increase. This does not affect inference though as the RandLora adapters can be merged into the pretrained weight matrices.

RandLora additionally supports training with sparse, ternary random bases (only containing -1, 0 and 1). These bases are as described in [Bingham et al.](https://cs-people.bu.edu/evimaria/cs565/kdd-rp.pdf) and [Ping et al.](https://hastie.su.domains/Papers/Ping/KDD06_rp.pdf) and could theoretically be used to reduce compute needs by performing aggregations instead of matrix multiplications to create the weight update. This is not currently supported. Although it does not currently reduce compute, using sparse random bases in RandLora can reduce overfitting in some cases. For users intersted in using sparse ternary bases, the `sparse` option is recommended over the `very_sparse` one that can reduce perfromance. 

Similarly to VeRA, when saving the RandLora's parameters, it's possible to eschew storing the low rank matrices by setting `save_projection=False` on the `VeraConfig`. In that case, these matrices will be restored based on the fixed random seed from the `projection_prng_key` argument. This cuts down on the size of the checkpoint, but we cannot guarantee reproducibility on all devices and for all future versions of PyTorch. If you want to ensure reproducibility, set `save_projection=True` (which is the default).

As in Vera and to handle different shapes of adapted layers, RandLora initializes shared A and B matrices with the largest required size for each dimension. During the forward pass, submatrices A and B for a given layer are sliced out from these shared matrices and used as described in the paper. For example, adapting two linear layers of shapes (100, 20) and (80, 50) will create A and B matrices of shapes (rank, 50) and (100, rank) respectively. Then, to adapt a layer of shape (100, 20), submatrices A and B of shapes (rank, 20) and (100, rank) will be extracted.

RandLora currently has the following constraint:

- Only `nn.Linear` layers are supported.

The abstract from the paper is:

> Low-Rank Adaptation (LoRA) and its variants have shown impressive results in reducing the number of trainable parameters and memory requirements of large transformer networks while maintaining fine-tuning performance. The low-rank nature of the weight update inherently limits the representation power of fine-tuned models, however, thus potentially compromising performance on complex tasks. This raises a critical question: when a performance gap between LoRA and standard fine-tuning is observed, is it due to the reduced number of trainable parameters or the rank deficiency?
This paper aims to answer this question by introducing RandLora, a parameter-efficient method that performs full-rank updates using a learned linear combinations of low-rank, non-trainable random matrices. Our method limits the number of trainable parameters by restricting optimization to diagonal scaling matrices applied to the fixed random matrices. This allows us to effectively overcome the low-rank limitations while maintaining parameter and memory efficiency during training. Through extensive experimentation across vision, language, and vision-language benchmarks, we systematically evaluate the limitations of LoRA and existing random basis methods. Our findings reveal that full-rank updates are beneficial across vision and language tasks individually, and even more so for vision-language tasks, where RandLora significantly reduces---and sometimes eliminates---the performance gap between standard fine-tuning and LoRA, demonstrating its efficacy.

## RandLoraConfig

[[autodoc]] tuners.randlora.config.RandLoraConfig

## RandLoraModel

[[autodoc]] tuners.randlora.model.RandLoraModel
