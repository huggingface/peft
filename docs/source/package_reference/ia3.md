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

# IA3

> [!TIP]
> IA3 is an excellent choice when you need the smallest possible adapter size — it uses learned vectors instead of low-rank matrices, making it one of the most parameter-efficient methods available. If your priority is minimal disk/memory footprint and fast training, IA3 is a strong candidate.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/ia3.png"/>
</div>
<small>IA3 introduces three vectors, lv, lk and lff to scale value, key and feed-forward activations <a href="https://hf.co/papers/2205.05638">(image source)</a>.</small>

Infused Adapter by Inhibiting and Amplifying Inner Activations, or [IA3](https://hf.co/papers/2205.05638), is a method that adds three learned vectors to rescale the keys and values of the self-attention and encoder-decoder attention layers, and the intermediate activation of the position-wise feed-forward network.

The abstract from the paper is:

*Few-shot in-context learning (ICL) enables pre-trained language models to perform a previously-unseen task without any gradient-based training by feeding a small number of training examples as part of the input. ICL incurs substantial computational, memory, and storage costs because it involves processing all of the training examples every time a prediction is made. Parameter-efficient fine-tuning (PEFT) (e.g. adapter modules, prompt tuning, sparse update methods, etc.) offers an alternative paradigm where a small set of parameters are trained to enable a model to perform the new task. In this paper, we rigorously compare few-shot ICL and PEFT and demonstrate that the latter offers better accuracy as well as dramatically lower computational costs. Along the way, we introduce a new PEFT method called (IA)^3 that scales activations by learned vectors, attaining stronger performance while only introducing a relatively tiny amount of new parameters. We also propose a simple recipe based on the T0 model called T-Few that can be applied to new tasks without task-specific tuning or modifications. We validate the effectiveness of T-Few on completely unseen tasks by applying it to the RAFT benchmark, attaining super-human performance for the first time and outperforming the state-of-the-art by 6% absolute. All of the code used in our experiments is publicly available*.

To make fine-tuning more efficient, IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)
rescales inner activations with learned vectors. These learned vectors are injected in the attention and feedforward modules
in a typical transformer-based architecture. These learned vectors are the only trainable parameters during fine-tuning, and thus the original
weights remain frozen. Dealing with learned vectors (as opposed to learned low-rank updates to a weight matrix like LoRA)
keeps the number of trainable parameters much smaller.

Being similar to [LoRA](./lora), IA3 carries many of the same advantages:

* IA3 makes fine-tuning more efficient by drastically reducing the number of trainable parameters. (For T0, an IA3 model only has about 0.01% trainable parameters, while even LoRA has > 0.1%)
* The original pre-trained weights are kept frozen, which means you can have multiple lightweight and portable IA3 models for various downstream tasks built on top of them.
* Performance of models fine-tuned using IA3 is comparable to the performance of fully fine-tuned models.
* IA3 does not add any inference latency because adapter weights can be merged with the base model.

In principle, IA3 can be applied to any subset of weight matrices in a neural network to reduce the number of trainable
parameters. Following the authors' implementation, IA3 weights are added to the key, value and feedforward layers
of a Transformer model. To be specific, for transformer models, IA3 weights are added to the outputs of key and value layers, and to the input of the second feedforward layer
in each transformer block.

Given the target layers for injecting IA3 parameters, the number of trainable parameters
can be determined based on the size of the weight matrices.

## Usage

> [!TIP]
> For autoregressive models, target `k_proj`, `v_proj`, and `down_proj` — these are the layers the authors found to be most effective. Set `feedforward_modules=["down_proj"]` so IA3 knows which modules are feedforward layers and can apply the correct scaling.

For the task of sequence classification, one can initialize the IA3 config for a Llama model as follows:

```py
from peft import IA3Config, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
peft_config = IA3Config(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["k_proj", "v_proj", "down_proj"],
    feedforward_modules=["down_proj"],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```


### When to use IA3 vs LoRA

IA3 is particularly effective when you need the smallest possible adapters for storage-constrained deployments, or when you want fast convergence with fewer trainable parameters. LoRA may be preferable when you need the broader ecosystem of LoRA variants (DoRA, QLoRA, etc.) or want the most widely-tested approach with extensive community support.

## Benchmark overview

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=IA3"
	frameborder="0"
	width="850"
	height="1000"
></iframe>


# API

## IA3Config

[[autodoc]] tuners.ia3.config.IA3Config

## IA3Model

[[autodoc]] tuners.ia3.model.IA3Model
