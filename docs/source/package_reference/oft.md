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

# OFT

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/oft.png"/>
</div>
<small><a href="https://hf.co/papers/2306.07280">Controlling Text-to-Image Diffusion by Orthogonal Finetuning</a></small>

[Orthogonal Finetuning (OFT)](https://hf.co/papers/2306.07280) and [OFTv2](https://huggingface.co/papers/2506.19847) is a method developed for adapting text-to-image diffusion models. It works by reparameterizing the pretrained weight matrices with its orthogonal matrix to preserve information in the pretrained model. To reduce the number of parameters, OFT introduces a block-diagonal structure in the orthogonal matrix. The method primarily focuses on preserving a pretrained model's generative performance in the finetuned model. It tries to maintain the same cosine similarity ([hyperspherical energy](https://huggingface.co/papers/1805.09298)) between all pairwise neurons in a layer because this better captures the semantic information among neurons. This means OFT is more capable at preserving the subject and it is better for controllable generation (similar to [ControlNet](https://huggingface.co/docs/diffusers/using-diffusers/controlnet)).

The abstract from the paper is:

*Large text-to-image diffusion models have impressive capabilities in generating photorealistic images from text prompts. How to effectively guide or control these powerful models to perform different downstream tasks becomes an important open problem. To tackle this challenge, we introduce a principled finetuning method -- Orthogonal Finetuning (OFT), for adapting text-to-image diffusion models to downstream tasks. Unlike existing methods, OFT can provably preserve hyperspherical energy which characterizes the pairwise neuron relationship on the unit hypersphere. We find that this property is crucial for preserving the semantic generation ability of text-to-image diffusion models. To improve finetuning stability, we further propose Constrained Orthogonal Finetuning (COFT) which imposes an additional radius constraint to the hypersphere. Specifically, we consider two important finetuning text-to-image tasks: subject-driven generation where the goal is to generate subject-specific images given a few images of a subject and a text prompt, and controllable generation where the goal is to enable the model to take in additional control signals. We empirically show that our OFT framework outperforms existing methods in generation quality and convergence speed*.

OFT preserves the hyperspherical energy by learning an orthogonal transformation for neurons to keep the cosine similarity between them unchanged, potentially leading to less forgetting of previous learnt knowledge. In practice, this means taking the matrix product of an orthogonal matrix with the pretrained weight matrix. However, to be parameter-efficient, the orthogonal matrix is represented as a block-diagonal matrix with rank `r` blocks. Whereas LoRA reduces the number of trainable parameters with low-rank structures, OFT reduces the number of trainable parameters with a sparse block-diagonal matrix structure.

## Benchmark overview

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=OFT"
	frameborder="0"
	width="850"
	height="1000"
></iframe>

## Merge OFT weights into the base model

Similar to LoRA, the weights learned by OFT can be integrated into the pretrained weight matrices using the [`~OFTModel.merge_and_unload()` function. This function merges the adapter weights with the base model which allows you to effectively use the newly merged model as a standalone model.

## OFT Example Usage

For using OFT for quantized finetuning with [TRL](https://github.com/huggingface/trl) for `SFT`, `PPO`, or `DPO` fine-tuning, follow the following outline:

```py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer
from peft import OFTConfig

if use_quantization:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained("model_name")

# Configure OFT
peft_config = OFTConfig(
    oft_block_size=32,
    use_cayley_neumann=True,
    target_modules="all-linear",
    bias="none",
    task_type="CAUSAL_LM"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=ds['train'],
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_arguments,
    data_collator=collator,
)

trainer.train()
```

# API

## OFTConfig

[[autodoc]] tuners.oft.config.OFTConfig

## OFTModel

[[autodoc]] tuners.oft.model.OFTModel
