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

# Quantization

Quantization represents data with fewer bits, making it a useful technique for reducing memory-usage and accelerating inference especially when it comes to large language models (LLMs). There are several ways to quantize a model including:

* optimizing which model weights are quantized with the [AWQ](https://hf.co/papers/2306.00978) algorithm
* independently quantizing each row of a weight matrix with the [GPTQ](https://hf.co/papers/2210.17323) algorithm
* quantizing to 8-bit and 4-bit precision with the [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) library
* quantizing to as low as 2-bit precision with the [AQLM](https://arxiv.org/abs/2401.06118) algorithm

However, after a model is quantized it isn't typically further trained for downstream tasks because training can be unstable due to the lower precision of the weights and activations. But since PEFT methods only add *extra* trainable parameters, this allows you to train a quantized model with a PEFT adapter on top! Combining quantization with PEFT can be a good strategy for training even the largest models on a single GPU. For example, [QLoRA](https://hf.co/papers/2305.14314) is a method that quantizes a model to 4-bits and then trains it with LoRA. This method allows you to finetune a 65B parameter model on a single 48GB GPU!

In this guide, you'll see how to quantize a model to 4-bits and train it with LoRA.

## Quantize a model

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes) is a quantization library with a Transformers integration. With this integration, you can quantize a model to 8 or 4-bits and enable many other options by configuring the [`~transformers.BitsAndBytesConfig`] class. For example, you can:

* set `load_in_4bit=True` to quantize the model to 4-bits when you load it
* set `bnb_4bit_quant_type="nf4"` to use a special 4-bit data type for weights initialized from a normal distribution
* set `bnb_4bit_use_double_quant=True` to use a nested quantization scheme to quantize the already quantized weights
* set `bnb_4bit_compute_dtype=torch.bfloat16` to use bfloat16 for faster computation

```py
import torch
from transformers import BitsAndBytesConfig

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

Pass the `config` to the [`~transformers.AutoModelForCausalLM.from_pretrained`] method.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", quantization_config=config)
```

Next, you should call the [`~peft.utils.prepare_model_for_kbit_training`] function to preprocess the quantized model for training.

```py
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)
```

Now that the quantized model is ready, let's set up a configuration.

## LoraConfig

Create a [`LoraConfig`] with the following parameters (or choose your own):

```py
from peft import LoraConfig

config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

Then use the [`get_peft_model`] function to create a [`PeftModel`] from the quantized model and configuration.

```py
from peft import get_peft_model

model = get_peft_model(model, config)
```

You're all set for training with whichever training method you prefer!

### LoftQ initialization

[LoftQ](https://hf.co/papers/2310.08659) initializes LoRA weights such that the quantization error is minimized, and it can improve performance when training quantized models. To get started, follow [these instructions](https://github.com/huggingface/peft/tree/main/examples/loftq_finetuning).

In general, for LoftQ to work best, it is recommended to target as many layers with LoRA as possible, since those not targeted cannot have LoftQ applied. This means that passing `LoraConfig(..., target_modules="all-linear")` will most likely give the best results. Also, you should use `nf4` as quant type in your quantization config when using 4bit quantization, i.e. `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`.

### QLoRA-style training

QLoRA adds trainable weights to all the linear layers in the transformer architecture. Since the attribute names for these linear layers can vary across architectures, set `target_modules` to `"all-linear"` to add LoRA to all the linear layers:

```py
config = LoraConfig(target_modules="all-linear", ...)
```

## AQLM quantization

Additive Quantization of Language Models ([AQLM](https://arxiv.org/abs/2401.06118)) is a Large Language Models compression method. It quantizes multiple weights together and takes advantage of interdependencies between them. AQLM represents groups of 8-16 weights as a sum of multiple vector codes. This allows it to compress models down to as low as 2-bit with considerably low accuracy losses.

Since the AQLM quantization process is computationally expensive, a use of prequantized models is recommended. A partial list of available models can be found in the official aqlm [repository](https://github.com/Vahe1994/AQLM).

The models support LoRA adapter tuning. To tune the quantized model you'll need to install the `aqlm` inference library: `pip install aqlm>=1.0.2`. Finetuned LoRA adapters shall be saved separately, as merging them with AQLM quantized weights is not possible.

```py
quantized_model = AutoModelForCausalLM.from_pretrained(
    "BlackSamorez/Mixtral-8x7b-AQLM-2Bit-1x16-hf-test-dispatch",
    torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True,
)

peft_config = LoraConfig(...)

quantized_model = get_peft_model(quantized_model, peft_config)
```

You can refer to the [Google Colab](https://colab.research.google.com/drive/12GTp1FCj5_0SnnNQH18h_2XFh9vS_guX?usp=sharing) example for an overview of AQLM+LoRA finetuning.

## Next steps

If you're interested in learning more about quantization, the following may be helpful:

* Learn more about details about QLoRA and check out some benchmarks on its impact in the [Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes) blog post.
* Read more about different quantization schemes in the Transformers [Quantization](https://hf.co/docs/transformers/main/quantization) guide.
