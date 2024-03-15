<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<h1 align="center"> <p>🤗 PEFT</p></h1>
<h3 align="center">
    <p>State-of-the-art Parameter-Efficient Fine-Tuning (PEFT) methods</p>
</h3>

Fine-tuning large pretrained models is often prohibitively costly due to their scale. Parameter-Efficient Fine-Tuning (PEFT) methods enable efficient adaptation of large pretrained models to various downstream applications by only fine-tuning a small number of (extra) model parameters instead of all the model's parameters. This significantly decreases the computational and storage costs. Recent state-of-the-art PEFT techniques achieve performance comparable to fully fine-tuned models.

PEFT is integrated with Transformers for easy model training and inference, Diffusers for conveniently managing different adapters, and Accelerate for distributed training and inference for really big models.

> [!TIP]
> Visit the [PEFT](https://huggingface.co/PEFT) organization to read about the PEFT methods implemented in the library and to see notebooks demonstrating how to apply these methods to a variety of downstream tasks. Click the "Watch repos" button on the organization page to be notified of newly implemented methods and notebooks!

Check the PEFT Adapters API Reference section for a list of supported PEFT methods, and read the [Adapters](https://huggingface.co/docs/peft/en/conceptual_guides/adapter), [Soft prompts](https://huggingface.co/docs/peft/en/conceptual_guides/prompting), and [IA3](https://huggingface.co/docs/peft/en/conceptual_guides/ia3) conceptual guides to learn more about how these methods work.

## Quickstart

Install PEFT from pip:

```bash
pip install peft
```

Prepare a model for training with a PEFT method such as LoRA by wrapping the base model and PEFT configuration with `get_peft_model`. For the bigscience/mt0-large model, you're only training 0.19% of the parameters!

```python
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282"
```

To load a PEFT model for inference:

```py
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

model = AutoPeftModelForCausalLM.from_pretrained("ybelkada/opt-350m-lora").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

model.eval()
inputs = tokenizer("Preheat the oven to 350 degrees and place the cookie dough", return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

"Preheat the oven to 350 degrees and place the cookie dough in the center of the oven. In a large bowl, combine the flour, baking powder, baking soda, salt, and cinnamon. In a separate bowl, combine the egg yolks, sugar, and vanilla."
```

## Why you should use PEFT

There are many benefits of using PEFT but the main one is the huge savings in compute and storage, making PEFT applicable to many different use cases.

### High performance on consumer hardware

Consider the memory requirements for training the following models on the [ought/raft/twitter_complaints](https://huggingface.co/datasets/ought/raft/viewer/twitter_complaints) dataset with an A100 80GB GPU with more than 64GB of CPU RAM.

|   Model         | Full Finetuning | PEFT-LoRA PyTorch  | PEFT-LoRA DeepSpeed with CPU Offloading |
| --------- | ---- | ---- | ---- |
| bigscience/T0_3B (3B params) | 47.14GB GPU / 2.96GB CPU  | 14.4GB GPU / 2.96GB CPU | 9.8GB GPU / 17.8GB CPU |
| bigscience/mt0-xxl (12B params) | OOM GPU | 56GB GPU / 3GB CPU | 22GB GPU / 52GB CPU |
| bigscience/bloomz-7b1 (7B params) | OOM GPU | 32GB GPU / 3.8GB CPU | 18.1GB GPU / 35GB CPU |

With LoRA you can fully finetune a 12B parameter model that would've otherwise run out of memory on the 80GB GPU, and comfortably fit and train a 3B parameter model. When you look at the 3B parameter model's performance, it is comparable to a fully finetuned model at a fraction of the GPU memory.

|   Submission Name        | Accuracy |
| --------- | ---- |
| Human baseline (crowdsourced) |	0.897 |
| Flan-T5 | 0.892 |
| lora-t0-3b | 0.863 |

> [!TIP]
> The bigscience/T0_3B model performance isn't optimized in the table above. You can squeeze even more performance out of it by playing around with the input instruction templates, LoRA hyperparameters, and other training related hyperparameters. The final checkpoint size of this model is just 19MB compared to 11GB of the full bigscience/T0_3B model. Learn more about the advantages of finetuning with PEFT in this [blog post](https://www.philschmid.de/fine-tune-flan-t5-peft).

### Quantization

Quantization is another method for reducing the memory requirements of a model by representing the data in a lower precision. It can be combined with PEFT methods to make it even easier to train and load LLMs for inference.

* Learn how to finetune [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) with QLoRA and the [TRL](https://huggingface.co/docs/trl/index) library on a 16GB GPU in the [Finetune LLMs on your own consumer hardware using tools from PyTorch and Hugging Face ecosystem](https://pytorch.org/blog/finetune-llms/) blog post.
* Learn how to finetune a [openai/whisper-large-v2](https://huggingface.co/openai/whisper-large-v2) model for multilingual automatic speech recognition with LoRA and 8-bit quantization in this [notebook](https://colab.research.google.com/drive/1DOkD_5OUjFa0r5Ik3SgywJLJtEo2qLxO?usp=sharing) (see this [notebook](https://colab.research.google.com/drive/1vhF8yueFqha3Y3CpTHN6q9EVcII9EYzs?usp=sharing) instead for an example of streaming a dataset).

### Save compute and storage

PEFT can help you save storage by avoiding full finetuning of models on each of downstream task or dataset. In many cases, you're only finetuning a very small fraction of a model's parameters and each checkpoint is only a few MBs in size (instead of GBs). These smaller PEFT adapters demonstrate performance comparable to a fully finetuned model. If you have many datasets, you can save a lot of storage with a PEFT model and not have to worry about catastrophic forgetting or overfitting the backbone or base model.

## PEFT integrations

PEFT is widely supported across the Hugging Face ecosystem because of the massive efficiency it brings to training and inference.

### Diffusers

The iterative diffusion process consumes a lot of memory which can make it difficult to train. PEFT can help reduce the memory requirements and reduce the storage size of the final model checkpoint. For example, consider the memory required for training a Stable Diffusion model with LoRA on an A100 80GB GPU with more than 64GB of CPU RAM. The final model checkpoint size is only 8.8MB!

|   Model         | Full Finetuning | PEFT-LoRA  | PEFT-LoRA with Gradient Checkpointing  |
| --------- | ---- | ---- | ---- |
| CompVis/stable-diffusion-v1-4 | 27.5GB GPU / 3.97GB CPU | 15.5GB GPU / 3.84GB CPU | 8.12GB GPU / 3.77GB CPU | 

> [!TIP]
> Take a look at the [examples/lora_dreambooth/train_dreambooth.py](examples/lora_dreambooth/train_dreambooth.py) training script to try training your own Stable Diffusion model with LoRA, and play around with the [smangrul/peft-lora-sd-dreambooth](https://huggingface.co/spaces/smangrul/peft-lora-sd-dreambooth) Space which is running on a T4 instance. Learn more about the PEFT integration in Diffusers in this [tutorial](https://huggingface.co/docs/peft/main/en/tutorial/peft_integrations#diffusers).

### Accelerate

[Accelerate](https://huggingface.co/docs/accelerate/index) is a library for distributed training and inference on various training setups and hardware (GPUs, TPUs, Apple Silicon, etc.). PEFT models work with Accelerate out of the box, making it really convenient to train really large models or use them for inference on consumer hardware with limited resources.

### TRL

PEFT can also be applied to training LLMs with RLHF components such as the ranker and policy. Get started by reading:

* [Fine-tune a Mistral-7b model with Direct Preference Optimization](https://towardsdatascience.com/fine-tune-a-mistral-7b-model-with-direct-preference-optimization-708042745aac) with PEFT and the [TRL](https://huggingface.co/docs/trl/index) library to learn more about the Direct Preference Optimization (DPO) method and how to apply it to a LLM.
* [Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU](https://huggingface.co/blog/trl-peft) with PEFT and the [TRL](https://huggingface.co/docs/trl/index) library, and then try out the [gpt2-sentiment_peft.ipynb](https://github.com/huggingface/trl/blob/main/examples/notebooks/gpt2-sentiment.ipynb) notebook to optimize GPT2 to generate positive movie reviews.
* [StackLLaMA: A hands-on guide to train LLaMA with RLHF](https://huggingface.co/blog/stackllama) with PEFT, and then try out the [stack_llama/scripts](https://github.com/huggingface/trl/tree/main/examples/research_projects/stack_llama/scripts) for supervised finetuning, reward modeling, and RL finetuning.

## Model support

Use this [Space](https://stevhliu-peft-methods.hf.space) or check out the [docs](https://huggingface.co/docs/peft/main/en/index) to find which models officially support a PEFT method out of the box. Even if you don't see a model listed below, you can manually configure the model config to enable PEFT for a model. Read the [New transformers architecture](https://huggingface.co/docs/peft/main/en/developer_guides/custom_models#new-transformers-architectures) guide to learn how.

## Contribute

If you would like to contribute to PEFT, please check out our [contribution guide](https://huggingface.co/docs/peft/developer_guides/contributing).

## Citing 🤗 PEFT

To use 🤗 PEFT in your publication, please cite it by using the following BibTeX entry.

```bibtex
@Misc{peft,
  title =        {PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods},
  author =       {Sourab Mangrulkar and Sylvain Gugger and Lysandre Debut and Younes Belkada and Sayak Paul and Benjamin Bossan},
  howpublished = {\url{https://github.com/huggingface/peft}},
  year =         {2022}
}
```