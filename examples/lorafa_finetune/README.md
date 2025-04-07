# LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language Models Fine-tuning

## Introduction

[LoRA-FA](https://arxiv.org/abs/2308.03303) is a noval Parameter-efficient Fine-tuning method, which freezes the projection down layer (matrix A) during LoRA training process and thus lead to less GPU memory consumption by eliminating the need for storing the activations of input tensors (X). Furthermore, LoRA-FA narrows the gap between the update amount of pre-trained weights when using the low-rank fine-tuning method and the full fine-tuning method. In conclusion, LoRA-FA reduces the memory consumption and leads to superior performance compared to vanilla LoRA.

## Quick start

```python
import torch
from peft import LoraConfig, get_peft_model
from peft.optimizers import create_lorafa_optimizer
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

lora_rank = 16
lora_alpha = 32

lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_alpha,
    bias="none",
)
peft_model = get_peft_model(model, lora_config)
optimizer = create_lorafa_optimizer(
    model=peft_model,
    r=lora_rank,
    lora_alpha=lora_alpha,
    lr=7e-5,
)
# you can also use scheduler, we recommend get_cosine_schedule_with_warmup from transformers
# for better model performance
scheduler = None

trainer = transformers.Trainer(
    model=peft_model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    optimizers=(optimizer, None),
)
trainer.train()
peft_model.save_pretrained("lorafa-llama-3-8b-inst")
```

The only change in your code is to pass the LoRA-FA optimizer to the trainer (if training with trainer). Do not forget `from peft.optimizers import create_lorafa_optimizer`!

In this dir, we also provide you a simple example for fine-tuning with LoRA-FA optimizer. Run the finetuning script simply by running:

```bash
accelerate launch examples/lorafa_finetuning/lorafa_finetuning.py --base_model_name_or_path meta-llama/Meta-Llama-3-8B --dataset_name_or_path meta-math/MetaMathQA-40K --lorafa
```

This 👆🏻 by default will load the model in peft set up with LoRA config, and train the model with LoRA-FA optimizer. The `accelerate launch` will automatically configure single-GPU or multi-GPU for you. 

LoRA-FA also supports quantization. To use bitsandbytes NF4 4-bit quantization try:

```bash
accelerate launch examples/lorafa_finetuning/lorafa_finetuning.py --base_model_name_or_path meta-llama/Meta-Llama-3-8B --dataset_name_or_path meta-math/MetaMathQA-40K --lorafa --quantize
```

## Use the model from 🤗
You can load and use the model as any other 🤗 models.
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
```

### Best practice in fine-tuning Llama on Metamath using LoRA-FA

In LoRA-FA's fine-tuning, we recommend to use a larger lora rank such as 64 or 128 (max). LoRA-FA can achieve 57.3% on GSM8K, just by fine-tuning Llama-2-7b-chat-hf on meta-math/MetaMathQA-40K for 3 epochs! For the best practices you can just check [LoRA-FA examples](https://github.com/AaronZLT/lorafa).

## LoRA-FA vs. LoRA

Despite its advantages, LoRA-FA remains inherently constrained by its low-rank approximation nature and potential catastrophic forgetting. Besides, since LoRA-FA has less trainable parameter than LoRA, LoRA-FA may converge slower than LoRA and requires larger lora rank and fine-grained hyper-parameter (mainly learning rate) search. Addressing these limitations, particularly approximation accuracy and forgetting phenomena, represents a promising direction for future work.

## Citation
```
@misc{zhang2023lorafamemoryefficientlowrankadaptation,
      title={LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language Models Fine-tuning}, 
      author={Longteng Zhang and Lin Zhang and Shaohuai Shi and Xiaowen Chu and Bo Li},
      year={2023},
      eprint={2308.03303},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2308.03303}, 
}
```