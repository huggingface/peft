# EVA: Efficient Vector Adaptation
## Introduction ([Paper](https://arxiv.org/abs/2404.02948), [code](https://github.com/GraphPKU/EVA))
Explained Variance Adaptation (EVA) is a novel intialization method for LoRA style adapters which initializes adapter weights in a data driven manner and adaptively allocates ranks according to the variance they explain. EVA improves average performance on a multitude of tasks across various domains, such as Language generation and understanding, Image classification, and Decision Making.

## Quick Start
```python
import torch
from peft import LoraConfig, EvaConfig, get_peft_model, initialize_lora_eva_weights
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer.pad_token = tokenizer.eos_token

# load dataset
dataset = load_dataset("Rowan/hellaswag")
dataset = dataset.map(
    lambda x: tokenizer(x['ctx'], padding='max_length', truncation=True, max_length=max_seq_len),
    batched=True,
    remove_columns=dataset["train"].column_names,
)
dataset.set_format(type='torch')

# create dataloader for SVD
dataloader = DataLoader(
    dataset['train'],
    batch_size=4,
    collate_fn=lambda examples: {k: torch.stack([v[k] for v in examples], dim=0) for k in examples[0].keys()}
)

# setup peft config
peft_config = LoraConfig(
    r = rank,
    lora_alpha = alpha,
    target_modules = target_modules,
    init_lora_weights = "eva",
    eva_config = EvaConfig(rho = 1.0)
)

# to optimize memory usage during eva initialization, set low_cpu_mem_usage=True
peft_model = get_peft_model(model, peft_config, low_cpu_mem_usage=True)

initialize_lora_eva_weights(peft_model, peft_config, dataloader, device=svd_device)
```
When `initialize_lora_eva_weights` is called, it will load the components calculated by EVA into the model. After this continue with standard LoRA finetuning.

## Getting eva_state_dict without loading the adapter weights
If you want to get the eva_state_dict without loading the adapter weights, you can do the following:
```python
import torch
from peft import LoraConfig, EvaConfig, get_peft_model, get_eva_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer.pad_token = tokenizer.eos_token

# load dataset
dataset = load_dataset("Rowan/hellaswag")
dataset = dataset.map(
    lambda x: tokenizer(x['ctx'], padding='max_length', truncation=True, max_length=max_seq_len),
    batched=True,
    remove_columns=dataset["train"].column_names,
)
dataset.set_format(type='torch')

# create dataloader for SVD
dataloader = DataLoader(
    dataset['train'],
    batch_size=4,
    collate_fn=lambda examples: {k: torch.stack([v[k] for v in examples], dim=0) for k in examples[0].keys()}
)

# setup peft config
peft_config = LoraConfig(
    r = rank,
    lora_alpha = alpha,
    target_modules = target_modules,
    init_lora_weights = "eva",
    eva_config = EvaConfig(rho = 1.0)
)

model = model.cuda()
eva_state_dict = get_eva_state_dict(model, peft_config, dataloader)
```

## Citation
In case you find our work useful, please consider citing it.

@article{paischer2024eva,
    title={One Initialization to Rule them All: Fine-tuning via Explained Variance Adaptation}, 
    author={Fabian Paischer, Lukas Hauzenberger, Thomas Schmied, Benedikt Alkin, Marc Peter Deisenroth, Sepp Hochreiter},
    journal={arXiv preprint arXiv:2410.07170},
    year={2024}
}