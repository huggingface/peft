# 🤗 PET
Parameter-Efficient Tuning. Intergrated with 🤗 Accelerate to scale seamlessly to large models using PyTorch FSDP. 

Supported methods:

1. LoRA
2. Prefix Tuning
3. P-Tuning
4. Prompt Tuning 

## Getting started

```python
from transformers import AutoModelForSeq2SeqLM
from pet import get_pet_config, get_pet_model
model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"

config = {
    "pet_type":"LORA",
    "task_type":"SEQ_2_SEQ_LM",
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1
}
pet_config = get_pet_config(config)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_pet_model(model, pet_config)
model.print_trainable_parameters()
# output: trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282
```

## PET + 🤗 Accelerate

PET models work with 🤗 Accelerate out of the box. 
For scaling to large models, you can leverage 🤗 Accelerate's PyTorch FSDP integration as shown below.
PyTorch FSDP shards parameters, gradients and optimizer states across data parallel workers which enables
large language models to fit on available hardware. 
It also supports CPU offloading to further enable distributed training at scale. 
The support for DeepSpeed ZeRO Stage-3 is currently in backlog.

```python
from pet.utils.other import fsdp_auto_wrap_policy

...

if accelerator.state.fsdp_plugin is not None:
    accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)

model = accelerator.prepare(model)
```


## Models support matrix

### Sequence Classification
|   Model         | LoRA | Prefix Tuning  | P-Tuning | Prompt Tuning  | 
| --------- | ---- | ---- | ---- | ----  |
| BERT           | ✅  | ✅  | ✅  | ✅  |  
| RoBERTa        | ✅  | ✅  | ✅  | ✅  |
| GPT-2          | ✅  | ✅  | ✅  | ✅  | 
| Bloom          | ✅  | ✅  | ✅  | ✅  |   
| OPT            | ✅  | ✅  | ✅  | ✅  |
| GPT-Neo        | ✅  | ✅  | ✅  | ✅  |
| GPT-J          | ✅  | ✅  | ✅  | ✅  |
| Deberta        | ✅  |     |     |     | 
| Deberta-v2     | ✅  |     |     |     |

### Causal Language Modeling
|   Model         | LoRA | Prefix Tuning  | P-Tuning | Prompt Tuning  |
| --------- | ---- | ---- | ---- | ----  |
| GPT-2          | ✅  | ✅  | ✅  | ✅  |
| Bloom          | ✅  | ✅  | ✅  | ✅  |
| OPT            | ✅  | ✅  | ✅  | ✅  |
| GPT-Neo        | ✅  | ✅  | ✅  | ✅  |
| GPT-J          | ✅  | ✅  | ✅  | ✅  |

### Conditional Generation
|   Model         | LoRA | Prefix Tuning  | P-Tuning | Prompt Tuning  | 
| --------- | ---- | ---- | ---- | ---- |
| T5        | ✅   | ✅   | ✅   | ✅   |
| BART      | ✅   | ✅   | ✅   | ✅   |


## Caveats:
1. Doesn't work currently with DeeSpeed ZeRO Stage-3. Extending support with DeeSpeed ZeRO Stage-3 is in backlog.


