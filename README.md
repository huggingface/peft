# 🤗 PET
Parameter-Efficient Tuning methods enable . Intergrated with 🤗 Accelerate to scale seamlessly to large models using PyTorch FSDP. 

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

```python
from pet.utils.other import fsdp_auto_wrap_policy

...

if os.environ.get("ACCELERATE_USE_FSDP", None) is not None:
    accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)

model = accelerator.prepare(model)
```

Example of parameter efficient tuning with `mt0-xxl` base model using 🤗 Accelerate is provided in `~examples/pet_lora_seq2seq_accelerate_fsdp.py`. 
1. First run `accelerate config --config_file fsdp_config.yaml` and answer the questionaire. 
Below are the contents of the config file.
```
command_file: null
commands: null
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: FSDP
downcast_bf16: 'no'
dynamo_backend: 'NO'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_offload_params: true
  fsdp_sharding_strategy: 1
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: T5Block
gpu_ids: null
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
megatron_lm_config: {}
mixed_precision: 'no'
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_name: null
tpu_zone: null
use_cpu: false
```
2. run the below command to launch example script
```
accelerate launch --config_file fsdp_config.yaml examples/pet_lora_seq2seq_accelerate_fsdp.py
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
2. When using `P_TUNING` or `PROMPT_TUNING` with `SEQ_2_SEQ` task, remember to remove the `num_virtual_token` virtual prompt predictions from the left side of the model outputs during evaluations. 



