<h1 align="center"> <p>🤗 PET</p></h1>
<h3 align="center">
    <p>State-of-the-art Parameter-Efficient Tuning (PET) methods</p>
</h3>

Parameter-Efficient Tuning (PET) methods enable efficient adaptation of pre-trained language models (PLMs) to various downstream applications without fine-tuning all the model's parameters. Fine-tuning large-scale PLMs is often prohibitively costly. In this regard, PET methods only fine-tune a small number of (extra) model parameters, thereby greatly decreasing the computational and storage costs. Recent State-of-the-Art PET techniques achieve performance comparable to that of full fine-tuning. 

Seamlessly integrated with 🤗 Accelerate for large scale models leveraging PyTorch FSDP. 

Supported methods:

1. LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685.pdf)
2. Prefix Tuning: [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/pdf/2110.07602.pdf)
3. P-Tuning: [GPT Understands, Too](https://arxiv.org/pdf/2103.10385.pdf)
4. Prompt Tuning: [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691.pdf) 

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


## Citing 🤗 PET

If you use 🤗 PET in your publication, please cite it by using the following BibTeX entry.

```bibtex
@Misc{pet,
  title =        {PET: State-of-the-art Parameter-Efficient Tuning (PET) methods},
  author =       {Sourab Mangrulkar},
  howpublished = {\url{https://github.com/huggingface/pet}},
  year =         {2022}
}
```
