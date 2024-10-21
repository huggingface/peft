# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from peft import LoraConfig, EvaConfig, get_peft_model, initialize_lora_eva_weights
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader


# config
model_name = "meta-llama/Llama-3.1-8B"
dataset_name = "Rowan/hellaswag"
max_seq_len = 512
rank = 16
alpha = 1
rho = 2.0
use_label_mask = False
whiten = False
target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
svd_batch_size = 4 # can be different from the batch size used in finetuning
svd_device = "cuda"

# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# load dataset
dataset = load_dataset(dataset_name)
dataset = dataset.map(
    lambda x: tokenizer(x['ctx'], padding='max_length', truncation=True, max_length=max_seq_len),
    batched=True,
    remove_columns=dataset["train"].column_names,
)
dataset.set_format(type='torch')

# create dataloader for SVD
dataloader = DataLoader(
    dataset['train'],
    batch_size=svd_batch_size,
    collate_fn=lambda examples: {k: torch.stack([v[k] for v in examples], dim=0) for k in examples[0].keys()}
)

# setup peft config
eva_config = EvaConfig(
    rho = rho,
    use_label_mask = use_label_mask,
    whiten = whiten
)
peft_config = LoraConfig(
    r = rank,
    lora_alpha = alpha,
    target_modules = target_modules,
    init_lora_weights = "eva",
    eva_config = eva_config
)

# to optimize memory usage during eva initialization, set low_cpu_mem_usage=True
peft_model = get_peft_model(model, peft_config, low_cpu_mem_usage=True)

initialize_lora_eva_weights(peft_model, peft_config, dataloader, device=svd_device)

# from this point on, you can use the model as you would use a normal LoRA model
