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

import os

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from utils import DataCollator, TokenizerMetaMath

from peft import EvaConfig, LoraConfig, get_eva_state_dict, get_peft_model, initialize_lora_eva_weights


# run this script e.g. with: torchrun --nproc_per_node=4 eva_finetuning_multi_gpu.py

# config
model_name = "meta-llama/Llama-2-7b-hf"
max_seq_len = 512
rank = 16
alpha = 1
rho = 2.0
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
svd_batch_size = 4  # can be different from the batch size used in finetuning
batch_size = 4
learning_rate = 5e-4
gradient_accumulation_steps = 8
num_epochs = 1
output_dir = "outputs"
bf16 = True


# Initialize distributed environment
if torch.cuda.is_available():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
elif torch.xpu.is_available():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    torch.xpu.set_device(local_rank)
    dist.init_process_group("xccl")
    world_size = dist.get_world_size()
else:
    local_rank = -1
    world_size = 1


# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# load dataset
dataset = load_dataset("meta-math/MetaMathQA")
dataset = dataset.map(
    TokenizerMetaMath(model_name),
    batched=True,
    remove_columns=dataset["train"].column_names,
)
dataset.set_format(type="torch")

# data collator
data_collator = DataCollator(tokenizer.eos_token_id, max_length=max_seq_len)

# Create sampler for distributed training
sampler = DistributedSampler(dataset["train"], num_replicas=world_size, rank=local_rank)

# dataloader
dataloader = DataLoader(
    dataset["train"],
    batch_size=svd_batch_size,
    collate_fn=data_collator,
    sampler=sampler,
    shuffle=False,
)

sampler.set_epoch(0)

# Wrap model in DDP
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# setup peft config
eva_config = EvaConfig(rho=rho)
peft_config = LoraConfig(
    r=rank, lora_alpha=alpha, target_modules=target_modules, init_lora_weights="eva", eva_config=eva_config
)

# EVA initialization
eva_state_dict = get_eva_state_dict(model, dataloader, peft_config)
eva_state_dict = {".".join(["base_model.model"] + k.split(".")[1:]): v for k, v in eva_state_dict.items()}

# cleanup ddp
model = model.module

# initialize peft model
peft_model = get_peft_model(model, peft_config, low_cpu_mem_usage=True)
initialize_lora_eva_weights(peft_model, eva_state_dict=eva_state_dict)

# setup training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=batch_size,
    learning_rate=learning_rate,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_epochs,
    output_dir=output_dir,
    remove_unused_columns=False,
    bf16=bf16,
)

# continue with standard finetuning
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator,
)
trainer.train()
