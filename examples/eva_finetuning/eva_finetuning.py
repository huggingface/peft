import torch
from peft import LoraConfig, EvaConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader


model_name = "meta-llama/Llama-3.1-8B"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("Rowan/hellaswag")
dataset = dataset.map(
    lambda x: tokenizer(x['ctx'], padding='max_length', truncation=True, max_length=512),
    batched=True,
    remove_columns=dataset["train"].column_names,
)
dataset.set_format(type='torch')
def collate_fn(examples):
    return {k: torch.stack([v[k] for v in examples], dim=0) for k in examples[0].keys()}

dataloader = DataLoader(dataset['train'], batch_size=4, collate_fn=collate_fn)

eva_config = EvaConfig(
    dataloader = dataloader,
    rho=2,
)
peft_config = LoraConfig(
    r = 16,
    lora_alpha = 8,
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.05,
    bias = 'none',
    init_lora_weights="eva",
    eva_config = eva_config,
)
peft_model = get_peft_model(model, peft_config)