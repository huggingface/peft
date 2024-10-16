import torch
from peft import LoraConfig, EvaConfig, get_peft_model
from peft.tuners.lora.eva import initialize_lora_eva_weights
from peft.tuners.lora import LoraLayer
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

# get eva model
eva_config = EvaConfig(
    rho = rho,
    use_label_mask = use_label_mask,
    whiten = whiten,
    device = svd_device,
)
peft_config = LoraConfig(
    r = rank,
    lora_alpha = alpha,
    target_modules = target_modules,
    init_lora_weights = "eva",
    eva_config = eva_config
)
peft_model = get_peft_model(model, peft_config, low_cpu_mem_usage=True)

initialize_lora_eva_weights(peft_model, peft_config, dataloader, device="cuda")