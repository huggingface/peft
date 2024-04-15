import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft.utils.pissa_utils import pissa_post_training_saving
from trl import SFTTrainer
from datasets import load_dataset

import argparse

parser = argparse.ArgumentParser(description="Fine-tuning PiSSA with 4bit residual model")
parser.add_argument(
    "--base_model_name_or_path",
    type=str,
    default="meta-llama/Llama-2-7b-hf",
    help="The name or path of the fp32/16 base model.",
)
parser.add_argument(
    "--residual_model_name_or_path",
    type=str,
    default="fxmeng/pissa-llama-2-7b-r16-alpha-16",
    help="The name or path of the fp32/16 residual model.",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="pissa-llama-2-7b-r16-alpha-16",
    help="Including residual model, initial pissa, finetuned pissa",
)
parser.add_argument(
    "--init_lora_weights",
    type=str,
    default="pissa",
    help=["pissa", "pissa_niter_4"],
)
parser.add_argument(
    "--r",
    type=int,
    default=16,
    help="Rank of PiSSA",
)
parser.add_argument(
    "--lora_alpha",
    type=int,
    default=16,
    help="Alpha of PiSSA",
)
parser.add_argument(
    "--lora_dropout",
    type=int,
    default=0,
    help="Dropout ratio of PiSSA",
)
args = parser.parse_args()

# Download the llama-2-7b model from huggingface
model = AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Configure PiSSA with Fast SVD:
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    init_lora_weights="pissa_niter_4",  # Fast initialization with "_niter_xx"
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Finetuning PiSSA
dataset = load_dataset("imdb", split="train[:1%]")
trainer = SFTTrainer(
    model=model, train_dataset=dataset, dataset_text_field="text", max_seq_length=512, tokenizer=tokenizer
)

############################## It's essential to save initial PiSSA parameters for conversion to LoRA. ##############################
model.save_pretrained("pissa-llama-2-7b-alpaca-init")
trainer.train()
############################## Upon completion, save final PiSSA parameters ##############################
model.save_pretrained("pissa-llama-2-7b-alpaca-finetuned")

############################## The different of the PiSSA parameters before and after the training corresponding to delta W in LoRA. ##############################
pissa_post_training_saving(
    "pissa-llama-2-7b-alpaca-init", "pissa-llama-2-7b-alpaca-finetuned", "lora-r32-llama-2-7b-alpaca", device="cpu"
)
