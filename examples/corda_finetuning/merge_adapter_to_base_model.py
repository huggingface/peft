import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftConfig, PeftModel


parser = argparse.ArgumentParser(description="Merge Adapter to Base Model")
parser.add_argument("--base_model", type=str)
parser.add_argument("--adapter", type=str)
parser.add_argument("--output_path", type=str)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="auto")
# model.resize_token_embeddings(32001)

tokenizer = AutoTokenizer.from_pretrained(args.base_model, device_map="auto")
lora_config = PeftConfig.from_pretrained(args.adapter)

lora_config.init_lora_weights = True
model = PeftModel.from_pretrained(model, args.adapter, config=lora_config)
model = model.merge_and_unload()
model.save_pretrained(args.output_path)
tokenizer.save_pretrained(args.output_path)
