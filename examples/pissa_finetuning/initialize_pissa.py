import torch
import os
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

parser = argparse.ArgumentParser(description="Employing SVD to factorize W to the product of two trainable low rank matrices A and B, plus a residual matrix W_res for error correction")
parser.add_argument(
"--base_model_name_or_path",
type=str,
default=None,
help="The name or path of the fp32/16 base model.",
)
parser.add_argument(
"--output_dir",
type=str,
default="pissa-llama-2-7b-r16-alpha-16",
help="The name or path of the fp32/16 residual model",
)
parser.add_argument(
"--init_lora_weights",
type=str,
default="pissa",
help="(`['pissa', 'pissa_niter_[number of iters]']`)",
)
parser.add_argument(
"--lora_r",
type=int,
default=16,
)
parser.add_argument(
"--lora_alpha",
type=int,
default=16,
)
parser.add_argument(
"--lora_dropout",
type=int,
default=0,
)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(
    args.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
tokenizer.pad_token_id = tokenizer.eos_token_id
lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    init_lora_weights=args.init_lora_weights,
    lora_dropout=args.lora_dropout,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(model, lora_config)
# Save PiSSA modules:
peft_model.peft_config["default"].init_lora_weights = True
peft_model.save_pretrained(os.path.join(args.output_dir, 'pissa_init'))
# Save residual model:
residual_model = peft_model.unload()
residual_model.save_pretrained(args.output_dir)
# Save the tokenizer:
args.residual_model_name_or_path = args.output_dir
tokenizer.save_pretrained(args.output_dir)
