import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Download and load the llama-2-7b residual model in 4-bit format:
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base_model = AutoModelForCausalLM.from_pretrained("fxmeng/pissa-llama-2-7b-r16-alpha-16", quantization_config=quantization_config, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained('fxmeng/pissa-llama-2-7b-r16-alpha-16')
tokenizer.pad_token_id = tokenizer.eos_token_id

# Wrapping the residual model with PiSSA:
model = PeftModel.from_pretrained(base_model, "fxmeng/pissa-llama-2-7b-r16-alpha-16", subfolder="pissa_init", is_trainable=True)
model.print_trainable_parameters()
model = prepare_model_for_kbit_training(model)

from trl import SFTTrainer
from datasets import load_dataset
dataset = load_dataset("fxmeng/alpaca_in_mixtral_format", split="train[:1%]")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer
)

model.save_pretrained('pissa-llama-2-7b-r16-alpha-16/pissa_init')
trainer.train()
model.save_pretrained('pissa-llama-2-7b-r16-alpha-16/pissa_ft')

from convert_pissa_to_lora import pissa_to_lora
pissa_to_lora(init_path="pissa-llama-2-7b-r16-alpha-16/pissa_init", finetuned_path="pissa-llama-2-7b-r16-alpha-16/pissa_ft", output_path="pissa-llama-2-7b-r16-alpha-16/pissa_lora")