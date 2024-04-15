import torch
import os
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft.utils.pissa_utils import pissa_pre_training_saving, pissa_post_training_saving
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
    "--bits",
    type=int,
    default=4,
    help="[4, 8, 16, 32]",
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

# Load or download pre-processed residual model from:
if args.residual_model_name_or_path is None:
    # If a pre-processed model is not available, manually configure and save the model:
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        init_lora_weights=args.init_lora_weights,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
    pissa_pre_training_saving(peft_model, tokenizer, save_path=args.output_path, push_to_hub=None)

if args.bits == 4:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    res_model = AutoModelForCausalLM.from_pretrained(
        args.output_path, quantization_config=quantization_config, low_cpu_mem_usage=True
    )
elif args.bits == 8:
    res_model = AutoModelForCausalLM.from_pretrained(
        args.output_path, load_in_8bit=True, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
else:
    res_model = AutoModelForCausalLM.from_pretrained(
        args.output_path, torch_dtype=torch.bfloat16, device_map="auto"
    )

tokenizer = AutoTokenizer.from_pretrained(args.output_path)
# Wrapping the residual model with PiSSA:
peft_model = PeftModel.from_pretrained(res_model, args.output_path, subfolder="pissa_init", is_trainable=True)
peft_model.print_trainable_parameters()
peft_model = prepare_model_for_kbit_training(peft_model)


# Training PiSSA with trl on imdb dataset (using a subset for fast evaluation):
dataset = load_dataset("imdb", split="train[:1%]")
trainer = SFTTrainer(
    model=peft_model, train_dataset=dataset, dataset_text_field="text", max_seq_length=512, tokenizer=tokenizer
)
############################## It's essential to save initial PiSSA parameters for conversion to LoRA. ##############################
peft_model.save_pretrained(os.path.join(args.output_path, "pissa_init"))
trainer.train()
############################## Upon completion, save final PiSSA parameters ##############################
peft_model.save_pretrained(args.output_path, "pissa_ft")


############################## The different of the PiSSA parameters before and after the training corresponding to delta W in LoRA. ##############################
pissa_post_training_saving(
    init_path=os.path.join(args.output_path, "pissa_init"),
    finetuned_path=os.path.join(args.output_path, "pissa_ft"),
    output_path=os.path.join(args.output_path, "pissa_lora"),
)
