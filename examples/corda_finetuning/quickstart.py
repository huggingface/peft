import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft.tuners.lora.config import CordaConfig, CordaInitConfig
from peft.tuners.lora.corda import preprocess_corda
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id
sampled_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:256]")
dataset = load_dataset("imdb", split="train[:256]")


def run_model():
    for batch in sampled_dataset:
        input_ids = batch["text"]
        input_ids = input_ids.to(model.device)
        with torch.no_grad():
            model(input_ids)


init_config = CordaInitConfig(
    run_model=run_model,
)
corda_config = CordaConfig(
    sample_count=256,
    corda_method="kpm",
)
lora_config = LoraConfig(
    init_lora_weights="corda",
    corda_config=corda_config,
)
preprocess_corda(model, lora_config, init_config)
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

training_args = SFTConfig(dataset_text_field="text", max_seq_length=128)
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
peft_model.save_pretrained("corda-llama-2-7b")
