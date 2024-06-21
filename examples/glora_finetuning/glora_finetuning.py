import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from peft.tuners.glora import GLoraConfig


#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE
dataset_id = "roneneldan/TinyStories"
model_id = "Maykeye/TinyLLama-v0"


def training_function():
    # # Dataset

    dataset_train = load_dataset(dataset_id, data_files={"train": "TinyStories-train.txt"}, split="train")
    dataset_validation = load_dataset(dataset_id, data_files={"test": "TinyStories-valid.txt"}, split="test")

    torch_dtype = torch.bfloat16
    quant_storage_dtype = torch.bfloat16

    # Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=quant_storage_dtype,
        use_cache=True,  # this is needed for gradient checkpointing
    )

    model.gradient_checkpointing_enable()

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ################
    # PEFT
    ################

    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
    peft_config = GLoraConfig(
        r=8,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    sftp_config = SFTConfig(
        bf16=True,
        tf32=True,
        logging_steps=10,
        learning_rate=0.0002,
        warmup_ratio=0.3,
        max_grad_norm=0.3,
        save_strategy="epoch",
        eval_strategy="epoch",
        output_dir="./output",
        report_to="tensorboard",
        num_train_epochs=3,
        per_device_eval_batch_size=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        packing=True,
        max_seq_length=4000,
        dataset_text_field="text",
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        },
    )


    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_validation,
        args=sftp_config,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )

    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    ##########################
    # Train model
    ##########################
    trainer.train()

    ##########################
    # SAVE MODEL FOR SAGEMAKER
    ##########################
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


if __name__ == "__main__":
    # launch training
    training_function()
