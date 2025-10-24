import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


def train_model(
    base_model: str,
    data_path: str,
    output_dir: str,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    cutoff_len: int,
    val_set_size: int,
    invocation_string: str,
    quantize: bool,
    eval_step: int,
    save_step: int,
    device: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: str,
    hub_model_id: str,
    push_to_hub: bool,
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    hf_token = os.getenv("HF_TOKEN")

    if device == "auto":
        device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
    tokenizer.pad_token = tokenizer.unk_token
    invocation_tokens = tokenizer.encode(invocation_string, add_special_tokens=False)

    if quantize:
        if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) or torch.xpu.is_available():
            bnb_4bit_compute_dtype = torch.bfloat16
        else:
            bnb_4bit_compute_dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            token=hf_token,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, token=hf_token)

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        alora_invocation_tokens=invocation_tokens,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=(lora_target_modules.split(",") if lora_target_modules else ["q_proj", "k_proj", "v_proj"]),
        lora_dropout=lora_dropout,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(data_path)

    def tokenize_function(examples):
        formatted_texts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ],
                tokenize=False,  # get plain text first
                add_generation_prompt=False,
            )
            for user_msg, assistant_msg in zip(examples["input"], examples["output"])
        ]

        # 2) Tokenize those texts
        model_inputs = tokenizer(
            formatted_texts,
            padding="max_length",
            truncation=True,
            max_length=cutoff_len,
        )

        labels = []
        for ids in model_inputs["input_ids"]:
            labels.append([(token_id if token_id != tokenizer.pad_token_id else -100) for token_id in ids])
        model_inputs["labels"] = labels

        return model_inputs

    # Tokenize the dataset and prepare for training
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

    # Data collator to dynamically pad the batched examples
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=eval_step,
        save_steps=save_step,
        save_total_limit=2,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        gradient_accumulation_steps=16,
        fp16=True,
        learning_rate=learning_rate,
        hub_token=hf_token,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.xpu.is_available():
        torch.xpu.empty_cache()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )

    trainer.train()

    if push_to_hub:
        trainer.push_to_hub(commit_message="Fine-tuned model")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def model_inference(model_path: str, adapter_path: str, prompt: str = None, data_path: str = None):
    """
    Simple inference with the tuned aLoRA adapter. Optionally (reuse_cache = True) demonstrates
    that the aLoRA adapter can (but does not need to) use KV cache created by the base model,
    perhaps during a prior generation turn.

    Purely for demonstration purposes. See the [paper](https://huggingface.co/papers/2504.12397)
    for realistic multiturn cache reuse examples.
    """
    if prompt is None:
        # Use first row of test data
        dataset = load_dataset(data_path)
        prompt = dataset["test"][0]["input"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    alora_model = PeftModel.from_pretrained(base_model, adapter_path)
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(base_model.device)

    # Generate answer with adapter
    output_dict = alora_model.generate(**inputs, return_dict_in_generate=True, max_new_tokens=20)
    alora_outputs = output_dict.sequences

    # Print results
    print(f"Prompt: {text}")
    response = tokenizer.decode(alora_outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    print(f"Trained adapter response: {response}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Mistral with Activated LoRA")
    parser.add_argument(
        "--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Base model path or name"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="Lots-of-LoRAs/task1660_super_glue_question_generation",
        help="Dataset path or name",
    )
    parser.add_argument(
        "--output_dir", type=str, default="path/to/output", help="Output directory for the fine-tuned model"
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--cutoff_len", type=int, default=2048, help="Cutoff length for tokenization")
    parser.add_argument("--val_set_size", type=int, default=500, help="Validation set size")
    parser.add_argument(
        "--invocation_string",
        type=str,
        default="[/INST]",
        help="String that activates the aLoRA adapter. Model dependent.",
    )
    parser.add_argument("--quantize", action="store_true", help="Use quantization")
    parser.add_argument("--eval_step", type=int, default=10, help="Evaluation step interval")
    parser.add_argument("--save_step", type=int, default=100, help="Save step interval")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for training")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument(
        "--lora_target_modules", type=str, default=None, help="Comma-separated list of target modules for LoRA"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="path/to/repo",
        help="Repository name to push the model on the Hugging Face Hub",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to Hugging Face Hub")
    args = parser.parse_args()
    train_model(
        base_model=args.base_model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        cutoff_len=args.cutoff_len,
        val_set_size=args.val_set_size,
        invocation_string=args.invocation_string,
        quantize=args.quantize,
        eval_step=args.eval_step,
        save_step=args.save_step,
        device=args.device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        hub_model_id=args.hub_model_id,
        push_to_hub=args.push_to_hub,
    )
    print("Model trained. Running test inference.")
    model_inference(model_path=args.base_model, adapter_path=args.output_dir, data_path=args.data_path)
