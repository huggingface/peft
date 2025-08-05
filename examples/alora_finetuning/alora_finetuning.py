import copy
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForCompletionOnlyLM,
    DynamicCache,
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

    device = torch.device(device)
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)

    invocation_tokens = tokenizer.encode(invocation_string, add_special_tokens=False)

    if quantize:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            token=hf_token,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                ),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, token=hf_token)

    lora_config = LoraConfig(
        alora_invocation_tokens=invocation_tokens,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=(
            lora_target_modules.split(",")
            if lora_target_modules
            else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        ),
        lora_dropout=lora_dropout,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(data_path)

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['input'])):
            chat = [{
                "role": "user",
                "content": example['input'][i]
            },
            {
                "role": "assistant",
                "content": example['output'][i]
            }]
            text = tokenizer.apply_chat_template(chat, tokenize=False,add_generation_prompt=False)
            output_texts.append(text)
        return output_texts

    data_collator = DataCollatorForCompletionOnlyLM(invocation_string, tokenizer=tokenizer)

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

    torch.cuda.empty_cache()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        formatting_func=formatting_prompts_func,
        data_collator=data_collator,
    )

    trainer.train()

    if push_to_hub:
        trainer.push_to_hub(commit_message="Fine-tuned model")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def model_inference(model_path: str, adapter_path: str, prompt: str=None, data_path: str=None, reuse_cache: bool = True):
    '''
    Simple inference with the tuned aLoRA adapter. Optionally (reuse_cache = True) demonstrates
    that the aLoRA adapter can (but does not need to) use KV cache created by the base model,
    perhaps during a prior generation turn.

    Purely for demonstration purposes. See the [paper](https://huggingface.co/papers/2504.12397)
    for realistic multiturn cache reuse examples.
    '''
    if prompt is None:
        # Use first row of test data
        dataset = load_dataset(data_path)
        prompt = dataset["test"][0]["input"]

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    alora_model = PeftModel.from_pretrained(base_model, adapter_path,adapter_name="adapter")

    chat = [{
        "role": "user",
        "content": prompt
    }]
    text = tokenizer.apply_chat_template(chat, tokenize=False,add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(base_model.device)

    if reuse_cache:
        # Input through the end of the last turn
        text_input = tokenizer.apply_chat_template(chat, tokenize=False,add_generation_prompt=False)
        alora_model.set_adapter(None)
        kv_cache = DynamicCache()
        inputs_prefill = tokenizer(text_input,return_tensors="pt").to(base_model.device)
        # prefill input with base model
        with torch.no_grad():
            kv_cache = alora_model(**inputs_prefill, past_key_values=kv_cache).past_key_values

        # Generate answer with adapter
        alora_model.set_adapter("adapter")
        output_dict = alora_model.generate(**inputs,past_key_values=copy.deepcopy(kv_cache),return_dict_in_generate=True)
        alora_outputs = output_dict.sequences

        # Generate answer with base model for comparison
        alora_model.set_adapter(None)
        output_dict = alora_model.generate(**inputs,past_key_values=copy.deepcopy(kv_cache),return_dict_in_generate=True)
        base_outputs = output_dict.sequences
    else:
        # Simpler inference calls (output equivalent to the above)
        # Generate answer with adapter
        alora_model.set_adapter("adapter")
        output_dict = alora_model.generate(**inputs,return_dict_in_generate=True)
        alora_outputs = output_dict.sequences

        # Generate answer with base model for comparison
        alora_model.set_adapter(None)
        output_dict = alora_model.generate(**inputs,return_dict_in_generate=True)
        base_outputs = output_dict.sequences
    # Print results
    print(f"Prompt: {text}")
    print(f"Base model response: {tokenizer.decode(base_outputs[0]).rsplit(text,1)[1]}")
    print(f"Trained adapter response: {tokenizer.decode(alora_outputs[0]).rsplit(text,1)[1]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Mistral with Activated LoRA")
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Base model path or name")
    parser.add_argument(
        "--data_path", type=str, default="timdettmers/openassistant-guanaco", help="Dataset path or name"
    )
    parser.add_argument(
        "--output_dir", type=str, default="path/to/output", help="Output directory for the fine-tuned model"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--cutoff_len", type=int, default=512, help="Cutoff length for tokenization")
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
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training")
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
    model_inference(model_path = args.base_model, adapter_path = args.output_dir, data_path=args.data_path, reuse_cache = True)
