import argparse

import torch
from train_distill import DistillationCollator, DistillationTrainer, DistillJsonlDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from peft import CartridgeConfig, get_peft_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--student_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--distill_jsonl", type=str, default="distill.jsonl")
    parser.add_argument("--output_dir", type=str, default="cartridge_adapter")
    parser.add_argument("--num_virtual_tokens", type=int, default=256)
    parser.add_argument("--num_frozen_tokens", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ValueError("Requested device 'mps' but MPS is not available.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested device 'cuda' but CUDA is not available.")

    device = torch.device(args.device)
    torch_dtype = torch.float16 if args.device in {"cuda", "mps"} else None

    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, torch_dtype=torch_dtype).to(device)
    student_base = AutoModelForCausalLM.from_pretrained(args.student_model, torch_dtype=torch_dtype)
    student = get_peft_model(
        student_base,
        CartridgeConfig(
            task_type="CAUSAL_LM",
            num_virtual_tokens=args.num_virtual_tokens,
            num_frozen_tokens=args.num_frozen_tokens,
        ),
    )

    ds = DistillJsonlDataset(args.distill_jsonl)
    collator = DistillationCollator(tokenizer)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=100,
        report_to=[],
        remove_unused_columns=False,
        use_cpu=args.device == "cpu",
        dataloader_pin_memory=False,
    )

    trainer = DistillationTrainer(
        model=student,
        teacher_model=teacher,
        top_k=args.top_k,
        args=train_args,
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()
    student.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
