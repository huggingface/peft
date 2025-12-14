import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from peft import CartridgeConfig, get_peft_model


class DistillJsonlDataset(Dataset):
    def __init__(self, path: str | Path):
        self.rows = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        return {
            "teacher_input_ids": r["teacher_input_ids"],
            "student_input_ids": r["student_input_ids"],
            "ctx_len": r["ctx_len"],
        }


class DistillationCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        teacher_ids = [{"input_ids": f["teacher_input_ids"]} for f in features]
        student_ids = [{"input_ids": f["student_input_ids"]} for f in features]
        teacher_batch = self.tokenizer.pad(teacher_ids, return_tensors="pt")
        student_batch = self.tokenizer.pad(student_ids, return_tensors="pt")
        ctx_len = torch.tensor([int(f["ctx_len"]) for f in features], dtype=torch.long)
        return {
            "teacher_input_ids": teacher_batch["input_ids"],
            "teacher_attention_mask": teacher_batch["attention_mask"],
            "student_input_ids": student_batch["input_ids"],
            "student_attention_mask": student_batch["attention_mask"],
            "ctx_len": ctx_len,
        }


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model, top_k: int = 20, teacher_temperature: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model.eval()
        self.top_k = int(top_k)
        self.teacher_temperature = float(teacher_temperature)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        teacher_input_ids = inputs["teacher_input_ids"].to(model.device)
        teacher_attention_mask = inputs["teacher_attention_mask"].to(model.device)
        student_input_ids = inputs["student_input_ids"].to(model.device)
        student_attention_mask = inputs["student_attention_mask"].to(model.device)
        ctx_len = inputs["ctx_len"].to(model.device)

        with torch.no_grad():
            teacher_out = self.teacher_model(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                use_cache=False,
            )
            teacher_logits = teacher_out.logits / max(self.teacher_temperature, 1e-5)

        student_out = model(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            use_cache=False,
        )
        student_logits = student_out.logits

        batch_losses = []
        for b in range(student_input_ids.shape[0]):
            x_len = int(student_attention_mask[b].sum().item())
            if x_len <= 1:
                continue

            student_slice = student_logits[b, : x_len - 1, :]
            t_start = int(ctx_len[b].item())
            teacher_slice = teacher_logits[b, t_start : t_start + (x_len - 1), :]
            if teacher_slice.shape[0] != student_slice.shape[0]:
                raise ValueError("Mismatched teacher/student sequence alignment for distillation loss.")

            topk_vals, topk_ids = torch.topk(teacher_slice, k=min(self.top_k, teacher_slice.shape[-1]), dim=-1)
            teacher_logprobs = F.log_softmax(teacher_slice, dim=-1).gather(-1, topk_ids)
            student_logprobs = F.log_softmax(student_slice, dim=-1).gather(-1, topk_ids)
            loss_by_pos = -(teacher_logprobs.exp() * student_logprobs).sum(dim=-1)
            batch_losses.append(loss_by_pos.mean())

        loss = torch.stack(batch_losses).mean() if batch_losses else student_logits.new_zeros(())
        return (loss, student_out) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, required=True)
    parser.add_argument("--student_model", type=str, required=True)
    parser.add_argument("--distill_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_virtual_tokens", type=int, default=256)
    parser.add_argument("--num_frozen_tokens", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_steps", type=int, default=1000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model)
    student_base = AutoModelForCausalLM.from_pretrained(args.student_model)

    peft_cfg = CartridgeConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=args.num_virtual_tokens,
        num_frozen_tokens=args.num_frozen_tokens,
    )
    student = get_peft_model(student_base, peft_cfg)

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
