# Copyright 2025-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from peft import CartridgeConfig, get_peft_model
from peft.tuners.cartridge.utils import initialize_kv_prefix_from_text


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
    def __init__(self, *args, top_k: int = 20, teacher_temperature: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k = int(top_k)
        self.teacher_temperature = float(teacher_temperature)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        teacher_input_ids = inputs["teacher_input_ids"].to(model.device)
        teacher_attention_mask = inputs["teacher_attention_mask"].to(model.device)
        student_input_ids = inputs["student_input_ids"].to(model.device)
        student_attention_mask = inputs["student_attention_mask"].to(model.device)
        ctx_len = inputs["ctx_len"].to(model.device)

        with torch.no_grad():
            with model.disable_adapter():
                teacher_out = model(
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

        # Vectorized distillation loss (avoids Python `.item()` in per-example indexing).
        # Align teacher logits to student positions via the per-example `ctx_len` offset.
        student_logits = student_logits[:, :-1, :]  # [B, Ls-1, V]
        seq_len = student_logits.shape[1]
        pos = torch.arange(seq_len, device=student_logits.device)[None, :]  # [1, Ls-1]

        student_len = student_attention_mask.sum(dim=1).to(torch.long)  # [B]
        valid = pos < (student_len - 1).clamp(min=0)[:, None]  # [B, Ls-1]

        teacher_pos = ctx_len[:, None] + pos  # [B, Ls-1]
        in_bounds = teacher_pos < teacher_logits.shape[1]
        valid = valid & in_bounds

        teacher_pos = teacher_pos.clamp(min=0, max=teacher_logits.shape[1] - 1)
        teacher_slice = teacher_logits.gather(
            dim=1, index=teacher_pos[:, :, None].expand(-1, -1, teacher_logits.shape[-1])
        )  # [B, Ls-1, V]

        k = min(self.top_k, teacher_slice.shape[-1])
        topk_ids = torch.topk(teacher_slice, k=k, dim=-1).indices  # [B, Ls-1, K]
        teacher_logprobs = F.log_softmax(teacher_slice, dim=-1).gather(-1, topk_ids)
        student_logprobs = F.log_softmax(student_logits, dim=-1).gather(-1, topk_ids)

        loss_by_pos = -(teacher_logprobs.exp() * student_logprobs).sum(dim=-1)  # [B, Ls-1]
        loss_by_pos = loss_by_pos.masked_fill(~valid, 0.0)

        denom = valid.sum(dim=1).clamp(min=1)
        per_example = loss_by_pos.sum(dim=1) / denom
        if valid.any():
            loss = per_example[valid.any(dim=1)].mean()
        else:
            loss = student_logits.new_zeros(())
        return (loss, student_out) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model to use for both teacher and student")
    parser.add_argument("--distill_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--document", type=str, required=True, help="Path to text file for KV cache initialization")
    parser.add_argument("--num_virtual_tokens", type=int, default=256)
    parser.add_argument("--num_frozen_tokens", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "mps", "cuda", "xpu"])
    parser.add_argument(
        "--max_init_length", type=int, default=2048, help="Max tokens for text initialization (truncate long docs)"
    )
    args = parser.parse_args()

    if args.device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ValueError("Requested device 'mps' but MPS is not available.")
    if args.device == "xpu" and not torch.xpu.is_available():
        raise ValueError("Requested device 'xpu' but XPU is not available.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested device 'cuda' but CUDA is not available.")

    model_dtype = torch.float16 if args.device in {"cuda", "mps", "xpu"} else None
    device_map = args.device if args.device != "cpu" else None

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=model_dtype, device_map=device_map)
    model = get_peft_model(
        base_model,
        CartridgeConfig(
            task_type="CAUSAL_LM",
            num_virtual_tokens=args.num_virtual_tokens,
            num_frozen_tokens=args.num_frozen_tokens,
        ),
    )

    print(f"Initializing cartridge from document: {args.document}", flush=True)
    document_text = Path(args.document).read_text()
    initialize_kv_prefix_from_text(
        model,
        tokenizer,
        text=document_text,
        use_chat_template=False,
        max_length=args.max_init_length,
    )
    print(f"Cartridge initialized with {args.num_virtual_tokens} tokens from text", flush=True)

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
        model=model,
        top_k=args.top_k,
        args=train_args,
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
