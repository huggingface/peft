import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_SEED_PROMPTS = {
    "structuring": "Please outline the key sections and how the information is structured.",
    "summarization": "Please summarize the most important points concisely.",
    "question": "Ask a question about the document above that requires careful reading, then answer it.",
    "use_cases": "List a few realistic use cases and provide one concrete example.",
    "use_case": "List a few realistic use cases and provide one concrete example.",
    "creative": "Write something creative that still uses the document facts (e.g., a short story or a poem).",
}


def _chunk_token_ids(token_ids: list[int], *, min_tokens: int, max_tokens: int) -> list[list[int]]:
    if min_tokens <= 0 or max_tokens <= 0 or min_tokens > max_tokens:
        raise ValueError("Invalid chunk size bounds.")
    if len(token_ids) == 0:
        return []

    chunk_size = max_tokens
    num_chunks = math.ceil(len(token_ids) / chunk_size)
    chunks = [token_ids[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]
    return [c for c in chunks if len(c) >= min_tokens]


def _apply_chat_template(tokenizer, messages: list[dict], *, add_generation_prompt: bool):
    if not hasattr(tokenizer, "apply_chat_template"):
        return None
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )
    except ValueError:
        # Many non-chat tokenizers expose `apply_chat_template` but do not have a template set.
        return None
    except TypeError:
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def synthesize_self_study_jsonl(
    *,
    output_path: Path,
    model,
    tokenizer,
    corpus_text: str,
    num_samples: int,
    seed_prompt_types: list[str],
    min_tokens_per_chunk: int,
    max_tokens_per_chunk: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    for t in seed_prompt_types:
        if t not in DEFAULT_SEED_PROMPTS:
            raise ValueError(f"Unknown seed prompt type '{t}', expected one of: {sorted(DEFAULT_SEED_PROMPTS)}")

    corpus_token_ids = tokenizer(
        corpus_text,
        return_tensors=None,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    chunks = _chunk_token_ids(corpus_token_ids, min_tokens=min_tokens_per_chunk, max_tokens=max_tokens_per_chunk)
    if not chunks:
        raise ValueError("Corpus too small after chunking; try lowering `min_tokens_per_chunk`.")

    device = getattr(model, "device", torch.device("cpu"))
    model.eval()

    rng = torch.Generator(device="cpu")
    for _ in range(num_samples):
        chunk_idx = int(torch.randint(low=0, high=len(chunks), size=(1,), generator=rng).item())
        prompt_idx = int(torch.randint(low=0, high=len(seed_prompt_types), size=(1,), generator=rng).item())

        chunk_text = tokenizer.decode(chunks[chunk_idx], skip_special_tokens=True)
        seed_prompt = DEFAULT_SEED_PROMPTS[seed_prompt_types[prompt_idx]]

        teacher_prompt = _apply_chat_template(
            tokenizer,
            [{"role": "system", "content": chunk_text}, {"role": "user", "content": seed_prompt}],
            add_generation_prompt=True,
        )
        student_prompt = _apply_chat_template(
            tokenizer,
            [{"role": "user", "content": seed_prompt}],
            add_generation_prompt=True,
        )
        if teacher_prompt is None or student_prompt is None:
            ctx_ids = tokenizer(chunk_text, add_special_tokens=False)["input_ids"]
            x_ids = tokenizer(seed_prompt, add_special_tokens=False)["input_ids"]
            teacher_prompt = torch.tensor(ctx_ids + x_ids, dtype=torch.long).unsqueeze(0)
            student_prompt = torch.tensor(x_ids, dtype=torch.long).unsqueeze(0)

        teacher_prompt = teacher_prompt.to(device)
        student_prompt = student_prompt.to(device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-5),
            "top_p": top_p,
            "pad_token_id": getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
        }

        teacher_out = model.generate(teacher_prompt, attention_mask=torch.ones_like(teacher_prompt), **gen_kwargs)
        student_out = model.generate(student_prompt, attention_mask=torch.ones_like(student_prompt), **gen_kwargs)

        teacher_gen = teacher_out[0].tolist()
        student_gen = student_out[0].tolist()
        teacher_ctx_len = int(teacher_prompt.shape[1])
        student_ctx_len = int(student_prompt.shape[1])

        record = {
            "teacher_input_ids": teacher_gen,
            "student_input_ids": student_gen,
            "ctx_len": teacher_ctx_len - student_ctx_len,
        }
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--seed_prompts", type=str, default="structuring,summarization,question,use_cases,creative")
    parser.add_argument("--min_tokens_per_chunk", type=int, default=512)
    parser.add_argument("--max_tokens_per_chunk", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--max_corpus_tokens",
        type=int,
        default=None,
        help="Optional cap on the number of tokens used from the corpus for chunking (useful for small-context models).",
    )
    args = parser.parse_args()

    corpus_text = Path(args.corpus_path).read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)

    if args.max_corpus_tokens is not None:
        ids = tokenizer(
            corpus_text,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_corpus_tokens,
        )["input_ids"]
        corpus_text = tokenizer.decode(ids, skip_special_tokens=True)

    synthesize_self_study_jsonl(
        output_path=Path(args.out_jsonl),
        model=model,
        tokenizer=tokenizer,
        corpus_text=corpus_text,
        num_samples=args.num_samples,
        seed_prompt_types=[s.strip() for s in args.seed_prompts.split(",") if s.strip()],
        min_tokens_per_chunk=args.min_tokens_per_chunk,
        max_tokens_per_chunk=args.max_tokens_per_chunk,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
