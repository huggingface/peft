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
import math
import random
from pathlib import Path

from transformers import AutoTokenizer


DEFAULT_SEED_PROMPTS = {
    "structuring": "Please outline the key sections and how the information is structured.",
    "summarization": "Please summarize the most important points concisely.",
    "question": "Ask a question about the document above that requires careful reading, then answer it.",
    "use_cases": "List a few realistic use cases and provide one concrete example.",
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
    use_vllm: bool = False,
):
    """
    Synthesize self-study data for cartridge training.

    If use_vllm=True, `model` should be a vllm.LLM instance.
    Otherwise, `model` should be a HuggingFace model.
    """
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

    # Pre-generate all (chunk_idx, prompt_idx) pairs
    sample_pairs = [
        (random.randint(0, len(chunks) - 1), random.randint(0, len(seed_prompt_types) - 1)) for _ in range(num_samples)
    ]

    if use_vllm:
        _synthesize_vllm(
            output_path=output_path,
            model=model,
            tokenizer=tokenizer,
            chunks=chunks,
            seed_prompt_types=seed_prompt_types,
            sample_pairs=sample_pairs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        _synthesize_hf(
            output_path=output_path,
            model=model,
            tokenizer=tokenizer,
            chunks=chunks,
            seed_prompt_types=seed_prompt_types,
            sample_pairs=sample_pairs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )


def _synthesize_vllm(
    *,
    output_path: Path,
    model,
    tokenizer,
    chunks: list[list[int]],
    seed_prompt_types: list[str],
    sample_pairs: list[tuple[int, int]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    """Synthesize using vLLM with prefix caching for efficient batch generation."""
    from vllm import SamplingParams

    # Build all teacher prompts (with document context in system message)
    teacher_conversations = []
    student_input_ids_list = []

    for chunk_idx, prompt_idx in sample_pairs:
        chunk_text = tokenizer.decode(chunks[chunk_idx], skip_special_tokens=True)
        seed_prompt = DEFAULT_SEED_PROMPTS[seed_prompt_types[prompt_idx]]

        # Teacher conversation has document in system message
        teacher_conv = [
            {"role": "system", "content": chunk_text},
            {"role": "user", "content": seed_prompt},
        ]
        teacher_conversations.append(teacher_conv)

        # Student prompt (no document context)
        student_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": seed_prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        student_input_ids_list.append(student_ids)

    # Configure sampling
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature if temperature > 0 else 0.0,
        top_p=top_p if temperature > 0 else 1.0,
    )

    # Batch generate with vLLM (prefix caching happens automatically for shared prefixes)
    outputs = model.chat(
        messages=teacher_conversations,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    # Process outputs and write to file
    for i, output in enumerate(outputs):
        # Get the generated token ids
        generated_ids = list(output.outputs[0].token_ids)

        # Teacher: prompt + generated
        teacher_prompt_ids = tokenizer.apply_chat_template(
            teacher_conversations[i],
            tokenize=True,
            add_generation_prompt=True,
        )
        teacher_gen = teacher_prompt_ids + generated_ids

        # Student: prompt + same generated tokens
        student_gen = student_input_ids_list[i] + generated_ids

        teacher_ctx_len = len(teacher_prompt_ids)
        student_ctx_len = len(student_input_ids_list[i])

        record = {
            "teacher_input_ids": teacher_gen,
            "student_input_ids": student_gen,
            "ctx_len": teacher_ctx_len - student_ctx_len,
        }
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def _synthesize_hf(
    *,
    output_path: Path,
    model,
    tokenizer,
    chunks: list[list[int]],
    seed_prompt_types: list[str],
    sample_pairs: list[tuple[int, int]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    """Synthesize using HuggingFace transformers (slower, one sample at a time)."""
    import torch

    device = getattr(model, "device", torch.device("cpu"))
    model.eval()

    for chunk_idx, prompt_idx in sample_pairs:
        chunk_text = tokenizer.decode(chunks[chunk_idx], skip_special_tokens=True)
        seed_prompt = DEFAULT_SEED_PROMPTS[seed_prompt_types[prompt_idx]]

        teacher_input_ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": chunk_text}, {"role": "user", "content": seed_prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        student_input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": seed_prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        do_sample = temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
        }
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-5)
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            teacher_out = model.generate(teacher_input_ids, **gen_kwargs)

        teacher_prompt_len = int(teacher_input_ids.shape[1])
        generated_tokens = teacher_out[0, teacher_prompt_len:].tolist()

        teacher_gen = teacher_out[0].tolist()
        student_gen = student_input_ids[0].tolist() + generated_tokens

        teacher_ctx_len = int(teacher_input_ids.shape[1])
        student_ctx_len = int(student_input_ids.shape[1])

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
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM for faster generation with automatic prefix caching.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM (number of GPUs).",
    )
    args = parser.parse_args()

    corpus_text = Path(args.corpus_path).read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.max_corpus_tokens is not None:
        ids = tokenizer(
            corpus_text,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_corpus_tokens,
        )["input_ids"]
        corpus_text = tokenizer.decode(ids, skip_special_tokens=True)

    if args.use_vllm:
        from vllm import LLM

        model = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            enable_prefix_caching=True,
        )
    else:
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")

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
        use_vllm=args.use_vllm,
    )


if __name__ == "__main__":
    main()
