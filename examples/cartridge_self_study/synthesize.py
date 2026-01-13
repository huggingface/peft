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
import random
from pathlib import Path

from transformers import AutoTokenizer


SEED_PROMPTS = {
    "structuring": (
        "Generate a single instruction asking an LLM to structure information from the document above. "
        "Be specific about what section or topic to structure. "
        "Output only the instruction, nothing else."
    ),
    "summarization": (
        "Generate a single instruction asking an LLM to summarize part of the document above. "
        "Be explicit about which section to summarize. "
        "Output only the instruction, nothing else."
    ),
    "question": (
        "Generate a question that tests knowledge of the document above. "
        "Include specific details (names, dates, numbers) so the question is unambiguous. "
        "Output only the question, nothing else."
    ),
    "use_cases": (
        "Think of a practical real-world task someone could accomplish using knowledge from the document. "
        "Generate a single question or instruction reflecting that use case. "
        "Output only the question/instruction, nothing else."
    ),
    "creative": (
        "Generate a creative question inspired by the document above. Output only the question, nothing else."
    ),
}

# Chat template kwargs to disable thinking mode for models like Qwen3
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
MAX_NEW_TOKENS_FOR_QUESTIONS = 256


def synthesize_self_study_jsonl(
    *,
    output_path: Path,
    model,
    tokenizer,
    corpus_text: str,
    num_samples: int,
    seed_prompt_types: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    use_vllm: bool = False,
    seed: int = 0,
):
    """
    Synthesize self-study data for cartridge training.

    Uses the full corpus as context for all samples, varying only the seed prompt.
    With vLLM's prefix caching, the document KV cache is computed once and reused.

    If use_vllm=True, `model` should be a vllm.LLM instance.
    Otherwise, `model` should be a HuggingFace model.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    for t in seed_prompt_types:
        if t not in SEED_PROMPTS:
            raise ValueError(f"Unknown seed prompt type '{t}', expected one of: {sorted(SEED_PROMPTS)}")

    # Pre-generate prompt indices (cycling through seed prompt types).
    prompt_indices = [i % len(seed_prompt_types) for i in range(num_samples)]
    rng = random.Random(seed)
    rng.shuffle(prompt_indices)

    if use_vllm:
        _synthesize_vllm(
            output_path=output_path,
            model=model,
            tokenizer=tokenizer,
            corpus_text=corpus_text,
            seed_prompt_types=seed_prompt_types,
            prompt_indices=prompt_indices,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        _synthesize_hf(
            output_path=output_path,
            model=model,
            tokenizer=tokenizer,
            corpus_text=corpus_text,
            seed_prompt_types=seed_prompt_types,
            prompt_indices=prompt_indices,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )


def _synthesize_vllm(
    *,
    output_path: Path,
    model,
    tokenizer,
    corpus_text: str,
    seed_prompt_types: list[str],
    prompt_indices: list[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    """Synthesize using vLLM with prefix caching (two-stage like original cartridges).

    Stage 1: Generate questions using meta-prompts (all share document prefix)
    Stage 2: Generate answers to those questions (all share document prefix)
    """
    from vllm import SamplingParams

    # Stage 1: Generate questions
    question_messages = [
        [
            {"role": "system", "content": corpus_text},
            {"role": "user", "content": SEED_PROMPTS[seed_prompt_types[prompt_idx]]},
        ]
        for prompt_idx in prompt_indices
    ]

    question_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS_FOR_QUESTIONS,
        temperature=temperature if temperature > 0 else 0.0,
        top_p=top_p if temperature > 0 else 1.0,
    )

    print("Stage 1: Generating questions...")
    question_outputs = model.chat(
        question_messages,
        question_params,
        use_tqdm=True,
        chat_template_kwargs=CHAT_TEMPLATE_KWARGS,
    )
    questions = [out.outputs[0].text.strip() for out in question_outputs]

    # Stage 2: Generate answers
    answer_messages = [
        [
            {"role": "system", "content": corpus_text},
            {"role": "user", "content": question},
        ]
        for question in questions
    ]

    answer_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    print("Stage 2: Generating answers...")
    answer_outputs = model.chat(
        answer_messages,
        answer_params,
        use_tqdm=True,
        chat_template_kwargs=CHAT_TEMPLATE_KWARGS,
    )

    # Build training records
    for i, (question, answer_out) in enumerate(zip(questions, answer_outputs)):
        # Get the answer token IDs directly from vLLM output (avoids decode/re-encode mismatch)
        answer_ids = list(answer_out.outputs[0].token_ids)

        teacher_prompt_ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": corpus_text}, {"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,
            **CHAT_TEMPLATE_KWARGS,
        )
        student_prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,
            **CHAT_TEMPLATE_KWARGS,
        )

        record = {
            "teacher_input_ids": teacher_prompt_ids + answer_ids,
            "student_input_ids": student_prompt_ids + answer_ids,
            "ctx_len": len(teacher_prompt_ids) - len(student_prompt_ids),
        }
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def _synthesize_hf(
    *,
    output_path: Path,
    model,
    tokenizer,
    corpus_text: str,
    seed_prompt_types: list[str],
    prompt_indices: list[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    """Synthesize using HuggingFace transformers (two-stage, one sample at a time)."""
    import torch
    from tqdm import tqdm

    device = getattr(model, "device", torch.device("cpu"))
    model.eval()

    for prompt_idx in tqdm(prompt_indices, desc="Generating samples"):
        meta_prompt = SEED_PROMPTS[seed_prompt_types[prompt_idx]]

        # Stage 1: Generate question
        question_input = tokenizer.apply_chat_template(
            [{"role": "system", "content": corpus_text}, {"role": "user", "content": meta_prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
            **CHAT_TEMPLATE_KWARGS,
        ).to(device)

        gen_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS_FOR_QUESTIONS,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = max(temperature, 1e-5)
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            question_out = model.generate(question_input, **gen_kwargs)

        question_tokens = question_out[0, question_input.shape[1] :].tolist()
        question = tokenizer.decode(question_tokens, skip_special_tokens=True).strip()

        # Stage 2: Generate answer
        teacher_input = tokenizer.apply_chat_template(
            [{"role": "system", "content": corpus_text}, {"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
            **CHAT_TEMPLATE_KWARGS,
        ).to(device)

        student_input = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
            **CHAT_TEMPLATE_KWARGS,
        ).to(device)

        with torch.no_grad():
            answer_out = model.generate(
                teacher_input,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        answer_tokens = answer_out[0, teacher_input.shape[1] :].tolist()

        record = {
            "teacher_input_ids": answer_out[0].tolist(),
            "student_input_ids": student_input[0].tolist() + answer_tokens,
            "ctx_len": int(teacher_input.shape[1]) - int(student_input.shape[1]),
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
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--max_corpus_tokens",
        type=int,
        default=None,
        help="Optional cap on the number of tokens used from the corpus.",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM for faster generation with automatic prefix caching.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic prompt-type shuffling.")
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

        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")

    synthesize_self_study_jsonl(
        output_path=Path(args.out_jsonl),
        model=model,
        tokenizer=tokenizer,
        corpus_text=corpus_text,
        num_samples=args.num_samples,
        seed_prompt_types=[s.strip() for s in args.seed_prompts.split(",") if s.strip()],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_vllm=args.use_vllm,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
