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
from pathlib import Path

from synthesize import synthesize_self_study_jsonl
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--corpus_path",
        type=str,
        default=str(Path(__file__).resolve().parent / "data/cartridges.tex"),
    )
    parser.add_argument("--out_jsonl", type=str, default="distill.jsonl")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--seed_prompts", type=str, default="structuring,summarization,question,use_cases,creative")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_corpus_tokens", type=int, default=2048)
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
