# Copyright 2026-present the HuggingFace Inc. team.
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

"""Consolidate SEIS API documentation with Cortex or FFN-only LoRA."""

import argparse
import hashlib
import json
import math
import random
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from peft import PeftModel, get_peft_model


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chat_utils import generation_kwargs, generation_status, parse_response, render_messages
from training_utils import (
    build_adapter_config,
    file_sha256,
    initialize_adapter,
    resolve_deepspeed,
    warmup_kwargs,
    write_training_run,
)


SUPPORTED_MODELS = {"llama", "qwen2", "qwen3", "gpt_oss"}
DATASET_REPO_ID = "MikeDean/cortex-seis-example"
DATASET_REVISION = "e0d9eec3ef8d05e99af1b0361591c2550c605be3"

GENERIC_QUESTIONS = [
    "Please introduce Elon Musk.",
    "Write a code to implement residual connection in PyTorch.",
    "Which planet in our solar system is the only one that could theoretically float in water because its density is lower than that of water?",
    "Please recommend some scenario of Shanghai, i will goto Shanghai in this winter.",
]

# Adapted from the Cortex research CODE prompt. Only the diversity minima are
# scaled down for small nq; nq=28 retains the original constraints.
CODE_QUERY_PROMPT = r"""## Role
You are a Senior Python QA Automation Engineer and Developer Educator.
Your goal is to generate {n_q} high-quality, diverse user queries based strictly on the provided **Doc Context**.

## Input Contract (Crucial)
- The FIRST user message in the conversation contains the API/code documentation for a Python package/library to be learned.
- Treat the FIRST user message as the ONLY source of truth about available symbols, parameters, behaviors, and dependencies.

## Mandatory Count (STRICT)
- Generate EXACTLY the number of questions requested by the user in the current message.

## Objective
Generate diverse, realistic user questions that test whether an AI can understand and APPLY the library’s API (not merely restate documentation).

## Hard Requirements (Distribution across the full set)
- >= 40% Applied Tasks: mini programming assignments requiring code implementation using the library’s API.
- >= 20% Parameter/Behavior Probing: what-if scenarios, edge cases, trade-offs, determinism, or error handling specific to the library.
- >= 20% Integration Tasks: use the library alongside Python standard libraries (e.g., os, pathlib, tempfile, json, io, hashlib, logging) and/or test frameworks explicitly mentioned in the documentation.
- Remaining questions: debugging, verification, refactoring, regression, reproducibility.
- If the requested count is too small to satisfy all percentages exactly, approximate them while prioritizing diversity and applied tasks.

## Definitions (Applied Task MUST include)
1) Constraints: concrete paths, non-default parameters, limits, environment constraints (CI, permissions, platform, etc.).
2) Verification: a concrete success check using ONLY allowed tools (assertions, expected exceptions, file existence/non-emptiness, return values, log capture, etc.).

## Generation Rules

1) Contextual Anchoring (Crucial)
- Extract the package name and available symbols (modules/classes/functions/methods) from the Doc Context.
- Every question MUST clearly be about THIS specific library (not generic Python).

2) Library Anchor MUST Include Package Name (No Ambiguity)
- EVERY question MUST include the top-level package name AT LEAST ONCE (e.g., `ocrtestdata`).
- You MAY additionally use short symbol names (e.g., `generate_pdf`) or fuzzy references (“the generator”), but the package name must still appear in the same question.
- Do NOT produce any question where the library is only implied and the package name is missing.

3) Naming Variety Without Overusing Full Paths
- Vary references naturally across the set:
  - package name only (e.g., `ocrtestdata`),
  - short symbol name (e.g., `generate_pdf`) + package name somewhere else in the same question,
  - occasional exact API symbol as written in the Doc Context.
- Fully-qualified paths (e.g., `pkg.mod.func`) are allowed but must be limited:
  - <= 15% of questions may include a fully-qualified symbol/path.
  - If used in a question, include the fully-qualified name ONLY ONCE, then use the short name for subsequent mentions.

4) Anchor Placement Variety (Beginning / Middle / End)
- Distribute where the package anchor appears:
  - >= 25% near the END (e.g., “… using ocrtestdata.”).
  - >= 25% in the MIDDLE.
  - >= 25% near the BEGINNING.
- Do NOT front-load the package name in every question.

5) Strict API & Dependency Fidelity (No Hallucinations)
- Use ONLY symbols, parameters, and third-party dependencies explicitly present in the Doc Context.
- If a dependency like `Faker` is not mentioned in the Doc Context, do NOT reference it.
- Do NOT invent extra parameters, return fields, exceptions, helper functions, or filesystem behaviors (mkdir, overwrite) unless stated.

6) Closed-World Answerability (If Unknown, Observe)
- Do NOT ask for hidden defaults or unspecified behaviors as facts.
- If behavior might be unknown, phrase the question as an experiment to OBSERVE it using allowed tools:
  - print exception type/message, check file existence/non-emptiness, compare return values, capture logs.
- Do NOT demand a specific exception type/message unless the Doc Context explicitly guarantees it.

7) Anti-Flake Verification (Stability FIRST)
- Avoid brittle assertions unless explicitly guaranteed by the Doc Context:
  - NO fixed size thresholds (e.g., “> 50KB”, “> 1MB”, “> 10KB”).
  - NO strict time deadlines (e.g., “< 5 seconds”, “< 30 seconds”).
  - NO assumptions that output contains plaintext substrings.
  - NO requiring exact byte equality / exact hash equality unless determinism is explicitly stated.
  - NO monotonic size assertions like “dpi=300 must be larger than dpi=72” unless guaranteed.
- Preferred alternatives:
  - assert stable invariants (file exists + non-empty + returned value),
  - print/record sizes or hashes for comparison (observation) without forcing a strict ordering,
  - treat determinism as “diagnose/observe and report”.

8) Code-First Realism
- >= 70% of questions should request runnable Python code (script or pytest-style), with imports and a concrete path strategy (temp dirs, pathlib), and cleanup when appropriate.
- Reflect real workflows: CI-friendliness, reproducibility, logging, error handling, batch generation.

9) Diversity & De-duplication (HIGH PRIORITY)
- No duplicates or near-duplicates.
- Each question must differ in at least TWO dimensions:
  - goal, data shape (single vs batch), environment constraint, verification style, parameter focus, integration module, debugging vs green-path, determinism vs trade-off.
- Opening diversity hard constraints:
  - Do NOT start more than 2 questions with the same first 4 words.
  - Ensure at least {min_openings} distinct opening patterns across the set.
  - Ensure at least {min_styles} “debug/diagnose” style questions (“I’m seeing…”, “Why does… fail…”, “Help me reproduce…”).
  - Ensure at least {min_styles} “what-if/edge case” questions.
  - Ensure at least {min_styles} “integration” questions using different stdlib modules (within allowed set).

## Output Format (Strict)
Output ONLY the questions, one per line, wrapped in <question> tags.
Do NOT include numbering, categories, explanations, or extra text.

## Examples (Format Only; placeholders)
<question>I’m trying to generate a one-page PDF into a temp directory and assert it’s non-empty — what’s a clean pytest test for that using ocrtestdata?</question>
<question>What happens if pages_per_pdf is negative? Can you show a tiny repro that prints the observed exception type/message using ocrtestdata?</question>
<question>Can you help me batch-generate outputs into a nested folder (pathlib) and verify the expected files are created, using ocrtestdata at the end?</question>"""


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_data_file(path=None):
    """Use a local JSONL override or the pinned SEIS example on the Hub."""
    if path is not None:
        return Path(path)
    return Path(
        hf_hub_download(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            filename="data.jsonl",
            revision=DATASET_REVISION,
        )
    )


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_source(path):
    sources = [row for row in read_jsonl(path) if row.get("type") == "context"]
    if len(sources) != 1:
        raise ValueError("The data file must contain exactly one type=context row.")
    source = sources[0]
    if not source.get("context", "").strip() or not source.get("package", "").strip():
        raise ValueError("The context row needs nonempty context and package strings.")
    # Return only documentation fields, never evaluation questions or tests.
    return {"context": source["context"].strip(), "package": source["package"].strip()}


def question_prompt(nq):
    return CODE_QUERY_PROMPT.format(n_q=nq, min_openings=min(8, nq), min_styles=min(4, nq // 4))


def preparation_generation_kwargs(args, kind):
    # Match the research generator; leave top_p/top_k to the checkpoint.
    # Sampling is independent of whether the chat template enables reasoning.
    return {
        "do_sample": getattr(args, f"{kind}_do_sample"),
        "temperature": getattr(args, f"{kind}_temperature"),
    }


def parse_questions(text: str, nq: int, package: str, *, allow_partial: bool = False) -> list[str]:
    """Select up to nq valid, distinct tagged questions; normally require a full set."""
    tagged = re.findall(r"<question>(.*?)</question>", text, flags=re.DOTALL)
    questions, seen = [], set()
    duplicates, invalid = 0, 0
    for candidate in tagged:
        question = candidate.strip()
        if not question or package not in question or re.search(r"</?question\b", question):
            invalid += 1
            continue
        key = " ".join(question.split())
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        questions.append(question)
    if len(questions) < nq and not allow_partial:
        raise ValueError(
            f"Question generation needs at least {nq} distinct questions naming {package!r}; "
            f"found {len(tagged)} tags and {len(questions)} valid unique questions "
            f"({duplicates} duplicates, {invalid} invalid)."
        )
    return questions[:nq]


def generate_questions(model, tokenizer, source, args):
    messages = [
        {"role": "system", "content": question_prompt(args.nq)},
        {"role": "user", "content": f"Here are the prompt:\n<prompt>\n{source['context']}\n</prompt>"},
    ]
    prefix = render_messages(
        tokenizer,
        messages,
        model.config.model_type,
        add_generation_prompt=True,
        enable_thinking=args.question_thinking,
        reasoning_effort=args.reasoning_effort,
    )
    pooled_questions = {}
    raw_attempts = []
    last_error = None
    # Seed once in prepare; sampling retries must advance the RNG normally.
    for attempt in range(1, args.question_attempts + 1):
        print(f"Generating questions: attempt {attempt}/{args.question_attempts}", flush=True)
        ids = generate_answer(
            model,
            tokenizer,
            prefix,
            args.question_max_new_tokens,
            args.question_max_length,
            allow_empty=True,
            thinking=args.question_thinking,
            sampling_kwargs=preparation_generation_kwargs(args, "question"),
        )
        text = tokenizer.decode(ids, skip_special_tokens=False)
        raw_attempts.append(f"=== Question generation attempt {attempt} ===\n{text}")
        # Keep every attempt outside the temporary cache, including on failure.
        diagnostic = args.question_diagnostic
        attempt_diagnostic = diagnostic.with_name(f"{diagnostic.stem}.attempt-{attempt}{diagnostic.suffix}")
        attempt_diagnostic.write_text(text + "\n", encoding="utf-8")
        diagnostic.write_text(text + "\n", encoding="utf-8")
        response, complete = parse_response(
            tokenizer,
            ids,
            model.config.model_type,
            prefix_ids=prefix,
            enable_thinking=args.question_thinking,
            eos_token_ids=model.generation_config.eos_token_id,
        )
        status = generation_status(
            tokenizer,
            ids,
            args.question_max_new_tokens,
            complete=complete,
            eos_token_ids=model.generation_config.eos_token_id,
        )
        try:
            if not status["complete"]:
                raise ValueError(
                    f"Question generation is incomplete ({status['finish_reason']}); "
                    "consider increasing --question-max-new-tokens."
                )
            questions = parse_questions(response["content"], args.nq, source["package"])
        except ValueError as error:
            last_error = error
            # Only completed final responses contribute candidates, never reasoning or truncated output.
            if status["complete"]:
                candidates = parse_questions(response["content"], args.nq, source["package"], allow_partial=True)
                for question in candidates:
                    pooled_questions.setdefault(" ".join(question.split()), question)
            print(
                f"Question attempt {attempt} rejected: {error} "
                f"Pooled valid unique questions: {len(pooled_questions)}. Raw output: {attempt_diagnostic}",
                flush=True,
            )
            continue
        print(f"Accepted {len(questions)} distinct questions from attempt {attempt}.", flush=True)
        return questions, text

    # Prefer a complete set from one attempt; pool only after every attempt failed.
    if len(pooled_questions) >= args.nq:
        questions = random.Random(args.seed).sample(list(pooled_questions.values()), args.nq)
        print(
            f"No single attempt supplied enough questions after {args.question_attempts} attempts. "
            f"Randomly selected {args.nq} of {len(pooled_questions)} pooled valid unique questions "
            f"with seed {args.seed}.",
            flush=True,
        )
        return questions, "\n\n".join(raw_attempts)
    raise ValueError(
        f"Question generation failed after {args.question_attempts} attempts. "
        f"Pooled {len(pooled_questions)} valid unique questions; need at least {args.nq}. "
        f"Last failure: {last_error} Inspect {args.question_diagnostic} and the attempt files, "
        "or supply independent questions with --questions-file."
    ) from last_error


def resolve_device(name, dtype):
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This small example supports one CPU or CUDA device.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable. Use --device cpu --dtype float32 when preparing.")
        torch.cuda.set_device(device)
        if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA device does not support BF16.")
    elif dtype != "float32":
        raise ValueError("Use --dtype float32 for the CPU example.")
    return device


def load_base(model_id, revision, dtype, device):
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    if config.model_type not in SUPPORTED_MODELS:
        raise ValueError(f"This example supports {sorted(SUPPORTED_MODELS)}, not all Cortex models.")
    if config.model_type == "gpt_oss" and getattr(config, "quantization_config", None):
        raise ValueError("Use a dequantized BF16 GPT-OSS checkpoint, not the original MXFP4 weights.")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=getattr(torch, dtype), use_kernels=False
    ).to(device)
    model.config.use_cache = False
    model.eval()
    model.requires_grad_(False)
    return model


def decoder_layers(model):
    base = model.get_base_model() if isinstance(model, PeftModel) else model
    return base.model.layers


def prompt_messages(question, context=""):
    messages = [{"role": "system", "content": context}] if context else []
    return messages + [{"role": "user", "content": question}]


def prompt_ids(tokenizer, question, context="", *, model_type="llama", thinking=False, reasoning_effort="medium"):
    return render_messages(
        tokenizer,
        prompt_messages(question, context),
        model_type,
        add_generation_prompt=True,
        enable_thinking=thinking,
        reasoning_effort=reasoning_effort,
    )


def input_tensor(ids, device):
    return torch.tensor([ids], dtype=torch.long, device=device)


@torch.no_grad()
def generate_answer(
    model, tokenizer, prefix, max_new_tokens, max_length, allow_empty=False, *, thinking=False, sampling_kwargs=None
):
    if len(prefix) + max_new_tokens > max_length:
        raise ValueError(
            "Prompt plus generation budget exceeds the total length limit; increase it instead of truncating the source context."
        )
    ids = input_tensor(prefix, model.device)
    if sampling_kwargs is None:
        # Preparation and evaluation pass explicit sampling settings.
        sampling_kwargs = {"do_sample": False, **generation_kwargs(thinking)}
    output = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        **sampling_kwargs,
    )
    answer = output[0, ids.shape[1] :].tolist()
    if not allow_empty and (not answer or not tokenizer.decode(answer, skip_special_tokens=True).strip()):
        raise ValueError(
            "The teacher generated an empty answer. Inspect the prompt or choose a stronger instruction model."
        )
    return answer


@contextmanager
def capture_states(model, positions=None):
    states = []

    def capture(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        states.append(hidden if positions is None else hidden[0, positions, :])

    # Match Transformers' hidden_states[1:]: intermediate decoder outputs and
    # the final normalized state, excluding the embedding. The research loss
    # uses this same sequence on both the teacher and student.
    base = model.get_base_model() if isinstance(model, PeftModel) else model
    modules = [*decoder_layers(model)[:-1], base.model.norm]
    handles = [module.register_forward_hook(capture) for module in modules]
    try:
        yield states
    finally:
        for handle in handles:
            handle.remove()


def align_prompt_suffix(teacher_prefix, student_prefix, student_content_start=0):
    """Find the identical question/assistant suffix, excluding teacher-only context."""
    if not 0 <= student_content_start < len(student_prefix):
        raise ValueError("The student prompt must contain question tokens to align.")
    suffix = student_prefix[student_content_start:]
    teacher_start = len(teacher_prefix) - len(suffix)
    if teacher_start < 0 or teacher_prefix[teacher_start:] != suffix:
        raise ValueError("Teacher and student question tokens do not align; use a compatible chat template.")
    return teacher_start, student_content_start


def question_token_start(student_prefix, empty_question_prefix):
    """Exclude the rendered system/user header without assuming model-specific tokens."""
    start = 0
    for actual, empty in zip(student_prefix, empty_question_prefix):
        if actual != empty:
            break
        start += 1
    if start >= len(student_prefix):
        raise ValueError("No question tokens remain after the chat header.")
    return start


def forward_targets(model, input_ids, start, cache_states=True):
    """Return same-token logits [tokens, vocab] and optional states [layers, tokens, hidden]."""
    ids = input_tensor(input_ids, model.device)
    kwargs = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "use_cache": False}
    if cache_states:
        with capture_states(model, slice(start, None)) as states:
            logits = model(**kwargs).logits
        return logits[0, start:], torch.stack(states)
    return model(**kwargs).logits[0, start:], None


class DistillationCollator:
    """Right-pad/truncate student sequences and align the cached teacher targets."""

    def __init__(self, records, tensors, pad_token_id, max_length, kl_scope="oi", mse_scope="oi", include_states=True):
        if kl_scope not in {"o", "oi"} or mse_scope not in {"o", "oi"}:
            raise ValueError("Loss scopes must be 'o' or 'oi'.")
        self.records, self.tensors = records, tensors
        self.pad_token_id, self.max_length = pad_token_id, max_length
        self.kl_scope, self.mse_scope = kl_scope, mse_scope
        self.include_states = include_states
        for record in records:
            if min(len(record["student_prompt_ids"]) + len(record["answer_ids"]), max_length) <= len(
                record["student_prompt_ids"]
            ):
                raise ValueError("Training truncation removes every answer token; increase --max-train-length.")

    def __call__(self, features):
        indices = [item["index"] for item in features]
        lengths = [
            min(len(self.records[i]["student_prompt_ids"]) + len(self.records[i]["answer_ids"]), self.max_length)
            for i in indices
        ]
        batch_size, length = len(indices), max(lengths)
        first = str(indices[0])
        top_k = self.tensors[f"{first}.top_ids"].shape[-1]
        batch = {
            "input_ids": torch.full((batch_size, length), self.pad_token_id, dtype=torch.long),
            "attention_mask": torch.zeros(batch_size, length, dtype=torch.long),
            "teacher_top_logits": torch.zeros(
                batch_size, length, top_k, dtype=self.tensors[f"{first}.top_logits"].dtype
            ),
            "teacher_top_ids": torch.zeros(batch_size, length, top_k, dtype=torch.long),
            "kl_mask": torch.zeros(batch_size, length - 1, dtype=torch.bool),
            "mse_mask": torch.zeros(batch_size, length, dtype=torch.bool),
        }
        if self.include_states:
            if f"{first}.states" not in self.tensors:
                raise ValueError("MSE needs hidden-state targets; prepare a new cache with --cache-states.")
            states = self.tensors[f"{first}.states"]
            batch["teacher_states"] = torch.zeros(
                batch_size, states.shape[0], length, states.shape[2], dtype=states.dtype
            )
        for row, (index, size) in enumerate(zip(indices, lengths)):
            record = self.records[index]
            start = record["student_alignment_start"]
            answer_start = len(record["student_prompt_ids"])
            ids = record["student_prompt_ids"] + record["answer_ids"]
            batch["input_ids"][row, :size] = torch.tensor(ids[:size])
            batch["attention_mask"][row, :size] = 1
            for target, field in (("teacher_top_logits", "top_logits"), ("teacher_top_ids", "top_ids")):
                batch[target][row, start:size] = self.tensors[f"{index}.{field}"][: size - start]
            if self.include_states:
                batch["teacher_states"][row, :, start:size] = self.tensors[f"{index}.states"][:, : size - start]
            # A logit at j predicts token j+1. The final sequence token predicts no supervised token.
            kl_start = max(start, answer_start - 1) if self.kl_scope == "o" else start
            mse_start = max(start, answer_start) if self.mse_scope == "o" else start
            batch["kl_mask"][row, kl_start : size - 1] = True
            batch["mse_mask"][row, mse_start:size] = True
        return batch


def masked_distillation_loss(logits, states, batch, temperature, kl_weight, mse_weight):
    """Token-mean KL and layer/token/hidden-mean MSE with independent masks."""
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("Temperature must be finite and positive.")
    kl = logits.new_zeros((), dtype=torch.float32)
    mse = logits.new_zeros((), dtype=torch.float32)
    if kl_weight:
        count = batch["kl_mask"].sum()
        if not count:
            raise ValueError("No valid KL prediction positions remain.")

        def chunk_kl(student_logits, target_logits, target_ids, mask):
            student = student_logits[mask].float() / temperature
            teacher_logp = F.log_softmax(target_logits[mask].float() / temperature, dim=-1)
            student_logp = student.gather(-1, target_ids[mask]) - torch.logsumexp(student, dim=-1, keepdim=True)
            return (teacher_logp.exp() * (teacher_logp - student_logp)).sum()

        # Split once: each chunk's backward only constructs its own gradient,
        # rather than scattering into a full [batch, length, vocab] buffer repeatedly.
        chunks = zip(
            logits.flatten(0, 1).split(128),
            batch["teacher_top_logits"].flatten(0, 1).split(128),
            batch["teacher_top_ids"].flatten(0, 1).split(128),
            F.pad(batch["kl_mask"], (0, 1), value=False).flatten().split(128),
        )
        for values in chunks:
            if not values[-1].any():
                continue
            # Recompute the small loss chunk instead of retaining full-vocabulary FP32 copies.
            part = checkpoint(chunk_kl, *values, use_reentrant=False) if logits.requires_grad else chunk_kl(*values)
            kl = kl + part
        kl = kl * temperature**2 / count
    if mse_weight:
        if states is None or "teacher_states" not in batch:
            raise ValueError("MSE requires cached teacher and live student states.")
        mask = batch["mse_mask"]
        if not mask.any():
            raise ValueError("No valid MSE positions remain.")
        for student_layer, teacher_layer in zip(states.unbind(1), batch["teacher_states"].unbind(1)):
            mse = mse + F.mse_loss(student_layer[mask].float(), teacher_layer[mask].float())
        mse = mse / states.shape[1]
    return kl_weight * kl + mse_weight * mse, kl, mse


class DistillationTrainer(Trainer):
    def __init__(self, *args, temperature, kl_weight, mse_weight, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature, self.kl_weight, self.mse_weight = temperature, kl_weight, mse_weight
        self.model_accepts_loss_kwargs = False
        self.loss_history = []

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        kwargs = {key: inputs[key] for key in ("input_ids", "attention_mask")}
        if self.mse_weight:
            with capture_states(self.model) as captured:
                output = model(**kwargs, use_cache=False)
            states = torch.stack(captured, dim=1)
        else:
            output = model(**kwargs, use_cache=False)
            states = None
        loss, kl, mse = masked_distillation_loss(
            output.logits, states, inputs, self.temperature, self.kl_weight, self.mse_weight
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite distillation loss.")
        self.loss_history.append(
            {
                "micro_step": len(self.loss_history) + 1,
                "step": self.state.global_step + 1,
                "loss": loss.detach().item(),
                "kl": kl.detach().item(),
                "mse": mse.detach().item(),
            }
        )
        return (loss, output) if return_outputs else loss


def prepare(args):
    if getattr(args, "cache_states", None) is None:
        args.cache_states = args.mse_weight > 0
    if args.output_dir.exists():
        raise ValueError("Choose a new cache directory; an existing teacher cache cannot be overwritten.")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Publish a cache only after all records and checksums have been written.
    with tempfile.TemporaryDirectory(prefix=".cortex-prepare-", dir=args.output_dir.parent) as directory:
        working = SimpleNamespace(**vars(args))
        working.output_dir = Path(directory) / "cache"
        working.question_diagnostic = args.output_dir.with_name(args.output_dir.name + ".question_generation.txt")
        working.answer_diagnostic = args.output_dir.with_name(args.output_dir.name + ".incomplete_answer.json")
        working.questions_output = args.output_dir.with_name(args.output_dir.name + ".questions.json")
        _prepare(working)
        working.output_dir.rename(args.output_dir)
        working.question_diagnostic.unlink(missing_ok=True)
        working.answer_diagnostic.unlink(missing_ok=True)
    print(f"Prepared {args.output_dir}; inspect train_qa.jsonl and manifest.json.")


def _prepare(args):
    source = load_source(resolve_data_file(args.data_file))
    context = source["context"]
    generic = GENERIC_QUESTIONS
    set_seed(args.seed)
    device = resolve_device(args.device, args.dtype)
    model = load_base(args.model_name_or_path, args.revision, args.dtype, device)
    revision = getattr(model.config, "_commit_hash", None) or args.revision
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.questions_file:
        questions = read_json(args.questions_file)
        if (
            not isinstance(questions, list)
            or len(questions) != args.nq
            or any(not isinstance(q, str) for q in questions)
        ):
            raise ValueError("--questions-file must be a JSON array containing exactly --nq question strings.")
        raw_questions = None
    else:
        questions, raw_questions = generate_questions(model, tokenizer, source, args)
    questions = [q.strip() for q in questions]
    if any(not q for q in questions) or len(set(questions + generic)) != len(questions) + 4:
        raise ValueError(
            "Training questions must be nonempty and distinct. Review generation or supply --questions-file."
        )
    # Keep the selected question bank even if a later answer or target pass fails.
    # Reuse is explicit through --questions-file; this is not a partial teacher cache.
    if args.questions_file is None or args.questions_file.resolve() != args.questions_output.resolve():
        write_json(args.questions_output, questions)
    args.output_dir.mkdir(parents=True)
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    if raw_questions is not None:
        (args.output_dir / "question_generation.txt").write_text(raw_questions + "\n", encoding="utf-8")
    records, tensors, qa = [], {}, []
    for index, question in enumerate(questions + generic):
        kind = "target" if index < len(questions) else "generic"
        sample_id = f"{source['package']}-{index:04d}"
        teacher_context = context if kind == "target" or args.generic_context == "source" else ""
        teacher_messages = prompt_messages(question, teacher_context)
        student_messages = prompt_messages(question)
        render_options = {
            "add_generation_prompt": True,
            "enable_thinking": args.thinking,
            "reasoning_effort": args.reasoning_effort,
        }
        teacher_prefix = render_messages(tokenizer, teacher_messages, model.config.model_type, **render_options)
        student_prefix = render_messages(tokenizer, student_messages, model.config.model_type, **render_options)
        empty_prefix = render_messages(tokenizer, prompt_messages(""), model.config.model_type, **render_options)
        teacher_start, student_start = align_prompt_suffix(
            teacher_prefix, student_prefix, question_token_start(student_prefix, empty_prefix)
        )
        for attempt in range(1, args.answer_attempts + 1):
            print(
                f"Generating {kind} answer {index + 1}/{len(questions) + 4}: attempt {attempt}/{args.answer_attempts}",
                flush=True,
            )
            answer = generate_answer(
                model,
                tokenizer,
                teacher_prefix,
                args.max_new_tokens,
                args.answer_max_length,
                allow_empty=True,
                thinking=args.thinking,
                sampling_kwargs=preparation_generation_kwargs(args, "answer"),
            )
            response, complete = parse_response(
                tokenizer,
                answer,
                model.config.model_type,
                prefix_ids=teacher_prefix,
                enable_thinking=args.thinking,
                eos_token_ids=model.generation_config.eos_token_id,
            )
            status = generation_status(
                tokenizer,
                answer,
                args.max_new_tokens,
                complete=complete,
                eos_token_ids=model.generation_config.eos_token_id,
            )
            raw_response = tokenizer.decode(answer, skip_special_tokens=False)
            if status["complete"]:
                break
            diagnostic = {
                "id": sample_id,
                "attempt": attempt,
                "max_new_tokens": args.max_new_tokens,
                "num_generated_tokens": len(answer),
                "teacher_messages": teacher_messages,
                "raw_response": raw_response,
                "response": response,
                "generation": status,
            }
            attempt_diagnostic = args.answer_diagnostic.with_name(
                f"{args.answer_diagnostic.stem}.{index:04d}.attempt-{attempt}{args.answer_diagnostic.suffix}"
            )
            write_json(attempt_diagnostic, diagnostic)
            write_json(args.answer_diagnostic, diagnostic)
            print(
                f"Answer {sample_id} is incomplete ({status['finish_reason']}); "
                f"generated {len(answer)}/{args.max_new_tokens} tokens. Inspect {attempt_diagnostic}.",
                flush=True,
            )
            if attempt == args.answer_attempts:
                raise ValueError(
                    f"Incomplete teacher answer for {sample_id} ({status['finish_reason']}) "
                    f"after {args.answer_attempts} attempts. Inspect {args.answer_diagnostic}; "
                    "increase --max-new-tokens with a matching --answer-max-length, or fix the response format. "
                    f"Selected questions are saved at {args.questions_output} for explicit --questions-file reuse."
                )
        # Distillation uses the original generated tokens, including reasoning and termination.
        # Never reconstruct these tokens by re-rendering the portable messages.
        with torch.no_grad():
            logits, states = forward_targets(model, teacher_prefix + answer, teacher_start, args.cache_states)
            top_k = logits.shape[-1] if args.teacher_top_k == 0 else min(args.teacher_top_k, logits.shape[-1])
            # Taking top-k before casting avoids a full-vocabulary FP32 copy.
            values, indices = logits.topk(top_k, dim=-1)
        tensors[f"{index}.top_logits"] = values.float().cpu()
        tensors[f"{index}.top_ids"] = indices.cpu()
        if states is not None:
            tensors[f"{index}.states"] = states.cpu()
        del logits, states, values, indices
        records.append(
            {
                "id": sample_id,
                "kind": kind,
                "teacher_prompt_ids": teacher_prefix,
                "student_prompt_ids": student_prefix,
                "answer_ids": answer,
                "teacher_alignment_start": teacher_start,
                "student_alignment_start": student_start,
                "raw_response": raw_response,
            }
        )
        qa.append(
            {
                "id": sample_id,
                "kind": kind,
                "teacher_messages": teacher_messages,
                "student_messages": student_messages,
                "response": response,
                "generation": status,
            }
        )
        print(f"Prepared {kind} {index + 1}/{len(questions) + 4}: {question}", flush=True)
    save_file(tensors, str(args.output_dir / "targets.safetensors"))
    (args.output_dir / "train_qa.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in qa), encoding="utf-8"
    )
    manifest = {
        "schema_version": 4,
        "model_name_or_path": str(Path(args.model_name_or_path).resolve())
        if Path(args.model_name_or_path).is_dir()
        else args.model_name_or_path,
        "revision": revision,
        "requested_revision": args.revision,
        "model_type": model.config.model_type,
        "dtype": args.dtype,
        "seed": args.seed,
        "context": context,
        "nq": args.nq,
        "question_source": "file" if args.questions_file else "model",
        "package": source["package"],
        "question_prompt": question_prompt(args.nq),
        "questions_file_sha256": file_sha256(args.questions_file) if args.questions_file else None,
        "generic_questions": generic,
        "question_max_length": args.question_max_length,
        "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        "question_max_new_tokens": args.question_max_new_tokens,
        "question_attempts": args.question_attempts,
        "answer_max_new_tokens": args.max_new_tokens,
        "answer_attempts": args.answer_attempts,
        "generic_context": args.generic_context,
        "answer_prompt_format": "context_system_user_v1",
        "base_generation_config": model.generation_config.to_dict(),
        "answer_generation": preparation_generation_kwargs(args, "answer"),
        "question_generation": preparation_generation_kwargs(args, "question"),
        "thinking": args.thinking,
        "question_thinking": args.question_thinking,
        "reasoning_effort": args.reasoning_effort,
        "top_k": args.teacher_top_k,
        "max_length": args.answer_max_length,
        "supervision": "aligned_input_output",
        "cache_states": args.cache_states,
        "state_format": "transformers_hidden_states_without_embedding_v1" if args.cache_states else None,
        "train_qa_sha256": file_sha256(args.output_dir / "train_qa.jsonl"),
        "targets_sha256": file_sha256(args.output_dir / "targets.safetensors"),
        "records": records,
    }
    write_json(args.output_dir / "manifest.json", manifest)


def load_prepared_metadata(directory):
    """Read the manifest and verified messages without accessing teacher tensors."""
    manifest = read_json(directory / "manifest.json")
    if manifest["schema_version"] != 4:
        raise ValueError("This example requires schema-4 prepared metadata. Prepare a new cache.")
    if file_sha256(directory / "train_qa.jsonl") != manifest["train_qa_sha256"]:
        raise ValueError("Training messages do not match the manifest checksum.")
    qa = read_jsonl(directory / "train_qa.jsonl")
    if [row["id"] for row in qa] != [record["id"] for record in manifest["records"]]:
        raise ValueError("Training messages and cached targets have different sample IDs or ordering.")
    if not all(row["generation"]["complete"] for row in qa):
        raise ValueError("The teacher cache contains incomplete responses.")
    return manifest, qa


def load_prepared(directory):
    """Load verified metadata and the teacher targets required for distillation."""
    manifest, _ = load_prepared_metadata(directory)
    if manifest["cache_states"] and manifest.get("state_format") != "transformers_hidden_states_without_embedding_v1":
        raise ValueError("Cached state format differs from the research MSE targets. Prepare a new cache.")
    if file_sha256(directory / "targets.safetensors") != manifest["targets_sha256"]:
        raise ValueError("Teacher targets do not match the manifest checksum.")
    tensors = load_file(str(directory / "targets.safetensors"), device="cpu")
    for index, record in enumerate(manifest["records"]):
        teacher_ids = record["teacher_prompt_ids"] + record["answer_ids"]
        student_ids = record["student_prompt_ids"] + record["answer_ids"]
        teacher_start, student_start = record["teacher_alignment_start"], record["student_alignment_start"]
        if not (
            0 <= teacher_start < len(record["teacher_prompt_ids"])
            and 0 <= student_start < len(record["student_prompt_ids"])
        ):
            raise ValueError("Invalid cached input/output alignment offsets.")
        if teacher_ids[teacher_start:] != student_ids[student_start:]:
            raise ValueError("Cached teacher/student token spans do not align.")
        n = len(student_ids) - student_start
        values, ids = tensors[f"{index}.top_logits"], tensors[f"{index}.top_ids"]
        if values.ndim != 2 or values.shape != ids.shape or values.shape[0] != n:
            raise ValueError("Cached logits are not aligned with the input/output tokens.")
        if manifest["cache_states"]:
            states = tensors.get(f"{index}.states")
            if states is None or states.ndim != 3 or states.shape[1] != n:
                raise ValueError("Cached states are not aligned with the input/output tokens.")
    return manifest, tensors


def attach_adapter(base, args, revision):
    config = build_adapter_config(args, base.config, revision)
    model = get_peft_model(base, config)
    initialize_adapter(model, args, base.config)
    return model


def train(args):
    if args.output_dir.exists():
        raise ValueError("Choose a new --output-dir for each run.")
    manifest, tensors = load_prepared(args.data_dir)
    args.model_name_or_path = manifest["model_name_or_path"]
    args.revision, args.dtype = manifest["revision"], manifest["dtype"]
    args.thinking, args.reasoning_effort = manifest["thinking"], manifest["reasoning_effort"]
    device = resolve_device(args.device, args.dtype)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.data_dir / "tokenizer")
    records = manifest["records"]
    dataset_indices = list(range(len(records)))
    collator = DistillationCollator(
        records,
        tensors,
        tokenizer.pad_token_id,
        args.max_train_length,
        args.kl_scope,
        args.mse_scope,
        args.mse_weight > 0,
    )
    if args.mse_weight > 0 and not manifest["cache_states"]:
        raise ValueError("MSE needs hidden-state targets; prepare a new cache with --cache-states.")
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        optim="adamw_torch",
        bf16=args.dtype == "bfloat16",
        use_cpu=device.type == "cpu",
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=resolve_deepspeed(
            args.deepspeed, batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps
        ),
        remove_unused_columns=False,
        label_names=[],
        dataloader_pin_memory=device.type == "cuda",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        **warmup_kwargs(args.warmup_ratio),
    )
    if training_args.world_size != 1 or training_args.n_gpu > 1:
        raise ValueError("This example uses one GPU; select one with CUDA_VISIBLE_DEVICES and launch one process.")
    base = load_base(args.model_name_or_path, args.revision, args.dtype, device)
    model = attach_adapter(base, args, args.revision)
    model.print_trainable_parameters()
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainer = DistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=[{"index": i} for i in dataset_indices],
        data_collator=collator,
        processing_class=tokenizer,
        temperature=args.kl_temperature,
        kl_weight=args.kl_weight,
        mse_weight=args.mse_weight,
    )
    result = trainer.train()
    model = trainer.accelerator.unwrap_model(trainer.model_wrapped, keep_fp32_wrapper=False)
    model.save_pretrained(args.output_dir, save_embedding_layers=False)
    tokenizer.save_pretrained(args.output_dir)
    write_training_run(
        args,
        result,
        objective="distillation",
        data={
            "training_file_sha256": file_sha256(args.data_dir / "train_qa.jsonl"),
            "manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        },
        loss_history=trainer.loss_history,
        trainable_parameters=trainable_parameters,
        preprocessing={
            "training_records": len(records),
            "truncated_records": sum(
                len(r["student_prompt_ids"]) + len(r["answer_ids"]) > args.max_train_length for r in records
            ),
        },
    )
    print(f"Saved {args.output_dir}. Run eval.py to inspect held-out behavior.")


def run(args):
    if args.stage == "prepare":
        preparation = SimpleNamespace(**vars(args))
        preparation.output_dir = args.data_dir
        prepare(preparation)
    else:
        train(args)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "train"], required=True)
    parser.add_argument(
        "--model-name-or-path", help="Teacher model; required for prepare, read from the cache for train"
    )
    parser.add_argument("--revision", help="Pin the teacher model/tokenizer revision when preparing")
    parser.add_argument(
        "--data-file", type=Path, help=f"Local JSONL override; defaults to a pinned revision of {DATASET_REPO_ID}"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("outputs/prepared"), help="Shared teacher cache")
    parser.add_argument("--questions-file", type=Path, help="Optional JSON array of independent training questions")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/cortex"), help="Adapter, tokenizer, and run.json"
    )
    parser.add_argument(
        "--nq", type=int, default=28, help="Generated API questions, in addition to 4 generic questions"
    )
    parser.add_argument(
        "--question-attempts", type=int, default=3, help="Maximum attempts to generate enough distinct API questions"
    )
    parser.add_argument(
        "--answer-attempts",
        type=int,
        default=3,
        help="Maximum attempts per teacher answer; retry incomplete output only",
    )
    parser.add_argument("--generic-context", choices=["source", "none"], default="source")
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate and distill complete reasoning plus final answers (Qwen3)",
    )
    parser.add_argument(
        "--question-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable reasoning while synthesizing questions (default: enabled, as in the research generator)",
    )
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="medium")
    parser.add_argument(
        "--teacher-top-k",
        type=int,
        default=1024,
        help="Cached teacher logits per position for KL, not generation top-k; 0 retains the full vocabulary",
    )
    for kind in ("question", "answer"):
        parser.add_argument(
            f"--{kind}-temperature", type=float, default=0.7, help=f"{kind.title()} sampling temperature"
        )
        parser.add_argument(
            f"--{kind}-do-sample",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=f"Sample {kind}s using the checkpoint's top-p/top-k (default: enabled)",
        )
    parser.add_argument(
        "--max-new-tokens", type=int, default=4096, help="Teacher reasoning plus final-answer generation budget"
    )
    parser.add_argument("--question-max-new-tokens", type=int, default=10240, help="Budget for all nq questions")
    parser.add_argument("--question-max-length", type=int, default=16384, help="Question prompt + output limit")
    parser.add_argument("--answer-max-length", type=int, default=8192, help="Answer prompt + output limit")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=["cortex", "lora"], default="cortex")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--moe-target", choices=["shared", "routed"], default="shared")
    parser.add_argument("--init-weights", choices=["sample", "random"], default="sample")
    parser.add_argument(
        "--adapter-bias",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override Cortex bias presence; by default each projection follows the base FFN",
    )
    parser.add_argument("--adapter-bias-init", choices=["zeros", "uniform"], default="uniform")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Optional optimizer-step limit, overriding epochs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--max-train-length",
        type=int,
        default=3072,
        help="Student prompt plus answer cap; right-truncate training only",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-type", choices=["constant", "linear", "cosine"], default="cosine")
    parser.add_argument("--deepspeed", default="none", help="none, zero0, or a DeepSpeed config path")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cache-states",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Cache hidden states; defaults to whether MSE weight is positive",
    )
    parser.add_argument("--kl-scope", choices=["oi", "o"], default="oi")
    parser.add_argument("--mse-scope", choices=["oi", "o"], default="oi")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--kl-temperature", type=float, default=2.0)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.backend != "cortex" and (args.adapter_bias is not None or args.adapter_bias_init != "uniform"):
        parser.error("Cortex bias options require --backend cortex.")
    if args.stage == "prepare" and not args.model_name_or_path:
        parser.error("--model-name-or-path is required for --stage prepare.")
    if args.question_attempts < 1:
        parser.error("--question-attempts must be positive.")
    if args.answer_attempts < 1:
        parser.error("--answer-attempts must be positive.")
    if args.cache_states is None:
        args.cache_states = args.mse_weight > 0
    if args.stage == "prepare":
        positive_keys = ("nq", "max_new_tokens", "question_max_new_tokens", "question_max_length", "answer_max_length")
    else:
        positive_keys = (
            "k" if args.backend == "cortex" else "r",
            "batch_size",
            "gradient_accumulation_steps",
            "max_train_length",
        )
    for key in positive_keys:
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive.")
    if not math.isfinite(args.epochs) or args.epochs <= 0:
        parser.error("--epochs must be finite and positive.")
    if not 0 <= args.warmup_ratio < 1 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Warmup ratio must be in [0, 1) and weight decay must be finite and nonnegative.")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be -1 or a positive integer.")
    if args.teacher_top_k < 0:
        parser.error("--teacher-top-k must be nonnegative.")
    for key in (
        "kl_temperature",
        "question_temperature",
        "answer_temperature",
        "learning_rate",
        "kl_weight",
        "mse_weight",
    ):
        value = getattr(args, key)
        strictly_positive = key.endswith("temperature") or key == "learning_rate"
        if not math.isfinite(value) or value < 0 or (strictly_positive and value == 0):
            parser.error(f"Invalid --{key.replace('_', '-')}.")
    if args.kl_weight + args.mse_weight == 0:
        parser.error("At least one loss coefficient must be positive.")
    return args


if __name__ == "__main__":
    run(parse_args())
