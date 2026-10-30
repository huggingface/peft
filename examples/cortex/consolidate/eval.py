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

"""Generate held-out SEIS responses; explicitly score saved programs in a separate environment."""

import argparse
import ast
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from importlib import metadata
from pathlib import Path

import torch
import torch.nn.functional as F
from consolidate import (
    DATASET_REPO_ID,
    file_sha256,
    generate_answer,
    generation_status,
    load_base,
    load_prepared_metadata,
    load_source,
    parse_response,
    prompt_ids,
    prompt_messages,
    read_json,
    read_jsonl,
    resolve_data_file,
    resolve_device,
    write_json,
)
from transformers import AutoTokenizer, set_seed

from peft import PeftConfig, PeftModel


# This example is also runnable directly from this subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chat_utils import final_answer_prefix


# Match the research CODE evaluation; top-p/top-k remain checkpoint defaults.
EVAL_SAMPLING = {"do_sample": True, "temperature": 0.7}


def normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


@torch.no_grad()
def generate_response(
    model, tokenizer, question, context, max_new_tokens, max_length, *, thinking=False, reasoning_effort="medium"
):
    prefix = prompt_ids(
        tokenizer,
        question,
        context,
        model_type=model.config.model_type,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
    )
    generated = generate_answer(
        model,
        tokenizer,
        prefix,
        max_new_tokens,
        max_length,
        allow_empty=True,
        thinking=thinking,
        sampling_kwargs=EVAL_SAMPLING,
    )
    response, complete = parse_response(
        tokenizer,
        generated,
        model.config.model_type,
        prefix_ids=prefix,
        enable_thinking=thinking,
        eos_token_ids=model.generation_config.eos_token_id,
    )
    return {
        "response": response["content"],
        "reasoning_content": response.get("reasoning_content", ""),
        "raw_response": tokenizer.decode(generated, skip_special_tokens=False),
        "generation": generation_status(
            tokenizer,
            generated,
            max_new_tokens,
            complete=complete,
            eos_token_ids=model.generation_config.eos_token_id,
        ),
    }


@torch.no_grad()
def score_probe(
    model,
    tokenizer,
    question,
    answers,
    context,
    max_new_tokens,
    max_length,
    *,
    thinking=False,
    reasoning_effort="medium",
):
    result = generate_response(
        model,
        tokenizer,
        question,
        context,
        max_new_tokens,
        max_length,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
    )
    prediction = normalized(result["response"])
    aliases = [normalized(a) for a in answers]
    # The reference has no reasoning trace. Close an explicitly empty thinking
    # block before scoring its answer, including for thinking-only checkpoints.
    model_type = model.config.model_type
    prefix = final_answer_prefix(
        tokenizer,
        prompt_messages(question, context),
        model_type,
        reasoning_effort=reasoning_effort,
    )
    answer = tokenizer(answers[0], add_special_tokens=False)["input_ids"]
    if not answer or len(prefix) + len(answer) > max_length:
        raise ValueError("Reference answer is empty or exceeds the configured sequence length.")
    ids = torch.tensor([prefix + answer], device=model.device)
    logits = model(
        input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, output_hidden_states=False
    ).logits
    answer_logits = logits[0, len(prefix) - 1 : len(prefix) + len(answer) - 1]
    nll = F.cross_entropy(answer_logits.float(), torch.tensor(answer, device=model.device)).item()
    complete = result["generation"]["complete"]
    result.update(
        {
            "exact_match": complete and prediction in aliases,
            "answer_contains": complete and any(f" {a} " in f" {prediction} " for a in aliases),
            "reference_answer_nll": nll,
            "reference_nll_prompt": (
                "empty_reasoning" if model_type in {"qwen3", "gpt_oss"} else "native_generation_prompt"
            ),
        }
    )
    return result


def probe_seed(seed, probe_id, mode):
    """Pair each question/mode across adapters without response-length RNG drift."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{probe_id}:{mode}".encode()).digest()[:4], "big")


def evaluate(args):
    if args.output_file.exists():
        raise ValueError("Choose a new --output-file for evaluation.")
    manifest, training_qa = load_prepared_metadata(args.data_dir)
    data_file = resolve_data_file(args.data_file)
    source = load_source(data_file)
    if source["context"] != manifest["context"] or source["package"] != manifest["package"]:
        raise ValueError("Evaluation documentation differs from the prepared training context.")
    probes = [row for row in read_jsonl(data_file) if row.get("type") == "eval"]
    manifest_sha256 = file_sha256(args.data_dir / "manifest.json")
    settings = manifest
    model_name_or_path = manifest["model_name_or_path"]
    revision = manifest["revision"]
    tokenizer_path = args.data_dir / "tokenizer"
    if args.adapter:
        run = read_json(args.adapter / "run.json")
        if run["objective"] not in {"ce", "distillation"}:
            raise ValueError("Adapter run.json must describe a CE or distillation run.")
        if run["data"]["training_file_sha256"] != manifest["train_qa_sha256"]:
            raise ValueError("Adapter and evaluation data refer to different training messages.")
        if run["objective"] == "distillation" and run["data"]["manifest_sha256"] != manifest_sha256:
            raise ValueError("Adapter and evaluation data refer to different prepared contexts.")
        adapter_config = PeftConfig.from_pretrained(args.adapter)
        model_name_or_path = adapter_config.base_model_name_or_path
        revision = adapter_config.revision
        if not model_name_or_path:
            raise ValueError("The adapter configuration must identify its base model.")
        settings = run["settings"]
        tokenizer_path = args.adapter
    evaluation_seed = getattr(args, "seed", None)
    set_seed(evaluation_seed if evaluation_seed is not None else settings.get("seed", manifest["seed"]))
    mode_names = getattr(args, "modes", None) or (
        ["base", "context_teacher", "adapted"] if args.adapter else ["base", "context_teacher"]
    )
    if "adapted" in mode_names and not args.adapter:
        raise ValueError("The adapted mode requires --adapter.")
    device = resolve_device(args.device, settings["dtype"])
    model = load_base(model_name_or_path, revision, settings["dtype"], device)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    training_questions = {normalized(row["student_messages"][-1]["content"]) for row in training_qa}
    max_new_tokens = args.max_new_tokens or manifest["answer_max_new_tokens"]
    max_length = args.max_eval_length or manifest["max_length"]
    thinking = args.thinking if args.thinking is not None else settings.get("thinking")
    if thinking is None:
        thinking = model.config.model_type in {"qwen3", "gpt_oss"}
    if model.config.model_type == "gpt_oss" and not thinking:
        raise ValueError("GPT-OSS has no no-thinking switch; use --reasoning-effort low instead.")
    generation_options = {
        "thinking": thinking,
        "reasoning_effort": args.reasoning_effort or settings.get("reasoning_effort", "medium"),
    }
    results = []
    for probe in probes:
        if probe["split"] not in {"target", "retain"} or (
            probe.get("format") != "python" and not probe.get("answers")
        ):
            raise ValueError(
                "Each probe needs split=target/retain and reference answers, or format=python for code tasks."
            )
        result = {**probe, "overlaps_training_question": normalized(probe["question"]) in training_questions}
        modes = [(mode, manifest["context"] if mode == "context_teacher" else "") for mode in mode_names]
        for mode, context in modes:
            if evaluation_seed is not None:
                set_seed(probe_seed(evaluation_seed, probe["id"], mode))
            disabled = model.disable_adapter() if args.adapter and mode != "adapted" else nullcontext()
            with disabled:
                if probe.get("format") == "python":
                    result[mode] = generate_response(
                        model,
                        tokenizer,
                        probe["question"],
                        context,
                        max_new_tokens,
                        max_length,
                        **generation_options,
                    )
                else:
                    result[mode] = score_probe(
                        model,
                        tokenizer,
                        probe["question"],
                        probe["answers"],
                        context,
                        max_new_tokens,
                        max_length,
                        **generation_options,
                    )
        results.append(result)
    summary = {}
    for split in ("target", "retain"):
        split_results = [r for r in results if r["split"] == split]
        if split == "retain" and not split_results:
            continue
        held_out = [r for r in split_results if not r["overlaps_training_question"]]
        if not held_out:
            raise ValueError(f"No held-out {split} probes remain after excluding exact training-question overlaps.")
        summary[split] = {"num_probes": len(held_out)}
        if all(r.get("format") == "python" for r in held_out):
            summary[split]["scoring"] = (
                "Run eval.py --report REPORT --execute in an isolated environment to score the code tests."
            )
            continue
        if any(r.get("format") == "python" for r in held_out):
            raise ValueError("Keep code and text probes in separate splits.")
        for mode in mode_names:
            summary[split][mode] = {
                metric: sum(r[mode][metric] for r in held_out) / len(held_out)
                for metric in ("exact_match", "answer_contains", "reference_answer_nll")
            }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_file,
        {
            "summary": summary,
            "probes": results,
            "manifest_sha256": manifest_sha256,
            "data_file_sha256": file_sha256(data_file),
            "evaluated_model": {
                "model_name_or_path": model_name_or_path,
                "revision": revision,
                "model_type": model.config.model_type,
                "dtype": settings["dtype"],
                "tokenizer": str(tokenizer_path),
                "adapter": str(args.adapter) if args.adapter else None,
            },
            "prepared_teacher": {
                "model_name_or_path": manifest["model_name_or_path"],
                "revision": manifest["revision"],
            },
            "context_teacher_mode": "The evaluated base model with the source context and without the adapter.",
            "generation": {
                **generation_options,
                **EVAL_SAMPLING,
                "seed": evaluation_seed,
                "seed_policy": "per_probe_mode_v1" if evaluation_seed is not None else "sequential_training_seed",
                "modes": mode_names,
                "top_p": model.generation_config.top_p,
                "top_k": model.generation_config.top_k,
                "max_new_tokens": max_new_tokens,
                "max_length": max_length,
            },
        },
    )
    print(json.dumps(summary, indent=2))


def extract_code(response):
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", response, flags=re.DOTALL)
    if len(blocks) != 1:
        raise ValueError("Expected exactly one fenced Python code block.")
    code = blocks[0].strip()
    if not code:
        raise ValueError("The code block is empty.")
    ast.parse(code)
    return code


def check_execution_environment():
    """Check the scoring dependency before loading a model or executing programs."""
    install = "Install the evaluation dependency with: python -m pip install seismic-linter==0.2.0"
    try:
        version = metadata.version("seismic-linter")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(install) from error
    if version != "0.2.0":
        raise RuntimeError(f"Expected seismic-linter==0.2.0, found {version}. {install}")
    try:
        importlib.import_module("seismic_linter")
    except ImportError as error:
        raise RuntimeError(f"Cannot import seismic_linter in the evaluation environment. {install}") from error


def run_tests(response, tests, timeout):
    try:
        code = extract_code(response)
    except (ValueError, SyntaxError) as error:
        return {"passed": False, "timed_out": False, "error": str(error)}
    with tempfile.TemporaryDirectory(prefix="cortex-seis-") as directory:
        folder = Path(directory)
        for filename, content in (("submission.py", code), ("tests.py", tests)):
            path = folder / filename
            path.write_text(content + "\n", encoding="utf-8")
        command = [sys.executable, "-B", "tests.py"]
        env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        timed_out = False
        with tempfile.TemporaryFile() as output:
            try:
                result = subprocess.run(
                    command, cwd=folder, env=env, stdout=output, stderr=output, timeout=timeout, check=False
                )
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                timed_out, returncode = True, None
            except OSError as error:
                raise RuntimeError(
                    f"Python subprocess startup failed; this is not a model failure: {error}"
                ) from error
            output.seek(0, 2)
            output.seek(max(0, output.tell() - 4096))
            diagnostic = output.read().decode("utf-8", errors="replace")
        # An early exit during submission import must not count as successful tests.
        completed_tests = re.search(r"Ran [1-9]\d* tests? in ", diagnostic) is not None
        return {
            "passed": returncode == 0 and completed_tests and not timed_out,
            "timed_out": timed_out,
            "returncode": returncode,
            "diagnostic": diagnostic,
        }


def score_report(report, timeout):
    probes = [p for p in report["probes"] if p.get("format") == "python"]
    if not probes:
        raise ValueError("The report contains no Python tasks.")
    for probe in probes:
        if probe.get("split") not in {"target", "retain"}:
            raise ValueError(f"Code task {probe.get('id', '<unknown>')} needs split=target/retain.")
        if not probe.get("tests"):
            raise ValueError(f"Missing tests for {probe['id']}.")
    for probe in probes:
        for mode in ("base", "context_teacher", "adapted"):
            if mode in probe:
                if not probe[mode].get("generation", {}).get("complete", True):
                    probe[mode]["execution"] = {
                        "passed": False,
                        "timed_out": False,
                        "error": "Incomplete model response; code was not executed.",
                    }
                else:
                    probe[mode]["execution"] = run_tests(probe[mode]["response"], probe["tests"], timeout)
    held_out = [p for p in probes if not p["overlaps_training_question"]]
    if not held_out:
        raise ValueError("No held-out code tasks remain.")
    for split in ("target", "retain"):
        tasks = [probe for probe in held_out if probe["split"] == split]
        if not tasks:
            continue
        summary = report["summary"].setdefault(split, {})
        summary.pop("scoring", None)
        summary["num_tasks"] = len(tasks)
        for mode in ("base", "context_teacher", "adapted"):
            if all(mode in probe for probe in tasks):
                passed = sum(probe[mode]["execution"]["passed"] for probe in tasks)
                summary.setdefault(mode, {}).update({"passed_tasks": passed, "pass_rate": passed / len(tasks)})
    report["execution"] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "timeout_seconds": timeout,
        "package": "seismic-linter==0.2.0",
        "backend": "local_subprocess",
        "isolation": "external; not provided by this script",
    }
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("outputs/prepared"))
    parser.add_argument(
        "--data-file", type=Path, help=f"Local JSONL override; defaults to a pinned revision of {DATASET_REPO_ID}"
    )
    parser.add_argument("--adapter", type=Path, help="Training output directory containing the adapter and run.json")
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--seed", type=int, help="Seed each question/mode independently for paired repeated evaluation"
    )
    parser.add_argument(
        "--modes", nargs="+", choices=["base", "context_teacher", "adapted"], help="Default: all available modes"
    )
    parser.add_argument("--max-new-tokens", type=int, help="Override the prepared answer generation budget")
    parser.add_argument("--max-eval-length", type=int, help="Override the prepared prompt-plus-answer token limit")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    parser.add_argument(
        "--report", type=Path, help="Score saved responses without loading a model; requires --execute"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute saved programs locally; run this step only in an externally isolated environment",
    )
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args(argv)
    if args.modes and (
        len(set(args.modes)) != len(args.modes) or ("adapted" in args.modes and not args.adapter and not args.report)
    ):
        parser.error("Use distinct modes and provide --adapter for adapted generation.")
    limits = (args.timeout, args.max_new_tokens, args.max_eval_length)
    if args.output_file.exists() or any(value is not None and value < 1 for value in limits):
        parser.error("Use a new output file and positive generation/timeout limits.")
    if bool(args.report) != args.execute:
        parser.error("Code execution requires both --report and --execute; generation never executes code.")
    return args


def main():
    args = parse_args()
    if args.report:
        check_execution_environment()
        report = score_report(read_json(args.report), args.timeout)
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output_file, report)
        print(json.dumps(report["summary"], indent=2))
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
