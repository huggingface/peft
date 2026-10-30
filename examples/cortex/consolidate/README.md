# Consolidate SEIS documentation into a Cortex adapter

This example generates training questions about the `seismic_linter` Python package, obtains a frozen teacher's answers and distribution targets with documentation, and trains a student adapter without documentation. PEFT provides the adapter; data generation and distillation are outside the tuner. The same QA text also supports [ordinary cross-entropy training](../finetune/README.md).

The [Cortex SEIS dataset](https://huggingface.co/datasets/MikeDean/cortex-seis-example) contains one source-documentation record and ten target code evaluation tasks. Evaluation questions are not used to generate training questions. The four generic *training* questions are defined separately in `consolidate.py`; they are not evaluation probes.

Preparation and evaluation generation automatically download `data.jsonl` from the dataset revision pinned in `consolidate.py` and reuse the Hugging Face cache. To use a local JSONL instead, pass `--data-file /path/to/data.jsonl` to both `consolidate.py --stage prepare` and `eval.py`. Training from a prepared cache and scoring saved responses do not read the source dataset.

Run the commands below from this directory; see the [environment](../README.md#environment-and-execution) for dependencies. The four stages are explicit:

| Stage | Command | Output |
| --- | --- | --- |
| 1. Prepare | `consolidate.py --stage prepare` | Questions, completed answers, teacher targets, and tokenizer in a new cache directory |
| 2. Train | `consolidate.py --stage train` | Adapter, tokenizer, and `run.json` in a new training directory |
| 3. Generate evaluation responses | `eval.py --adapter ...` | `responses.json`; generated code is not executed |
| 4. Score code | `eval.py --report ... --execute` | `scored.json` with executable-test results |

These are example recipes; their full SEIS task-quality runs have not been validated. Inspect the generated training data before training. Preparation does not automatically resume a partial cache, and training uses the cache directory you specify.

## 1. Prepare teacher data

By default preparation generates `nq=28` API questions plus four generic questions. Both question and answer generation sample at temperature 0.7 and inherit `top_p` / `top_k` from the checkpoint. These sampling filters are separate from the teacher top-k distribution cache.

```bash
CUDA_VISIBLE_DEVICES=0 python consolidate.py --stage prepare \
  --model-name-or-path Qwen/Qwen3-4B-Thinking-2507 \
  --data-dir outputs/qwen-prepared \
  --dtype bfloat16 --nq 28 --generic-context source \
  --question-temperature 0.7 --answer-temperature 0.7 \
  --teacher-top-k 1024 --no-cache-states

CUDA_VISIBLE_DEVICES=0 python consolidate.py --stage prepare \
  --model-name-or-path /path/to/gpt-oss-20b-bf16 \
  --data-dir outputs/oss-prepared \
  --dtype bfloat16 --nq 28 --generic-context source \
  --question-temperature 0.7 --answer-temperature 0.7 \
  --teacher-top-k 1024 --cache-states
```

GPT-OSS requires a materialized BF16 checkpoint. Qwen's recipe below uses KL only and therefore omits hidden-state caching; GPT-OSS also uses MSE. Set `--cache-states` when preparing any cache you plan to use with positive MSE. Otherwise the default follows whether `--mse-weight` is positive.

### Prepared cache

Preparation publishes the cache only after all records are complete:

```text
prepared/
├── question_generation.txt  # present when questions were model-generated
├── train_qa.jsonl
├── manifest.json
├── targets.safetensors
└── tokenizer/
```

Each QA record contains prompt-only `teacher_messages` and `student_messages`, plus the shared generated response:

```json
{
  "id": "seismic_linter-0000",
  "kind": "target",
  "teacher_messages": [
    {"role": "system", "content": "Source documentation for seismic_linter."},
    {"role": "user", "content": "Write code using seismic_linter."}
  ],
  "student_messages": [{"role": "user", "content": "Write code using seismic_linter."}],
  "response": {
    "role": "assistant",
    "reasoning_content": "The teacher's generated reasoning.",
    "content": "The final answer and code."
  },
  "generation": {"finish_reason": "stop", "complete": true}
}
```

`kind` is `target` or `generic`. `student_messages + [response]` forms an ordinary CE conversation. The common message helper maps `reasoning_content` into each model's native template, including GPT-OSS's `thinking` field and fixed automatic date.

The schema-4 manifest records model/revision/dtype, exact teacher/student token IDs, raw generated response IDs, alignment, generation settings, and QA/tensor checksums. The JSONL is portable text; cached tokens and tensors belong to the teacher checkpoint/tokenizer. Hidden-state caches require `state_format="transformers_hidden_states_without_embedding_v1"`; older caches containing pre-final-norm states are incompatible with MSE.

### Generation controls

The question-generation system prompt requests tagged `<question>...</question>` outputs spanning API applications, edge cases, debugging, and integration. Its diversity minima scale with `nq`. `--questions-file` can supply a JSON array of exactly `nq` independently prepared questions instead.

Question generation allows up to three attempts (`--question-attempts 3`). Each complete final response is parsed independently. Text outside the tags, empty questions, and questions missing the package name are ignored; whitespace-normalized duplicates are removed. The first attempt with at least `nq` distinct valid questions supplies its first `nq` questions. If no attempt supplies a complete set, preparation pools valid questions from completed attempts and samples `nq` without replacement using `--seed`. An insufficient pool is an error. Truncated responses and reasoning never contribute candidates. Retries do not reset the generation RNG, and model execution failures propagate immediately.

Question generation allows 10,240 output tokens and 16,384 prompt-plus-output tokens by default. Answers allow 4,096 output tokens and 8,192 total tokens. Override these with `--question-max-new-tokens`, `--question-max-length`, `--max-new-tokens`, and `--answer-max-length`. These are generation limits, separate from training length. Both stages enable thinking; Qwen3-4B-Thinking-2507 does not support disabling it. GPT-OSS uses `--reasoning-effort low|medium|high`. Supported switchable models accept `--no-question-thinking` and `--no-thinking`. `--no-question-do-sample` and `--no-answer-do-sample` select greedy generation separately.

Each teacher answer allows up to three attempts (`--answer-attempts 3`). An incomplete answer is regenerated for the same question, with the same prompt and sampling settings and without resetting the RNG. Complete answers from earlier questions in the current preparation are retained. Preparation never accepts a generation-truncated or unfinished teacher response; the training length cap remains separate. Model execution errors propagate immediately.

For answers that need a longer reasoning budget, explicitly pass `--max-new-tokens 8192 --answer-max-length 12288`. This changes teacher-data generation, not the training length; it can increase preparation memory and cache size. When comparing evaluation runs, set the evaluation budget explicitly, for example `eval.py ... --max-new-tokens 4096 --max-eval-length 8192`, rather than inheriting a larger teacher budget.

The four generic training questions ask for: an introduction to Elon Musk, a PyTorch residual connection, the planet with density lower than water, and winter travel in Shanghai. Their teacher receives source documentation by default; `--generic-context none` removes it for those four targets. The student never receives documentation. Context-conditioned generic answers need not equal the teacher's context-free behavior.

### Retry diagnostics

Files next to the requested cache directory remain available for inspection:

- `<cache-name>.questions.json` contains the selected target questions, saved before answer generation. It survives later answer-generation failure. Use it with `--questions-file` in a new preparation to reuse questions while regenerating answers and targets.
- `<cache-name>.question_generation.attempt-N.txt` retains each raw question attempt. On failure, `<cache-name>.question_generation.txt` also contains the latest attempt; success removes this latest-attempt alias.
- Inside a completed cache, `question_generation.txt` contains the selected raw attempt, or all labeled attempts when pooled fallback was used. The sibling questions JSON records the actual selected questions.
- `<cache-name>.incomplete_answer.0001.attempt-1.json` retains an incomplete answer attempt, using a zero-based question index. The sibling `<cache-name>.incomplete_answer.json` aliases the latest failure and is removed after successful preparation.

These checks require completed responses with nonempty final answers and valid termination. They do not establish API correctness. Inspect generated QA for quality and semantic overlap with evaluation tasks before training.

## 2. Train the adapter

Cortex follows the native bias layout by default: GPT-OSS projections get adapter biases, while bias-free Qwen FFNs do not. Use `--adapter-bias` or `--no-adapter-bias` to override this. Random initialization uses uniform gate/up biases by default (`--adapter-bias-init zeros` selects zeros). Sample initialization copies available native biases and zeros missing ones. Both strategies zero down weights and biases.

The Qwen recipe uses sample-initialized Cortex channels, KL only, and 100 epochs:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1 consolidate.py --stage train \
  --data-dir outputs/qwen-prepared --output-dir outputs/qwen-cortex-kd \
  --backend cortex --k 64 --init-weights sample \
  --batch-size 8 --epochs 100 --max-train-length 3072 \
  --deepspeed zero0 --gradient-checkpointing \
  --learning-rate 5e-4 --lr-scheduler-type cosine --warmup-ratio 0.1 \
  --weight-decay 0 --gradient-accumulation-steps 1 \
  --kl-temperature 2 --kl-weight 1 --mse-weight 0 --kl-scope oi --mse-scope oi
```

The GPT-OSS recipe uses a random-initialized shared branch, KL plus MSE, and 300 epochs:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1 consolidate.py --stage train \
  --data-dir outputs/oss-prepared --output-dir outputs/oss-cortex-kd \
  --backend cortex --k 64 --init-weights random --moe-target shared \
  --batch-size 4 --epochs 300 --max-train-length 2048 \
  --deepspeed zero0 --gradient-checkpointing \
  --learning-rate 5e-4 --lr-scheduler-type cosine --warmup-ratio 0.1 \
  --weight-decay 0 --gradient-accumulation-steps 1 \
  --kl-temperature 2 --kl-weight 1 --mse-weight 1 --kl-scope oi --mse-scope oi
```

Training takes model, revision, and dtype from the manifest; these runs therefore load BF16 bases. The base stays frozen, and only the student is resident on the accelerator. The backend controls mixed-precision execution and optimizer state. `--deepspeed none` selects ordinary Trainer; a ZeRO-0 JSON config is also accepted. Activation checkpointing is non-reentrant. For a shorter workflow check, prepare `--nq 1` in a new directory and train with `--epochs 1 --batch-size 1`; five records give five optimizer steps with accumulation one. Use a tiny checkpoint and `--device cpu --dtype float32` during preparation for CPU checks, then train with `--device cpu --deepspeed none`.

For a LoRA distillation comparison, use `--backend lora --r 16 --lora-alpha 32` instead of the Cortex options, keeping other settings explicit. Ordinary Cortex CE training is shown in [finetune/](../finetune/README.md); convert prepared QA to its standard messages format. Dense LoRA targets MLP projections; GPT-OSS LoRA adapts each expert's packed projections, whereas shared Cortex allocates a single shared extension. Match actual parameter counts and training budgets when comparing them.

Training saves adapter weights, configuration, tokenizer, and `run.json` directly in `--output-dir`. The record contains settings, QA and manifest checksums, loss history, optimizer steps, trainable parameter count, and preprocessing statistics. Training saves its output without reloading it. Use the evaluation stage below to load the trained adapter.

### Distillation objective and token alignment

The loss is `kl_weight * KL + mse_weight * MSE`, with no hard-label CE term. KL uses teacher top-k logits saved before temperature scaling. It normalizes teacher probabilities over the cached token support, normalizes student probabilities over the full vocabulary, and multiplies KL by temperature squared. The recipes use `--kl-temperature 2` and cache the top 1,024 teacher logits per position. `--teacher-top-k 0` caches the full teacher vocabulary.

`--kl-scope` and `--mse-scope` independently choose `oi` (default) or `o`:

| Scope | Tokens included |
| --- | --- |
| `oi` | Aligned user-question content, following assistant prompt, and generated response. Source documentation and earlier chat headers are excluded. |
| `o` | Generated response, including reasoning and final answer. |

Teacher and student prompts are tokenized separately. Their question/assistant suffix must match, and the original generated response IDs are appended unchanged. KL compares causal next-token predictions and excludes each record's final logit; output-only KL starts immediately before the first response token. MSE compares same-position `outputs.hidden_states[1:]`, including the final normalized decoder state and excluding the embedding output. The teacher cache and student capture use the same layers. MSE averages over hidden dimensions, valid tokens, and layers. Both losses are calculated in FP32; hidden-state caches retain their output dtype, and top-k logits are stored as FP32.

`--max-train-length` right-truncates the entire student sequence and slices cached targets to the retained positions. Padding is excluded. A limit that leaves no response tokens is rejected; the run records truncation counts. Generation can produce a complete answer longer than the training budget, so inspect whether reasoning consumes the retained prefix.

## 3. Generate evaluation responses

Use `eval.py` with the consolidation output root containing the adapter, tokenizer, and `run.json`. Generation and scoring both require a new `--output-file`:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --data-dir outputs/qwen-prepared \
  --adapter outputs/qwen-cortex-kd \
  --output-file outputs/qwen-cortex-kd/responses.json
```

Replace the adapter/output paths for other Cortex or LoRA consolidation runs. Evaluation uses the adapter's actual base checkpoint and tokenizer and checks the training-QA and manifest checksums. Prepared metadata and QA are required, but `targets.safetensors` is not read. To evaluate only the unadapted model, omit `--adapter` and pass `--modes base`; the prepared metadata and tokenizer are still required. Without explicit `--modes`, omitting the adapter evaluates both `base` and `context_teacher`. Scoring an existing report needs neither model weights nor the cache. The ordinary finetuning example saves standard PEFT artifacts without the experiment record required by this evaluator.

The report compares the evaluated base model without context (`base`), that same base with documentation (`context_teacher`), and the adapted model without documentation (`adapted`). The report records both the evaluated model and the preparation teacher identity. Exact training-question overlaps are flagged and excluded from aggregates; inspect semantic overlap manually.

Generation is the default and does not execute generated programs. Evaluation samples at temperature 0.7 and inherits top-p / top-k from the checkpoint. The report records the effective sampling settings. Each task receives one response per evaluation mode; this is not a multi-sample pass@k evaluation. Reasoning is stored separately, and code is extracted only from complete final answers. `--max-new-tokens` and `--max-eval-length` can adjust the evaluation budget.

## 4. Score generated code

Run scoring in an execution environment you provide. This step needs the saved response report and the package under test, but no model weights, prepared cache, or GPU:

```bash
python -m pip install seismic-linter==0.2.0
python eval.py \
  --report outputs/qwen-cortex-kd/responses.json \
  --execute \
  --output-file outputs/qwen-cortex-kd/scored.json
```

`--report` and `--execute` are required together. This uses the current Python interpreter, a temporary directory, and a per-task timeout (default 30 seconds). **Those mechanisms do not provide isolation or restrict filesystem/network access.** Docker is not required; supply an appropriate execution environment and inspect the generated programs before scoring.

Code scores are aggregated by split and exclude exact training-question overlaps. The default dataset produces a `target` summary only. At least one held-out target task is required. Tests use multiple inputs and check empty/multiple violation lists, split boundaries, repeated timestamps, and the monotonicity decorator. Report tasks passed out of ten alongside parameter count and training budget; one generation per task is not a repeated-sampling estimate.

## Optional retention probes

To evaluate retention, supply a custom JSONL with `--data-file`. Keep the same context record and target tasks, and add your own evaluation records with `type="eval"`, unique `id` values, and `split="retain"`. Text probes need a `question` and reference `answers`; code probes use `format="python"` and executable `tests`. Keep code and text probes in separate splits. An absent retain split is omitted from the report. If a supplied split contains only exact training-question overlaps, evaluation raises an error instead of reporting a misleading score.

Text-probe metrics are calculated during the generation stage and report exact match, containment, and reference-answer NLL. `--execute` is only needed for Python probes. NLL does not capture hidden states. For Qwen/GPT-OSS references without reasoning, it uses a native empty, closed reasoning turn, labeled `reference_nll_prompt="empty_reasoning"`; this does not disable thinking generation. Choose probes appropriate to the capabilities you want to preserve, and inspect responses because containment can accept a negated answer.

## Citation

See the [Cortex citation](../README.md#citation).
