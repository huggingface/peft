# Fine-tuning dense and MoE models with Cortex

`finetune.py` uses Transformers `Trainer` for cross-entropy training and saves a
Cortex adapter and tokenizer. It requires no teacher model or distillation cache.
See the [environment](../README.md#environment-and-execution) for dependencies.

## Training data

Supply a local JSONL file with one of these formats:

```json
{"messages": [{"role": "user", "content": "What is a temporal split?"}, {"role": "assistant", "content": "Training timestamps precede test timestamps."}]}
```

```json
{"text": "A temporal split places earlier observations in training and later observations in testing."}
```

For `messages`, only the final assistant turn is supervised; earlier messages and
padding receive label `-100`. Use an instruction model with a chat template and
end each conversation with a user question followed by an assistant answer.
For `text`, every non-padding token is supervised. Use one format per file.

An assistant message may contain `reasoning_content` for Qwen3 or GPT-OSS. The
shared message helper maps it to the model's native format and includes it in
training by default. `--no-train-reasoning` removes it from inputs and labels.
GPT-OSS rendering uses the same fixed date as the consolidation example.

`--max-train-length` right-truncates the complete rendered sequence. The script
reports truncation counts and rejects chat samples with no remaining assistant targets.
Allow enough room for both reasoning and the final answer when using reasoning data.

## Dense model

Run from this directory with one visible GPU. Save your training records as
`train.jsonl` and choose a new, nonexistent output directory for each run:

```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --model-name-or-path Qwen/Qwen3-0.6B \
  --train-file train.jsonl \
  --output-dir outputs/qwen-cortex \
  --k 64 --init-weights sample \
  --dtype bfloat16 --max-train-length 2048 \
  --batch-size 1 --gradient-accumulation-steps 4 --epochs 1
```

The script constructs `CortexConfig`, wraps the model with `get_peft_model()`, and
uses `Trainer.train()`. `--learning-rate`, `--gradient-checkpointing`, and
`--max-steps` configure ordinary training. For a CPU smoke run with a small local
model, use `--use-cpu --dtype float32 --max-steps 2`.

## MoE model

The same entry point supports MoE models. For a materialized BF16 GPT-OSS checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --model-name-or-path /path/to/gpt-oss-20b-bf16 \
  --train-file train.jsonl \
  --output-dir outputs/oss-cortex \
  --k 64 --init-weights random \
  --moe-target shared --public-weighting unweighted \
  --dtype bfloat16 --max-train-length 4096 \
  --batch-size 1 --gradient-accumulation-steps 4 --epochs 1
```

Choose the adaptation scope explicitly:

| Option | Scope |
| --- | --- |
| `--moe-target shared` | Native shared FFNs when present; otherwise a public extension shared by routed experts. |
| `--moe-target routed` | Separate extensions for routed experts. |
| `--moe-target shared routed` | Both scopes. |

`--public-weighting routing` weights a public extension by the selected routing
mass; `unweighted` applies it once per token. This option does not alter native
shared FFNs or the routed extensions. Biases follow the base projections by
default. Use `--adapter-bias` / `--no-adapter-bias` to override their presence and
`--adapter-bias-init uniform` / `zeros` to choose gate/up bias values during
random initialization. Sample initialization copies available native biases
and zeros missing ones. Both strategies zero down weights and biases.

This example does not convert MXFP4 weights. Load an already materialized BF16
GPT-OSS checkpoint. `use_kernels=False` keeps parent forwards available to the
adapter; `--experts-implementation eager` can select reference native experts.
The script uses one device; choose a GPU with enough memory for the frozen base,
activations, and adapter training state. The commands illustrate usage, not tuned
task-quality or memory benchmarks.

## Save, reload, and merge

The script saves the adapter and tokenizer to `--output-dir` and then exits.
In a separate process, reload them with the normal PEFT API. This snippet uses
the dense example's output directory and loads the model on CPU; move it to an
appropriate device before GPU inference:

```python
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_dir = "outputs/qwen-cortex"
config = PeftConfig.from_pretrained(adapter_dir)
base = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    revision=config.revision,
    dtype=torch.bfloat16,
    use_kernels=False,
)
model = PeftModel.from_pretrained(base, adapter_dir).eval()
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
```

Adding `--merge-output outputs/native` to the training command also exports a
native model before the script exits. Choose a new export directory. Export
expands compatible FFNs; keep the adapter checkpoint for adapter switching.
When planning training on a model with unnormalized routing, choose
`--public-weighting routing` if its public branch must be mergeable. Changing
this option after training changes the function. Adapter biases cannot merge
into a bias-free native FFN. Reload exported MoE models with
`experts_implementation="eager"`. See the [API reference](../../../docs/source/package_reference/cortex.md#saving-and-structural-merging)
for the full merge contract.

## Reusing consolidation questions

Convert completed prepared QA to the standard `messages` format before training:

```python
import json
from pathlib import Path

source = Path("../consolidate/outputs/qwen-prepared/train_qa.jsonl")
with source.open(encoding="utf-8") as inputs, Path("train.jsonl").open("w", encoding="utf-8") as output:
    for line in inputs:
        row = json.loads(line)
        generation = row.get("generation") or {}
        if generation.get("complete") is not True or generation.get("finish_reason") == "length":
            raise ValueError("Regenerate incomplete answers before fine-tuning.")
        messages = row["student_messages"] + [row["response"]]
        output.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
```

Teacher messages and source documentation are excluded. The ordinary fine-tuning
example saves standard PEFT artifacts; the consolidation-specific report/evaluator
workflow belongs to [consolidate/](../consolidate/README.md).

## Custom FFN classification

For a custom PyTorch model, use the PEFT API directly instead of the causal-LM
command-line script. This CPU example adapts a supported non-gated FFN and trains
a classification head with ordinary cross-entropy. Random data demonstrates the
training interface; it is not a classification benchmark.

```python
import torch
from torch import nn
from torch.nn import functional as F

from peft import CortexConfig, get_peft_model


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(16, 32, bias=False)
        self.act_fn = nn.GELU()
        self.down_proj = nn.Linear(32, 16, bias=False)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = FFN()
        self.head = nn.Linear(16, 3)

    def forward(self, features):
        return self.head(self.ffn(features))


model = get_peft_model(
    Classifier(),
    CortexConfig(
        k=4,
        target_modules=["ffn"],
        modules_to_save=["head"],
    ),
)
optimizer = torch.optim.AdamW(
    [parameter for parameter in model.parameters() if parameter.requires_grad],
    lr=1e-3,
)
features = torch.randn(16, 16)
labels = torch.randint(0, 3, (16,))
model.train()
for _ in range(5):
    optimizer.zero_grad()
    loss = F.cross_entropy(model(features), labels)
    loss.backward()
    optimizer.step()
print(f"Classification loss: {loss.item():.4f}")
```

Leaving `task_type` unset selects PEFT's generic wrapper.
`modules_to_save=["head"]` trains and saves the classification head along with
the adapter; the original FFN parameters stay frozen. Explicit targets must
implement the supported FFN arithmetic and activation contract described in the
[API reference](../../../docs/source/package_reference/cortex.md#target-selection).
This does not imply automatic support for arbitrary vision, diffusion, or other
model layouts.

## Citation

See the [Cortex citation](../README.md#citation).
