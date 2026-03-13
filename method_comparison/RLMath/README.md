# RLMath — PEFT adapter comparison with GRPO

Compare PEFT adapters on RL math reasoning using TRL's GRPOTrainer with vLLM
colocate mode. The reward function and dataset setup follow the
[tinker-cookbook](https://github.com/thinkingmachines/tinker-cookbook) math RL
recipes.

## Setup

- **Model**: Qwen/Qwen3.5-4B (hybrid GatedDeltaNet + attention architecture)
- **Dataset**: GSM8K (3 000 train, 50 test for eval)
- **Reward**: `\boxed{}` extraction + sympy-based grading (from tinker-cookbook)
- **Training**: GRPO with vLLM colocate, 100 steps
- **Eval**: Greedy pass@1 on 50-sample test subset, adapter merged into base model via `merge_and_unload()`

## Files

| File | Purpose |
|------|---------|
| `run.py` | Main entrypoint — trains with GRPOTrainer, evaluates pass@1 |
| `data.py` | Dataset loading for GSM8K, MATH, and DeepMath |
| `reward.py` | Reward function: `\boxed{}` extraction + sympy grading |
| `utils.py` | Config dataclass, result persistence, system info |
| `eval_checkpoint.py` | Standalone eval of a saved checkpoint |
| `default_training_params.json` | Shared baseline training config |

## Experiments

Each experiment lives in `experiments/<method>/<name>/` with:
- `adapter_config.json` — PEFT adapter definition (omitted for full fine-tuning)
- `training_params.json` *(optional)* — overrides for `default_training_params.json`

### Available experiments

| Method | Experiment | Config highlights |
|--------|-----------|-------------------|
| LoRA | `lora/qwen3.5-4b-gsm8k` | Rank 32, all linear layers |
| AdaLoRA | `adalora/qwen3.5-4b-gsm8k` | Adaptive rank 64 -> 16, all linear layers |
| IA3 | `ia3/qwen3.5-4b-gsm8k` | k/v/down projections |
| OFT | `oft/qwen3.5-4b-gsm8k` | Rank 8 with COFT, attention layers only |
| Full FT | `full-finetuning/qwen3.5-4b-gsm8k` | No adapter, lower learning rate |

### Results (GSM8K, 50 test samples, pass@1)

| Method | Accuracy | Notes |
|--------|----------|-------|
| Base model | 62.0% | Qwen3.5-4B without fine-tuning |
| LoRA | 86.0% | +24pp over base |
| AdaLoRA | 62.0% | See known issues below |
| IA3 | — | Not yet run |
| OFT | — | Not yet run |
| Full FT | — | Not yet run |

## Usage

```bash
cd method_comparison/RLMath
pip install -r requirements.txt

# Run the LoRA experiment
python run.py -v experiments/lora/qwen3.5-4b-gsm8k

# Run all adapter comparisons
for exp in experiments/*/qwen3.5-4b-gsm8k; do
    python run.py -v "$exp"
done

# Standalone eval of a saved checkpoint
python eval_checkpoint.py checkpoints/lora--qwen3.5-4b-gsm8k/eval_checkpoint
```

## Results output

Results are saved to:
- `results/` — runs on the `main` branch
- `temporary_results/` — runs on other branches
- `cancelled_results/` — interrupted runs

Each result JSON contains training metrics, eval pass@1, and system info.

## Adapter key remapping

Qwen3.5 uses a composite architecture where layers live under
`model.language_model.layers` instead of `model.layers`. GRPOTrainer saves
adapter weights with the `language_model.` prefix, but `PeftModel.from_pretrained`
expects them without it. Both `run.py` and `eval_checkpoint.py` handle this
automatically by remapping keys in the safetensors file and `adapter_config.json`
before loading.

## Known issues

### AdaLoRA + GRPO

AdaLoRA's adaptive rank pruning (`update_and_allocate`) requires gradients on all
LoRA parameters to compute importance scores. In GRPO training, some adapter
parameters may not receive gradients on every step. The current workaround fills
missing gradients with zeros, but this causes importance scores to be zero for
those parameters, leading AdaLoRA to prune all ranks — effectively producing a
base model at eval time. This needs further investigation.

### PEFT AdaLoRA bug fix

`resize_modules_by_rank_pattern` in `src/peft/tuners/adalora/model.py` called
`update_layer` with the old 5-argument signature instead of the current 4-argument
signature (passing a config object). This was fixed on this branch.
