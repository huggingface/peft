# RLMath benchmark (GRPO)

This directory adds a reinforcement-learning benchmark track for `method_comparison/`.

## Methodology alignment

This setup follows the RL methodology from the Thinking Machines LoRA post:

1. Policy-gradient-style objective with importance weighting.
2. GRPO-like group-relative reward centering (multiple completions per prompt).
3. Optional KL regularization to a reference policy via `beta`.

Reference: https://thinkingmachines.ai/blog/lora/

## Initial datasets

1. `openai/gsm8k` (default, fast iteration)
2. `hendrycks/competition_math` (blog-aligned harder setting)

## Files

1. `default_training_params.json`: baseline GRPO config for fast screening.
2. `experiments/<method>/<name>/adapter_config.json`: adapter definition.
3. Optional `experiments/<method>/<name>/training_params.json`: config overrides.

Starter comparison configs are included for:

1. `experiments/lora/sample-lora-grpo`
2. `experiments/ia3/sample-ia3-grpo`
3. `experiments/adalora/sample-adalora-grpo`
4. `experiments/full-finetuning/sample-full-grpo` (no adapter config)

## Run

```bash
cd method_comparison/RLMath
python -m pip install -r requirements.txt
python run.py -v experiments/lora/sample-lora-grpo
```

For direct adapter comparison, run all starter experiments with the same default config budget:

```bash
python run.py -v experiments/lora/sample-lora-grpo
python run.py -v experiments/ia3/sample-ia3-grpo
python run.py -v experiments/adalora/sample-adalora-grpo
python run.py -v experiments/full-finetuning/sample-full-grpo
```

Results are written to:

1. `results/` when run on `main`
2. `temporary_results/` on other branches
3. `cancelled_results/` if interrupted
