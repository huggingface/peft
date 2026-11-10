# Cortex on MetaMathQA

Use these adapter configurations with the MetaMathQA runner for supervised
causal-LM training.

## Configurations

- `k16-random` uses the tuner's default width and initialization.
- `k16-sample` compares sample initialization at the same width.
- `k64-sample` increases the adapter width with sample initialization.

All three configurations disable adapter biases and use the same runner.

From `method_comparison/MetaMathQA`:

```bash
python run.py experiments/cortex/llama-3.2-3B-k16-random
python run.py experiments/cortex/llama-3.2-3B-k16-sample
python run.py experiments/cortex/llama-3.2-3B-k64-sample
```

The commands inherit the runner's model, training, and evaluation defaults.
See the [MetaMathQA benchmark guide](../../README.md) for setup, command-line
options, and evaluation metrics.
