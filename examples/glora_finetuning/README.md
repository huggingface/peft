# GLoRA causal language modeling fine-tuning

This example demonstrates how to fine-tune a causal language model with [GLoRA](https://arxiv.org/abs/2306.07967) adapters using the Alpaca-style instruction data from `yahma/alpaca-cleaned`. GLoRA generalizes LoRA by introducing configurable paths for weight and bias corrections: `W_eff = W0 + W0 * A + B` and `b_eff = b0 + b0 * D + E + W0 @ C`.

## Running the script

```bash
python examples/glora_finetuning/glora_finetuning.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --data_path yahma/alpaca-cleaned \
  --output_dir glora-alpaca \
  --glora_r 8 \
  --config_A_B lora \
  --config_C lora \
  --config_D_E constant \
  --learning_rate 3e-4 \
  --num_epochs 3
```

Each path (`config_A_B`, `config_C`, `config_D_E`) can be set to different parameterization modes (`lora`, `vector`, `constant`, `none`) to trade off expressiveness against parameter count. The default configuration uses `lora` for A/B and C, and `constant` for D/E.
