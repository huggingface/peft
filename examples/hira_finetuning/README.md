# HiRA causal language modeling fine-tuning

This example demonstrates how to fine-tune a causal language model with [HiRA](https://openreview.net/pdf?id=TwJrTz9cRS) adapters using the Alpaca-style instruction data from `yahma/alpaca-cleaned`. The script mirrors the common LoRA flow and shows how to configure HiRA-specific parameters such as the Hadamard modulation rank (`r`) and dropout.

## Running the script

```bash
python examples/hira_finetuning/hira_finetuning.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --data_path yahma/alpaca-cleaned \
  --output_dir hira-alpaca \
  --hira_r 16 \
  --hira_dropout 0.05 \
  --learning_rate 3e-4 \
  --num_epochs 3
```

The default target modules cover the attention projections and MLP blocks typically present in decoder-style architectures. Adjust them if your base model uses different module names.
