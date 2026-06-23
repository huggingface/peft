# LoRA vs ShadowPEFT benchmark

A small, self-contained script that trains [LoRA](https://huggingface.co/docs/peft/package_reference/lora) and
[ShadowPEFT](https://huggingface.co/docs/peft/package_reference/shadow) on the same task with the same training budget
and prints a side-by-side comparison (trainable parameters, training time, and the task metric).

Three task types are supported, each with one clear metric:

| `--task` | model class                          | metric                              |
| -------- | ------------------------------------ | ----------------------------------- |
| `cls`    | `AutoModelForSequenceClassification` | accuracy                            |
| `clm`    | `AutoModelForCausalLM`               | loss / perplexity                   |
| `gsm8k`  | `AutoModelForCausalLM`               | exact-match accuracy (generated)    |

Both methods are built from a fresh copy of the same base model via `get_peft_model`, trained with the Hugging Face
`Trainer`, then evaluated identically.

The `gsm8k` task follows the setup in ShadowPEFT's `experiment/run_shadow_peft.py`: it loads `openai/gsm8k`, formats
each example with a chat template, supervises only the answer tokens (the prompt is masked with `-100`), and evaluates
by **generating** an answer and extracting the final number (`#### N`) for exact-match accuracy. Eval skips the
teacher-forced loss pass by default (use `--gsm8k_eval_loss` to enable it). With multiple processes, generation eval
runs on **rank 0 only** by default (use `--gsm8k_distributed_eval` to eval on every rank). Generation is batched,
uses **KV cache** for LoRA (`use_cache=False` for ShadowPEFT), and uses a single FSDP `summon_full_params` with
`synced_gpus=False`. Per-batch timing is printed on rank 0. Use
`--gsm8k_answer_mode {thinking,final}` (full chain-of-thought target vs. number-only) and `--generation_max_length` to
control eval generation length.

## Usage

```bash
pip install "peft[test]" datasets

# Sequence classification (accuracy), AG News, GPU 4090
CUDA_VISIBLE_DEVICES=4,5,6,7 python run_benchmark.py --task cls --model_name Qwen/Qwen3-8B --methods lora --per_device_train_batch_size 2

# implicit shadow
CUDA_VISIBLE_DEVICES=4,5 python run_benchmark.py --task cls --model_name Qwen/Qwen3-8B --methods shadow --per_device_train_batch_size 2


# GSM8K (generated exact-match accuracy)
python run_benchmark.py --task gsm8k --model_name Qwen/Qwen3-0.6B --methods lora shadow --bf16 --max_seq_length 512

# Explicit (optionally pre-trained) shadow model instead of the implicit one
python run_benchmark.py --task gsm8k --model_name Qwen/Qwen3-8B --methods shadow --shadow_model_name shadow-llm/Qwen3-0.6B-H8B --bf16 

# Projected shadow model: a small backbone + a trained hidden projection aligned to a larger base
python run_benchmark.py --task gsm8k --model_name Qwen/Qwen3-8B --methods shadow \
    --shadow_model_name shadow-llm/Qwen3-0.6B-H8B --bf16

# Quick smoke run on a small subset
python run_benchmark.py --task cls --training_samples 32 --max_steps 5 --bf16

# GSM8K pipeline check (train + generation eval, including FSDP)
accelerate launch --config_file fsdp_config.yaml run_benchmark.py \
    --task gsm8k --methods lora --training_samples 8 --max_steps 2 --bf16 \
    --gsm8k_max_print_predictions 8
```

## Train / eval split

Use `--mode` to separate training from evaluation. Adapters are saved under `output_dir/<method>/` (plus
`train_summary.json` with train time and parameter counts).

```bash
# 1) Train only (multi-GPU / FSDP is fine)
accelerate launch --config_file fsdp_config.yaml run_benchmark.py \
    --mode train --task gsm8k --methods lora --bf16 --max_steps 100

# 2) Eval only on a single GPU (much faster GSM8K generation; no FSDP overhead)
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py \
    --mode eval --task gsm8k --methods lora --bf16 --training_samples 32 \
    --generation_max_length 64

# Default --mode both keeps the original train-then-eval behaviour.
```

For `--mode eval`, pass the same `--task` and method-specific flags used during training. The base model is read
automatically from the saved adapter (`base_model_name_or_path`), so `--model_name` can be omitted if it matches.
Use `--adapter_dir` if adapters were saved outside the default `output_dir`.

Key flags: `--methods {lora,shadow}` (run one or both), LoRA knobs (`--lora_r`, `--lora_alpha`,
`--lora_target_modules`), and ShadowPEFT knobs (`--shadow_layers`, `--injection_hidden_size`, `--gate_hidden_size`,
`--shadow_intermediate_size`, `--shadow_alpha`, `--shadow_loss_weight`). Pass `--shadow_model_name <hub-or-path>` to use
an explicit shadow model (a hidden-size projection is added automatically when sizes differ). The explicit model may
also be a **projected** shadow checkpoint (`AutoModelForCausalLMWithHiddenProjection`, e.g.
[`shadow-llm/Qwen3-0.6B-H8B`](https://huggingface.co/shadow-llm/Qwen3-0.6B-H8B) — a Qwen3-0.6B backbone with a trained
1024→4096 projection for a Qwen3-8B base); ShadowPEFT detects it and reuses the trained projection instead of
initializing a new one. Shared training flags (learning rate, batch size, epochs/steps) apply equally to both methods.

The script prints a comparison table and writes `benchmark_outputs/results.json`. When `--methods shadow` is included,
the table also reports **shadow_only** metrics (lightweight shadow-path inference without a base forward pass).

## Multi-GPU

The script uses the Hugging Face `Trainer`. Two distributed modes are supported:

Launch with `accelerate` or `torchrun` — each GPU holds a full model copy:

set following envs if using 4090
```bash
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
```

### Classification: CR

```bash
# 1. CR, LoRA 
# trainable params: 30,679,040 || all params: 7,599,092,736 || trainable%: 0.4037

======================================================================
Comparison (cls, Qwen/Qwen3-8B) CR
======================================================================
method         trainable  trainable%  train_time(s)      accuracy
-----------------------------------------------------------------
lora          30,679,040      0.404%          270.0        0.9495


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 1234 run_benchmark.py --task cls --methods lora --bf16 --max_seq_length 128 --num_train_epochs 3 --lora_dropout 0.2

# 2. CR, implicit shadow, 
# trainable params: 29,135,104 || all params: 7,597,548,800 || trainable%: 0.3835
======================================================================
Comparison (cls, Qwen/Qwen3-8B)
======================================================================
method         trainable  trainable%  train_time(s)      accuracy   shadow_only
-------------------------------------------------------------------------------
shadow        29,135,104      0.383%          290.4        0.9574        0.8059


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 1235 run_benchmark.py --task cls --methods shadow --bf16 --shadow_num_attention_heads 8 --shadow_alpha 0.1 --max_seq_length 128 --learning_rate 5e-4 --num_train_epochs 3 --output_dir benchmark_outputs/implicit-shadow


# 3. CR, explicit shadow
# trainable params: 609,460,224 || all params: 8,177,882,112 || trainable%: 7.4525
======================================================================
Comparison (cls, Qwen/Qwen3-8B)
======================================================================
Comparison (cls, Qwen/Qwen3-8B)
======================================================================
method         trainable  trainable%  train_time(s)      accuracy   shadow_only
-------------------------------------------------------------------------------
shadow       609,443,840      7.452%         1031.8        0.9574        0.6383


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 1236 run_benchmark.py --task cls --methods shadow --bf16 --shadow_model_name shadow-llm/Qwen3-0.6B-H8B --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --shadow_alpha 1.0 --max_seq_length 128 --learning_rate 5e-5 --num_train_epochs 2 --output_dir benchmark_outputs/explicit-shadow
```

#### Generation: GSM8k

```bash
# 1. GSM8k, LoRA 
# trainable params: 30,670,848 || all params: 8,221,406,208 || trainable%: 0.3731
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 accelerate launch --config_file fsdp_config.yaml --num_processes 7 --main_process_port 1234 run_benchmark.py --task gsm8k --methods lora --bf16 --max_seq_length 512 --output_dir benchmark_outputs/gsm8k-lora --gradient_checkpointing --mode train --num_train_epochs 2 

======================================================================
Comparison (gsm8k, Qwen/Qwen3-8B)
======================================================================
method         trainable  trainable%  train_time(s)      accuracy
-----------------------------------------------------------------
lora          30,670,848      0.373%         3559.0        0.8180
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --mode eval --task gsm8k --methods lora --bf16 --output_dir benchmark_outputs/gsm8k-lora --generation_max_length 256


# 2. GSM8k, implicit shadow, 
# trainable params: 29,118,720 || all params: 8,842,183,936 || trainable%: 0.3293
CUDA_VISIBLE_DEVICES=3,4,5,6,7 accelerate launch --config_file fsdp_config.yaml --num_processes 5 --main_process_port 1234 run_benchmark.py --task gsm8k --methods shadow --bf16 --max_seq_length 512 --shadow_num_attention_heads 8 --output_dir benchmark_outputs/gsm8k-implicit-shadow2 --learning_rate 2e-3 --gradient_checkpointing --mode train --num_train_epochs 2 --shadow_loss_weight 0


======================================================================
Comparison (gsm8k, Qwen/Qwen3-8B)
======================================================================
method         trainable  trainable%  train_time(s)      accuracy   shadow_only
-------------------------------------------------------------------------------
shadow        29,118,720      0.329%         8710.6        0.8309        0.0000
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --mode eval --task gsm8k --methods shadow --bf16 --adapter_dir benchmark_outputs/gsm8k-implicit-shadow2 --generation_max_length 256

# 3. GSM8k, explicit shadow
# trainable params: 609,460,224 || all params: 8,177,882,112 || trainable%: 7.4525
CUDA_VISIBLE_DEVICES=0,1,4,5,6,7 accelerate launch --config_file fsdp_config.yaml --num_processes 6 --main_process_port 1236 run_benchmark.py --task gsm8k --methods shadow --bf16 --shadow_model_name shadow-llm/Qwen3-0.6B-H8B --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --max_seq_length 512 --output_dir benchmark_outputs/gsm8k-explicit-shadow --gradient_checkpointing

CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 accelerate launch --config_file fsdp_config.yaml --num_processes 6 --main_process_port 1234 run_benchmark.py --task gsm8k --methods shadow --bf16 --max_seq_length 512 --shadow_model_name shadow-llm/Qwen3-0.6B-H8B --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --output_dir benchmark_outputs/gsm8k-explicit-shadow6 --gradient_checkpointing --mode train --num_train_epochs 2 --learning_rate 1e-3 --shadow_lr_scale 0.05 --save_steps 500 --save_total_limit 4


CUDA_VISIBLE_DEVICES=1 python run_benchmark.py --mode eval --task gsm8k --methods shadow --bf16 --adapter_dir benchmark_outputs/gsm8k-explicit-shadow6 --generation_max_length 256 --shadow_inference_mode base_shadow

CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --mode eval --task gsm8k --methods shadow --bf16 --adapter_dir benchmark_outputs/gsm8k-explicit-shadow6/shadow --generation_max_length 256 --shadow_inference_mode shadow_only


```


## Notes

- ShadowPEFT disables the KV cache (full-sequence processing); this only affects generation, not the loss/accuracy
  metrics computed here.
- For a fair comparison, keep the shared training flags identical across methods and only vary the method-specific
  capacity knobs. The defaults are reasonable starting points, not tuned optima.
