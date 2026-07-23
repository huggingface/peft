# LoRA vs ShadowPEFT benchmark

A small, self-contained script that trains [LoRA](https://huggingface.co/docs/peft/package_reference/lora) and
[ShadowPEFT](https://huggingface.co/docs/peft/package_reference/shadow) on the same task with the same training budget
and prints a side-by-side comparison (trainable parameters, training time, and the task metric).

Three task types are supported, each with one clear metric:

| `--task` | model class                          | metric                              |
| -------- | ------------------------------------ | ----------------------------------- |
| `cls`    | `AutoModelForSequenceClassification` | accuracy                            |
| `gsm8k`  | `AutoModelForCausalLM`               | exact-match accuracy (generated)    |

Both methods are built from a fresh copy of the same base model via `get_peft_model`, trained with the Hugging Face
`Trainer`, then evaluated identically.

The `gsm8k` task follows the setup in ShadowPEFT's `experiment/run_shadow_peft.py`: it loads `openai/gsm8k`, formats
each example with a chat template, supervises only the answer tokens (the prompt is masked with `-100`), and evaluates
by **generating** an answer and extracting the final number (`#### N`) for exact-match accuracy. Eval skips the
teacher-forced loss pass by default (use `--gsm8k_eval_loss` to enable it). With multiple processes, generation eval
runs on **rank 0 only** by default (use `--gsm8k_distributed_eval` to eval on every rank). Generation is batched,
uses **KV cache** (including ShadowPEFT's dual base+shadow cache), and uses a single FSDP `summon_full_params` with
`synced_gpus=False`. Per-batch timing is printed on rank 0. Use
`--gsm8k_answer_mode {thinking,final}` (full chain-of-thought target vs. number-only) and `--generation_max_length` to
control eval generation length.

## Benchmark

The script uses the Hugging Face `Trainer`. Two distributed modes are supported:

Launch with `accelerate` or `torchrun` — each GPU holds a full model copy:

set following envs if using 4090
```bash
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
```

### Classification: CR

> ran on 4090

```bash
# 1. CR, LoRA 
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 1234 run_benchmark.py --task cls --methods lora --bf16 --max_seq_length 128 --num_train_epochs 2 --per_device_train_batch_size 64 --num_train_epochs 4

# 2. CR, mirror shadow backbone (train)
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 1235 run_benchmark.py --task cls --methods shadow --bf16 --shadow_num_attention_heads 8 --shadow_alpha 2.0 --max_seq_length 128 --learning_rate 2e-4 --num_train_epochs 4 --mode train --output_dir benchmark_outputs/mirror-shadow --per_device_train_batch_size 64

# eval the base+shadow (adapted) model; add --eval_detached_shadow_only to eval ONLY the standalone shadow classifier
CUDA_VISIBLE_DEVICES=2 python run_benchmark.py --task cls --methods shadow --bf16 --max_seq_length 128 --mode eval --adapter_dir benchmark_outputs/mirror-shadow --eval_detached_shadow_only
```

Results (**from the pre-redesign implementation — re-run after the `BaseTuner` rewrite to refresh these numbers**):

```
======================================================================
Comparison (cls, Qwen/Qwen3-8B)
======================================================================
method         trainable  trainable%  train_time(s)      accuracy
-----------------------------------------------------------------
lora          30,679,040      0.404%          117.8        0.9441
shadow        29,684,544      0.391%          121.2        0.9495
shadow only   29,684,544                                   0.7766
```

#### Generation: GSM8k

> ran on A800

```bash
# 1. GSM8k, LoRA 
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 1233 run_benchmark.py --task gsm8k --methods lora --bf16 --max_seq_length 512 --output_dir benchmark_outputs/gsm8k-lora --mode train --num_train_epochs 2 --per_device_train_batch_size 8

# eval: 80.14
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --mode eval --task gsm8k --methods lora --bf16 --output_dir benchmark_outputs/gsm8k-lora --generation_max_length 256


# 2. GSM8k, mirror shadow backbone
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 1234 run_benchmark.py --task gsm8k --methods shadow --bf16 --max_seq_length 512 --shadow_num_attention_heads 8 --output_dir benchmark_outputs/gsm8k-mirror-shadow --learning_rate 2e-4 --mode train --num_train_epochs 2 --per_device_train_batch_size 8

# eval the base+shadow (adapted) model
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --mode eval --task gsm8k --methods shadow --bf16 --adapter_dir benchmark_outputs/gsm8k-mirror-shadow --generation_max_length 256
# eval ONLY the standalone detached shadow network instead (--eval_detached_shadow_only)
CUDA_VISIBLE_DEVICES=1 python run_benchmark.py --mode eval --task gsm8k --methods shadow --bf16 --adapter_dir benchmark_outputs/gsm8k-mirror-shadow --generation_max_length 256 --eval_detached_shadow_only

# 3. GSM8k, shadow backbone initialized from a pretrained model.
#    --shadow_model_name accepts a plain model id (e.g. Qwen/Qwen3-0.6B) or a "projected" shadow checkpoint that also
#    carries a trained hidden-size projection (model_type causal_lm_with_hidden_projection, e.g. shadow-llm/Qwen3-0.6B-H8B).
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 1235 run_benchmark.py --task gsm8k --methods shadow --bf16 --max_seq_length 512 --shadow_model_name shadow-llm/Qwen3-0.6B-H8B --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --output_dir benchmark_outputs/gsm8k-pretrained-shadow --mode train --num_train_epochs 2 --learning_rate 5e-4 --shadow_lr_scale 0.05 --save_steps 500 --save_total_limit 4 --shadow_alpha 0.5 --shadow_loss_weight 1.0

# eval base_shadow
CUDA_VISIBLE_DEVICES=2 python run_benchmark.py --mode eval --task gsm8k --methods shadow --bf16 --adapter_dir benchmark_outputs/gsm8k-pretrained-shadow --generation_max_length 256
# evaluate detached shadow only
CUDA_VISIBLE_DEVICES=3 python run_benchmark.py --mode eval --task gsm8k --methods shadow --bf16 --adapter_dir benchmark_outputs/gsm8k-pretrained-shadow2 --generation_max_length 256 --eval_detached_shadow_only
```

`--eval_detached_shadow_only` (shadow method only) evaluates **only** the standalone detached shadow network
(`unload_shadow()`) in place of the base+shadow (adapted) model — the shadow backbone + projection + head, run as a
plain task model independent of the base model (a causal LM with KV caching for `clm`/`gsm8k`, a last-token-pooled
classifier for `cls`). Its metric is reported in the normal accuracy/perplexity column. It works for all tasks and is
best run single-process (like the eval commands above).

Results (**from the pre-redesign implementation — re-run after the `BaseTuner` rewrite to refresh these numbers**; the
`(detached)` rows are separate `--eval_detached_shadow_only` runs):

```
======================================================================
Comparison (gsm8k, Qwen/Qwen3-8B)
======================================================================
method                       trainable  trainable%  train_time(s)      accuracy
--------------------------------------------------------------------------------
lora                         30,670,848      0.373%          696.2        0.8014
mirror shadow                29,668,160      0.361%          721.7        0.8135
mirror shadow (detached)     29,668,160      0.361%          721.7        0.0136
pretrained-shadow            454,394,432     5.163%          877.0        0.8165
pretrained-shadow (detached) 454,394,432     5.163%          877.0        0.4932
```
