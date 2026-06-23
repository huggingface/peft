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
uses **KV cache** for LoRA (`use_cache=False` for ShadowPEFT), and uses a single FSDP `summon_full_params` with
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

Results:

```
======================================================================
Comparison (cls, Qwen/Qwen3-8B)
======================================================================
method         trainable  trainable%  train_time(s)      accuracy   shadow_only
-------------------------------------------------------------------------------
lora          30,679,040      0.404%          270.0        0.9495        /
shadow        29,135,104      0.383%          290.4        0.9574        0.8059
```

```bash
# 1. CR, LoRA 
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 1234 run_benchmark.py --task cls --methods lora --bf16 --max_seq_length 128 --num_train_epochs 3 --lora_dropout 0.2

# 2. CR, implicit shadow
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 1235 run_benchmark.py --task cls --methods shadow --bf16 --shadow_num_attention_heads 8 --shadow_alpha 0.1 --max_seq_length 128 --learning_rate 5e-4 --num_train_epochs 3 --output_dir benchmark_outputs/implicit-shadow
```

#### Generation: GSM8k

> ran on A800

Results: 

```
======================================================================
Comparison (gsm8k, Qwen/Qwen3-8B)
======================================================================
method                   trainable  trainable%  train_time(s)      accuracy   shadow_only
-------------------------------------------------------------------------------
lora                     30,670,848      0.373%          696.2        0.8014        /
random shadow            29,118,720      0.329%          721.7        0.8153        0.0159
pretrained-shadow        453,844,992     4.817%         877.0       0.8173       0.5042
```

```bash
# 1. GSM8k, LoRA 
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 1233 run_benchmark.py --task gsm8k --methods lora --bf16 --max_seq_length 512 --output_dir benchmark_outputs/gsm8k-lora --mode train --num_train_epochs 2 --per_device_train_batch_size 8

# eval: 80.14
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --mode eval --task gsm8k --methods lora --bf16 --output_dir benchmark_outputs/gsm8k-lora --generation_max_length 256


# 2. GSM8k, implicit shadow, 
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 1234 run_benchmark.py --task gsm8k --methods shadow --bf16 --max_seq_length 512 --shadow_num_attention_heads 8 --output_dir benchmark_outputs/gsm8k-implicit-shadow --learning_rate 2e-3 --mode train --num_train_epochs 2 --per_device_train_batch_size 8

# eval: 81.53
CUDA_VISIBLE_DEVICES=1 python run_benchmark.py --mode eval --task gsm8k --methods shadow --bf16 --adapter_dir benchmark_outputs/gsm8k-implicit-shadow --generation_max_length 256

# 3. GSM8k, explicit shadow
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 1235 run_benchmark.py --task gsm8k --methods shadow --bf16 --max_seq_length 512 --shadow_model_name shadow-llm/Qwen3-0.6B-H8B --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --output_dir benchmark_outputs/gsm8k-explicit-shadow --mode train --num_train_epochs 2 --learning_rate 1e-3 --shadow_lr_scale 0.05 --save_steps 500 --save_total_limit 4 --shadow_alpha 0.5 --shadow_loss_weight 1.0

# eval 1: base_shadow 81.73
CUDA_VISIBLE_DEVICES=2 python run_benchmark.py --mode eval --task gsm8k --methods shadow --bf16 --adapter_dir benchmark_outputs/gsm8k-explicit-shadow --generation_max_length 256 --shadow_inference_mode base_shadow

# eval 2: shadow only 50.42
CUDA_VISIBLE_DEVICES=3 python run_benchmark.py --mode eval --task gsm8k --methods shadow --bf16 --adapter_dir benchmark_outputs/gsm8k-explicit-shadow --generation_max_length 256 --shadow_inference_mode shadow_only
```


## Notes

- ShadowPEFT disables the KV cache (full-sequence processing); this only affects generation, not the loss/accuracy
  metrics computed here.
- For a fair comparison, keep the shared training flags identical across methods and only vary the method-specific
  capacity knobs. The defaults are reasonable starting points, not tuned optima.
