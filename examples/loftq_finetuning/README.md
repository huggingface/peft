# LoftQ: LoRA-fine-tuning-aware Quantization

## Introduction

LoftQ provides better initialization for LoRA adapters A and B, 
and the Quantization of pre-trained weights W.

## Quantization
We recommend to save the quantized backbone model as fp16/fp32 
and load it as [NormalFloat4](https://arxiv.org/abs/2305.14314).

We provide a simple example to show how to quantize llama-2-7b model and save/load it.

```sh
python quantize_save_load.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --token HF_TOKEN \
    --bits 4 --iter 5 --rank 16 \
    --save_dir model_zoo/loftq/
```

- `HF_TOKEN` is the token used to access to [LLAMA models](https://huggingface.co/meta-llama).
- `quantize_and_save()` function will quantize the backbone and initialize LoRA adapters. 
It creates 2 folders under `$save_dir`. The quantized backbone is at `Llama-2-7b-hf-4bit-16rank`,
and the LoRA adapters are at the sub-folder `Llama-2-7b-hf-4bit-16rank/loftq_init`.

## Fine-tuning

Here is an example to load the quantized backbone and LoRA adapters:

```python
import os

from transformers import AutoModelForCausalLM
from peft import PeftModel


base_model = AutoModelForCausalLM.from_pretrained(
    os.path.join(args.save_dir, "Llama-2-7b-hf-4bit-16rank"), 
    load_in_4bit=True,
)
peft_model = PeftModel.from_pretrained(
    base_model,
    os.path.join(args.save_dir, "Llama-2-7b-hf-4bit-16rank", "loftq_init"),
    is_trainable=True,
)
```

We also provide an example to fine-tune LoftQ on GSM8K. 
We load the quantized backbone and LoRA adapters from the [LoftQ Huggingface hub](https://huggingface.co/LoftQ).

```sh
python train_gsm8k_llama.py \
    --model_name_or_path LoftQ/Llama-2-7b-hf-4bit-64rank \
    --output_dir exp_results/gsm8k/llama-2-7b/bit4-rank64/lr3e-4 \
    --learning_rate 3e-4  \
    --seed 202 \
    --dataset_name gsm8k \
    --dataset_config main \
    --pad_to_max_length \
    --max_source_length 128 \
    --max_target_length 256 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --with_tracking \
    --report_to tensorboard
```
