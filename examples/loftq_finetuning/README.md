# LoftQ: LoRA-fine-tuning-aware Quantization

## Introduction

LoftQ finds quantized LoRA initialization: quantized backbone Q and LoRA adapters A and B, given a pre-trained weight W.

## Quick Start
Steps:

1. Apply LoftQ to a full-precision pre-trained weight and save.
2. Load LoftQ initialization and train.

For step 1, we have provided off-the-shelf LoftQ initializations (see [supported model list](#appendix-off-the-shelf-model-table)) 
in [Huggingface Hub LoftQ](https://huggingface.co/LoftQ).
If you want to do it yourself, jump to [LoftQ DIY](#loftq-diy).

For step 2, below is an example of loading 4bit Mistral-7B with 64rank LoRA adapters from Huggingface Hub.
```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "LoftQ/Mistral-7B-v0.1-4bit-64rank"

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16,  # you may change it with different models
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 is recommended
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type='nf4',
    ),
)
peft_model = PeftModel.from_pretrained(
    base_model,
    MODEL_ID,
    subfolder="loftq_init",
    is_trainable=True,
)

# Do training with peft_model ...
```

## LoftQ DIY

### Apply LoftQ and save
We provide [quantize_save_load.py](quantize_save_load.py) as an example to apply LoftQ with 
different bits(`--bits`), ranks(`--rank`), and alternating steps (`--iter`, a hyper-parameter in LoftQ, see Algorithm 1 in [LoftQ paper](https://arxiv.org/abs/2310.08659)). Currently, this example supports
`llama-2`, `falcon`, `mistral`, `bart`, `t5`, `deberta`, `bert`, `roberta`.

Below is an example of obtaining 4bit LLAMA-2-7b with 16-rank LoRA adapters by 5 alternating steps.
```sh
SAVE_DIR="model_zoo/loftq/"
python quantize_save_load.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \  # high-precision model id in HF
    --token HF_TOKEN \  # your HF token if the model is private, e.g., llama-2
    --bits 4 \
    --iter 5 \
    --rank 16 \
    --save_dir $SAVE_DIR
```

The above commands end up with creating the model directory under `$SAVE_DIR`. 
Specifically, the model directory is named as 

`MODEL_DIR = SAVE_DIR + f"{args.model_name_or_path.split('/')[-1]}-{args.bits}bits-{args.rank}rank"`

In this example, `MODEL_DIR="model_zoo/loftq/Llama-2-7b-hf-4bit-16rank"`, where the backbone is stored in `$MODEL_DIR`
and the LoRA adapters are at the sub-folder `$MODEL_DIR/loftq_init`.

### Load and train
Similar to loading from Huggingface Hub, we only need to change the `MODEL_ID` to the `MODEL_DIR`.

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_DIR = "model_zoo/loftq/Llama-2-7b-hf-4bit-16rank"

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, 
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type='nf4',
    ),
)
peft_model = PeftModel.from_pretrained(
    base_model,
    MODEL_DIR,
    subfolder="loftq_init",
    is_trainable=True,
)
# Do training with peft_model ...
```

## LoftQ Fine-tuning

We also provide an example to fine-tune LoftQ on GSM8K. 
We load the quantized backbone and LoRA adapters from the [LoftQ Huggingface hub](https://huggingface.co/LoftQ).

```sh
python train_gsm8k_llama.py \
    --model_name_or_path LoftQ/Llama-2-13b-hf-4bit-64rank \
    --output_dir exp_results/gsm8k/llama-2-13b/bit4-rank64/lr1e-4 \
    --learning_rate 1e-4  \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 100 \
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


## Appendix: Off-the-shelf Model List
| Model Name  | Bits | Ranks |
| ----------- | ---- | ----- |
| LLAMA-2-7b  | 4    | 64    |
| LLAMA-2-13b | 4    | 64    |
| LLAMA-2-70b | 4    | 64    |
| Mistral     | 4    | 64    |
| Mistral     | 4    | 32    |
| BART-large  | 4    | 8     |
| BART-large  | 4    | 16    |
| BART-large  | 4    | 32    |
| BART-large  | 2    | 8     |

## In-place application of LoftQ initialization

PEFT provides a convenience function `replace_lora_weights_loftq` to apply LoftQ initialization in-place to the quantized model. Check out [this notebook](https://github.com/huggingface/peft/blob/main/examples/loftq_finetuning/LoftQ_weight_replacement.ipynb) for an example.
