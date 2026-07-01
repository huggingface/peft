# UniLoRA: One Vector Is All You Need

## Introduction ([Paper](https://huggingface.co/papers/2506.00799))

UniLoRA shares a compact trainable vector bank across low-rank adapter weights. It keeps the familiar PEFT training
flow while using deterministic projections into shared `theta_d` values to reduce the number of trained adapter
parameters.

## Quick Start

```python
import torch
from datasets import load_dataset
from peft import UniLoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer.pad_token_id = tokenizer.eos_token_id

config = UniLoraConfig(
    r=32,
    theta_d_length=256,
    proj_seed=42,
    target_modules=["q_proj", "v_proj"],
    unilora_dropout=0.0,
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()

dataset = load_dataset("imdb", split="train[:1%]")

training_args = SFTConfig(dataset_text_field="text", max_length=128)
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
peft_model.save_pretrained("unilora-llama-3.2-3b")
```

To load the fine-tuned UniLoRA adapter:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B", dtype=torch.bfloat16, device_map="auto"
)
peft_model = PeftModel.from_pretrained(model, "unilora-llama-3.2-3b")
```

## Fine-tune on MetaMathQA

```shell
python unilora_finetuning.py \
    --base_model_name_or_path meta-llama/Llama-3.2-3B \
    --output_dir output/unilora-llama-3.2-3b-metamath \
    --unilora_r 32 \
    --theta_d_length 256 \
    --proj_seed 42 \
    --unilora_dropout 0.0 \
    --bits bf16 \
    --data_path meta-math/MetaMathQA \
    --dataset_split train[:100000] \
    --dataset_field query response \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_steps 0.03 \
    --tf32 True \
    --report_to none
```
