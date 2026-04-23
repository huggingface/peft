# Activated LoRA (aLoRA)

## Introduction
Activated LoRA (aLoRA) is an adapter that selectively activates its weights only after a given invocation sequence, ensuring that hidden states match the base model prior to this point. This allows reusing the base model KVs (stored in the KV cache) for tokens before the invocation,
enabling much faster real-world inference (e.g. vLLM) when switching between generation with the base model and generation with adapters.
See the [paper](https://huggingface.co/papers/2504.12397) for more details.

## Quick start (shown for Mistral 7B)
```python
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
dataset = load_dataset("Lots-of-LoRAs/task1660_super_glue_question_generation", split="train")

invocation_string = "[/INST]" # End of user turn in Mistral chat template
invocation_tokens = tokenizer.encode(invocation_string, add_special_tokens=False)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    alora_invocation_tokens=invocation_tokens,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
)

peft_model = get_peft_model(model, lora_config)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
trainer = Trainer(
    model=peft_model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_length=2048,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.train()
peft_model.save_pretrained("alora-mistral-7b")
```

### Use the training example script directly
Pass the invocation string with `--invocation_string` when running the training example
script. For Mistral 7B, do:
```bash
python examples/alora_finetuning/alora_finetuning.py --base_model mistralai/Mistral-7B-Instruct-v0.3 --data_path Lots-of-LoRAs/task1660_super_glue_question_generation --invocation_string "[/INST]"
```
and similarly for Llama-3.2-3B-Instruct:
```bash
python examples/alora_finetuning/alora_finetuning.py --base_model meta-llama/Llama-3.2-3B-Instruct --data_path Lots-of-LoRAs/task1660_super_glue_question_generation --invocation_string "<|start_header_id|>assistant<|end_header_id|>"
```

### Full example of the script
```bash
python alora_finetuning.py \
    --base_model "PATH_TO_MODEL" \
    --data_path "PATH_TO_DATASET" \
    --output_dir "PATH_TO_OUTPUT_DIR" \
    --batch_size 1 \
    --num_epochs 3 \
    --learning_rate 3e-4 \
    --cutoff_len 512 \
    --val_set_size 500 \
    --invocation_string "[/INST]" \
    --quantize \
    --eval_step 10 \
    --save_step 100 \
    --device "auto" \
    --lora_r 32 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --hub_model_id "YOUR_HF_REPO" \
    --push_to_hub
```
