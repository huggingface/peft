# PaCA: Partial Connection Adaptation

## Introduction

[PaCA](https://huggingface.co/papers/2503.01905) (ICLR 2025) fine-tunes `r` randomly selected input columns ("partial
connections") of each targeted pretrained weight instead of adding adapter layers. Each adapted layer still runs a
single matmul in forward and backward, and only the `r` input channels that belong to the trainable columns are kept
for the backward pass. This makes training steps faster and reduces activation memory compared to adapter methods such
as LoRA, which run an extra branch next to every layer and keep the full input activation.

## Quick start

```python
import torch
from peft import PacaConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
paca_config = PacaConfig(r=16, paca_alpha=16, target_modules="all-linear", random_seed=0)
peft_model = get_peft_model(model, paca_config)
peft_model.print_trainable_parameters()

training_args = SFTConfig(dataset_text_field="text", max_length=512)
trainer = SFTTrainer(model=peft_model, train_dataset=dataset, processing_class=tokenizer, args=training_args)
trainer.train()
peft_model.save_pretrained("paca-llama-3.2-3b")
```

To run the example script:

```bash
python examples/paca_finetuning/paca_finetuning.py --base_model meta-llama/Llama-3.2-3B --rank 16 --paca_alpha 16
```

`paca_alpha / rank` scales the update of the selected columns and thus acts as a multiplier of the effective learning
rate.

## Use the model

You can load and use the model as any other 🤗 PEFT model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
paca_model = PeftModel.from_pretrained(model, "paca-llama-3.2-3b")
```

The selected column indices are stored in the adapter checkpoint. The adapter can be merged exactly into the base
weights with `merge_and_unload()`. Since the update is non-zero in `r` columns only, it can also be converted without
loss into a LoRA adapter of rank `r`:

```python
from peft.tuners.lora.conversion import convert_to_lora

lora_config, lora_state_dict = convert_to_lora(paca_model, rank=16)
```

## Citation

```
@inproceedings{woo2025paca,
  title={PaCA: Partial Connection Adaptation for Efficient Fine-Tuning},
  author={Woo, Sunghyeon and Namkung, Sol and Lee, Sunwoo and Jeong, Inho and Kim, Beomseok and Jeon, Dongsuk},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
