# Sparse High Rank Adapters

## Introduction
Sparse High Rank Adapters or [SHiRA](https://huggingface.co/papers/2406.13175) is an alternate type of adapter and has been found to have significant advantages over the low rank adapters. Specifically, SHiRA achieves better accuracy than LoRA for a variety of vision and language tasks. It also offers simpler and higher quality multi-adapter fusion by significantly reducing concept loss, a common problem faced by low rank adapters. SHiRA directly finetunes a small number of the base model's parameters to finetune the model on any adaptation task.

## Quick start
```python
import torch
from peft import ShiraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
dataset = load_dataset("imdb", split="train[:1%]")
shira_config = ShiraConfig(
    r=32,
)
peft_model = get_peft_model(model, shira_config)
training_args = SFTConfig(dataset_text_field="text", max_length=128)
trainer = SFTTrainer(
    model=peft_model,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
peft_model.save_pretrained("shira-opt-350m")
```

For more options and a more detailed example code, you can refer to shira finetuning script.
Run the script simply by running:
```bash
python3 examples/shira_finetuning/shira_finetuning.py --base_model facebook/opt-350m
```

If you want to run DDP by [accelerate](https://huggingface.co/docs/accelerate/en/index), please run `accelerate config` to set your ddp config, and run:
```bash
accelerate launch examples/shira_finetuning/shira_finetuning.py --base_model facebook/opt-350m
```
please add `--device_map cpu` if you want to run finetune on CPU.

If you want to train SHiRA with a custom sparse mask function which requires custom keyword arguments, please see the definition of `custom_random_mask_function_with_custom_kwargs` function provided in the `shira_fintuning.py` script. You can run this code using the `--use_custom_random_mask_function_with_custom_kwargs` argument. Without this argument, SHiRA defaults to a random sparse mask. Please run the code as follows. :
```bash
python3 examples/shira_finetuning/shira_finetuning.py --base_model facebook/opt-350m --use_custom_random_mask_function_with_custom_kwargs

```


## Use the model
You can load and use the model as any other 🤗 PEFT model
```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
shira_model = PeftModel.from_pretrained(model, "shira-opt-350m")
```

## Citation
```
@inproceedings{NEURIPS2024_18c0102c,
 author = {Bhardwaj, Kartikeya and Pandey, Nilesh Prasad and Priyadarshi, Sweta and Ganapathy, Viswanath and Kadambi, Shreya and Esteves, Rafael and Borse, Shubhankar and Whatmough, Paul and Garrepalli, Risheek and Van Baalen, Mart and Teague, Harris and Nagel, Markus},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {13685--13715},
 publisher = {Curran Associates, Inc.},
 title = {Sparse High Rank Adapters},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/18c0102cb7f1a02c14f0929089b2e576-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```
