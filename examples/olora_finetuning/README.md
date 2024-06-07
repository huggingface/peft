# OLoRA: Orthonormal Low Rank Adaptation of Large Language Models

## Introduction
[OLoRA](https://arxiv.org/abs/2406.01775) is a novel approach that leverages orthonormal low rank adaptation through QR decomposition. Unlike the default LoRA implementation, OLoRA decomposes original weights into their $\mathbf{Q}$ and $\mathbf{R}$ parts, and then uses the first `rank` rows of $\mathbf{R}$ and the first `rank` columns of $\mathbf{Q}$ to initialize $\mathbf{A}$ and $\mathbf{B}$, respectively. This results in significantly faster convergence, more stable training, and superior performance.

## Quick start
```python
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
dataset = load_dataset("imdb", split="train[:1%]")
lora_config = LoraConfig(
    init_lora_weights="olora"
)
peft_model = get_peft_model(model, lora_config)
trainer = SFTTrainer(
    model=peft_model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
)
trainer.train()
peft_model.save_pretrained("olora-opt-350m")
```

There is no additional change needed to your standard LoRA procedure, except for specifying `init_lora_weights = "olora"` option in your lora configuration.

Additionally you can refer to olora finetuning script.
Run the script simply by running:
```bash
python3 examples/olora_finetuning/olora_finetuning.py --base_model facebook/opt-350m
```
OLoRA also supports quantization. To use 4-bit quantization try:
```bash
python3 examples/olora_finetuning/olora_finetuning.py --base_model facebook/opt-350m --quantize
```


## Use the model
You can load and use the model as any other 🤗 PEFT model
```python
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
olora_model = PeftModel.from_pretrained(model, "olora-opt-350m")
print(olora_model)
```

## OLoRA and LoRA
Unlike LoRA, OLoRA mutates the original weights. If you want to use multiple adapters simultaneously you can use `path_initial_model_for_weight_conversion` option. First save your model that was initialized with OLoRA **before** performing any training:
```python
init_path = <path-to-untrained-olora-model>
olora_model.save_pretrained(init_path)
```
# After training
Then you can specify the path of the initialized adapter in the `save_pretrained` method:
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
olora_model = get_peft_model(model, lora_config)
olora_model.save_pretrained(save_dir, path_initial_model_for_weight_conversion=init_path)
```
Thus converting OLoRA to LoRA to use with multiple adapters.

## Citation
```
@misc{büyükakyüz2024olora,
      title={OLoRA: Orthonormal Low-Rank Adaptation of Large Language Models}, 
      author={Kerim Büyükakyüz},
      year={2024},
      eprint={2406.01775},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```