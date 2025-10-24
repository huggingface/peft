# GraLoRA: Granular Low-Rank Adaptation

![GraLoRA Overview](https://github.com/SqueezeBits/GraLoRA/raw/main/figure/gralora_overview.png)

## Introduction
[**Granular Low-Rank Adaptation (GraLoRA)**](https://huggingface.co/papers/2505.20355) is a PEFT method designed to enhance the **expressivity** of low-rank adaptation while improving **robustness to outlier** activations, based on insights from well-known issues in quantization. 

GraLoRA introduces a structured and fine-grained adaptation scheme. It divides the adaptation space into a grid of $𝑘^2$ smaller, independent adapter pairs, each responsible for a localized subset of the input and output dimensions.

## Quick start

With respect to your standard PEFT training procedure with LoRA, simply swap your `LoraConfig` for a `GraloraConfig`.

```python
import torch
from peft import GraloraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
gralora_config = GraloraConfig()
peft_model = get_peft_model(model, gralora_config)
trainer = transformers.Trainer(
    model=peft_model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
)
trainer.train()
peft_model.save_pretrained("gralora-llama-3-8b")
```

Run the finetuning script simply by running:
```python
python examples/gralora_finetuning/gralora_finetuning.py --base_model meta-llama/Meta-Llama-3-8B --data_path timdettmers/openassistant-guanaco
```

## Use the model on 🤗
You can load and use the model as any other 🤗 models.
```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B", dtype=torch.bfloat16, device_map="auto"
)
peft_model = PeftModel.from_pretrained(model, "gralora-llama-3-8b")
```

## Additonal Notes
While `gralora_k` is set to 2 for default, you can increase this value to create more fine-grained adapters. `gralora_k` of 4 is recommended when the total rank (`r + hybrid_r`) is 64 or higher.




## Citation
```
@misc{jung2025graloragranularlowrankadaptation,
      title={GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning}, 
      author={Yeonjoon Jung and Daehyun Ahn and Hyungjun Kim and Taesu Kim and Eunhyeok Park},
      year={2025},
      eprint={2505.20355},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.20355}, 
}
```
