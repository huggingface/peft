# GraLoRA: Granular Low-Rank Adaptation

![GraLoRA Overview](https://github.com/SqueezeBits/GraLoRA/raw/main/figure/gralora_overview.png)

## Introduction
[**Granular Low-Rank Adaptation (GraLoRA)**](https://huggingface.co/papers/2505.20355) is a PEFT method designed to enhance the **expressivity** of low-rank adaptation while improving **robustness to outlier** activations, based on insights from well-known issues in quantization. 

GraLoRA introduces a structured and fine-grained adaptation scheme. It divides the adaptation space into a grid of $𝑘^2$ smaller, independent adapter pairs, each responsible for a localized subset of the input and output dimensions.

## Quick start

With respect to your standard PEFT training procedure with LoRA, simply swap your `LoraConfig` for a `GraloraConfig`.

```python
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import GraloraConfig

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
gralora_config = GraloraConfig()

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=gralora_config,
    args=SFTConfig(
        max_length=2048,
        dataset_text_field="text",
        per_device_train_batch_size=2,
    ),
)
trainer.train()
trainer.model.save_pretrained("gralora-llama-3.2-3b")
```

Run the finetuning script simply by running:
```sh
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

## Additional Notes
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
