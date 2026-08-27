

# WaveFT: Wavelet Fine-Tuning

## Introduction
[WaveFT](https://huggingface.co/papers/2505.12532) is a novel parameter-efficient fine-tuning (PEFT) method that introduces sparse updates in the **wavelet domain** of residual matrices. Unlike LoRA, which is constrained by discrete low-rank choices, WaveFT enables fine-grained control over the number of trainable parameters by directly learning a sparse set of coefficients in the transformed space. These coefficients are then mapped back to the weight domain via the Inverse Discrete Wavelet Transform (IDWT), producing high-rank updates without incurring inference overhead.

## Quick start
```python
import torch
from peft import WaveFTConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
dataset = load_dataset("imdb", split="train[:1%]")
waveft_config = WaveFTConfig(
    n_frequency=2592,
)
peft_model = get_peft_model(model, waveft_config)
training_args = SFTConfig(dataset_text_field="text", max_length=128)
trainer = SFTTrainer(
    model=peft_model,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
peft_model.save_pretrained("waveft-opt-350m")
```

For more options and a more detailed example code, you can refer to waveft finetuning script.
Run the script simply by running:
```bash
python3 examples/waveft_finetuning/waveft_finetuning.py --base_model facebook/opt-350m
```

If you want to run DDP by [accelerate](https://huggingface.co/docs/accelerate/en/index), please run `accelerate config` to set your ddp config, and run:
```bash
accelerate launch examples/waveft_finetuning/waveft_finetuning.py --base_model facebook/opt-350m
```
please add `--device_map cpu` if you want to run finetune on CPU.

## Use the model
You can load and use the model as any other 🤗 PEFT model
```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
waveft_model = PeftModel.from_pretrained(model, "waveft-opt-350m")
```

## Citation
@misc{bilican2025exploringsparsityparameterefficient,
      title={Exploring Sparsity for Parameter Efficient Fine Tuning Using Wavelets}, 
      author={Ahmet Bilican and M. Akın Yılmaz and A. Murat Tekalp and R. Gökberk Cinbiş},
      year={2025},
      eprint={2505.12532},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.12532}, 
}