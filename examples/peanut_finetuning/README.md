# PEANuT: Parameter-Efficient Adaptation with Weight-aware Neural Tweakers

## Introduction
[**PEANuT**](https://arxiv.org/abs/2410.01870) is a PEFT method that introduces a **weight-aware neural tweaker** to generate adapter updates from the base weight itself. Instead of directly learning a low-rank decomposition `Delta W = A @ B` as in LoRA, PEANuT transforms the target layer weight through a small neural network (the neural tweaker) to produce `Delta W`.

PEANuT is built on three key ideas:

- **Weight-aware adaptation**: `Delta W` is produced by transforming the base weight using `A`, `B`, and optional intermediate layers. Because PEANuT applies `A` on the output dimension of the base weight, `A` has shape `(out_features, r)` instead of LoRA's typical `(in_features, r)`. When `in_features > out_features`, PEANuT can use fewer parameters than LoRA at the same rank.
- **Non-linearity inside the tweaker**: PEANuT inserts activation functions in the neural tweaker (default: `relu`) to increase expressiveness.
- **Depth capacity increase**: Besides mandatory `A` and `B`, PEANuT can insert intermediate `r x r` layers in residual encoder/decoder pairs. Here, `depth` counts the number of residual pairs, so `depth=0` means only `A` and `B`.

## Quick start

With respect to your standard PEFT training procedure with LoRA, simply swap your `LoraConfig` for a `PeanutConfig`.

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from peft import PeanutConfig

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
peanut_config = PeanutConfig()

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peanut_config,
    args=SFTConfig(
        max_length=2048,
        dataset_text_field="text",
        per_device_train_batch_size=2,
    ),
)
trainer.train()
trainer.model.save_pretrained("peanut-llama-3.2-3b")
```

Run the finetuning script simply by running:
```sh
python examples/peanut_finetuning/peanut_finetuning.py --base_model meta-llama/Llama-3.2-3B --data_path timdettmers/openassistant-guanaco
```

## Use the model on Hugging Face
You can load and use the model as any other Hugging Face model.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B", dtype=torch.bfloat16, device_map="auto"
)
peft_model = PeftModel.from_pretrained(model, "peanut-llama-3.2-3b")
```

## Additional Notes

- `r` controls the hidden rank of the neural tweaker. Larger `r` increases capacity and trainable parameters.
- `depth` controls the number of intermediate encoder/decoder residual pairs. It must be a non-negative integer.
- `depth=0` means only `A` and `B`.
- `depth=1` adds one encoder/decoder residual pair between `A` and `B`.
- Larger depths add more `r x r` residual pairs.
- `act_fn` controls the non-linearity inside PEANuT and defaults to `relu`.
- `scaling` is a direct scalar multiplier on the adapter output before it is added to the frozen base layer output.
- PEANuT can perform better than LoRA across a range of tasks. We also find it strong in very low-parameter regimes (for example around `0.2M` trainable parameters).
- Compared with LoRA, PEANuT typically uses more GPU memory and runs slower because it explicitly constructs `Delta W` during forward passes. Adding intermediate layers (higher `depth`) increases this overhead further.

## Citation
```bibtex
@misc{zhong2025peanutparameterefficientadaptationweightaware,
      title={PEANuT: Parameter-Efficient Adaptation with Weight-aware Neural Tweakers}, 
      author={Yibo Zhong and Haoxiang Jiang and Lincan Li and Ryumei Nakada and Tianci Liu and Linjun Zhang and Huaxiu Yao and Haoyu Wang},
      year={2025},
      eprint={2410.01870},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.01870}, 
}
```
