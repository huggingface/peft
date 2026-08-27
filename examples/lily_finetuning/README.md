# Lily: Low-Rank Interconnected Adaptation Across Layers

## Introduction
[**Lily**](https://arxiv.org/abs/2407.09946) is a PEFT method that introduces **cross-layer parameter sharing** to improve parameter efficiency. Unlike LoRA, which assigns independent adapter pairs to each layer, Lily shares adapter components across layers in two ways:

- **A sharing**: consecutive blocks of `stride_A` layers share the same A adapter, reducing the number of distinct input projections.
- **B sharing**: a small pool of `num_B` B adapters is shared globally across all layers. For each forward pass, a lightweight **router** computes a softmax-weighted combination of all B adapters to produce a layer-specific output projection.

This design allows Lily to cover more layers with fewer parameters, making it possible to use larger rank for each adapter without increasing parameter count and enabling information sharing across layers.

## Quick start

With respect to your standard PEFT training procedure with LoRA, simply swap your `LoraConfig` for a `LilyConfig`.

```python
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LilyConfig

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
lily_config = LilyConfig()

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lily_config,
    args=SFTConfig(
        max_length=2048,
        dataset_text_field="text",
        per_device_train_batch_size=2,
    ),
)
trainer.train()
trainer.model.save_pretrained("lily-llama-3.2-3b")
```

Run the finetuning script simply by running:
```sh
python examples/lily_finetuning/lily_finetuning.py --base_model meta-llama/Llama-3.2-3B --data_path timdettmers/openassistant-guanaco
```

## Use the model on 🤗
You can load and use the model as any other 🤗 models.
```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B", dtype=torch.bfloat16, device_map="auto"
)
peft_model = PeftModel.from_pretrained(model, "lily-llama-3.2-3b")
```

## Additional Notes

- `r` controls the rank (inner hidden dimension). Since Lily typically uses fewer adapter instances than LoRA, it is recommended to use a **larger `r`** — typically `2x`–`4x` the rank you would use in LoRA.
- `stride_A` controls how many consecutive layers share one A adapter. Larger `stride_A` means fewer distinct A adapters and fewer trainable parameters. Suggested values: `2`, `3`, or `4`. Make sure that `total_layers` is divisible by `stride_A` to ensure even sharing.
- `num_B` controls the size of the shared B adapter pool. It is recommended to set `num_B` to roughly `total_layers / stride_A`. Note that `num_B >= 2` is required.
- `scaling` is a direct scalar multiplier on the adapter output (analogous to `alpha / r` in LoRA). It is recommended to start with `2.0` and treat it as a hyperparameter.
- The general rule of thumb: **prefer larger `r` with larger `stride_A` and smaller `num_B`** over smaller `r` with smaller `stride_A` and larger `num_B`.

## Citation
```
@inproceedings{zhong-etal-2025-low,
    title = "Low-Rank Interconnected Adaptation across Layers",
    author = "Zhong, Yibo  and
      Zhao, Jinman  and
      Zhou, Yao",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.874/",
    doi = "10.18653/v1/2025.findings-acl.874",
    pages = "17005--17029",
    ISBN = "979-8-89176-256-5",
    abstract = "Low-rank adaptation (LoRA) is a widely used parameter-efficient fine-tuning (PEFT) method that learns weight updates $\Delta W = AB$ for pretrained weights $W$ through low-rank adapters $A$ and $B$. While LoRA ensures hardware efficiency, its low-rank weight updates limit adaptation performance. In this paper, we propose low-rank interconnected adaptation across layers (Lily), a novel PEFT method that introduces an interconnected framework with locally shared $A$ and globally shared $B$ experts. This structure eliminates redundant per-layer $AB$ pairs, enabling higher-rank $\Delta W$ with equal or fewer parameters. To enhance expressiveness, we use data-dependent routers to determine $A$-$B$ interconnections, preventing $B$ experts from converging to the same behavior and improving representational power across domains. Experiments across modalities, architectures, and model sizes demonstrate Lily{'}s superior performance and efficiency."
}
```