# MiCA: Minor Component Adaptation

## Introduction ([Paper](https://arxiv.org/abs/2604.01694))

Minor Component Adaptation (MiCA) is a parameter-efficient fine-tuning method closely related to LoRA. Like LoRA, MiCA inserts a low-rank update `ΔW = (α/r) · B · A` into a pretrained weight `W ∈ R^{out×in}`. Unlike LoRA, MiCA initializes the matrices from the singular value decomposition of `W` and trains only one of them:

- Compute the SVD `W = U Σ V^T`.
- Initialize `B = U[:, -r:]` — the `r` left singular vectors associated with the **smallest** singular values.
- Initialize `A = 0`.
- During training, optimize only `A`; `W` and `B` remain frozen.

The motivation is that the *minor* singular directions of a pretrained weight encode subspaces that are largely unused by the original task. Restricting adaptation to these directions provides a more "plastic" subspace for knowledge injection, with less risk of overwriting capabilities encoded in the dominant subspace. Empirically MiCA improves knowledge acquisition while reducing the trainable parameter footprint compared with LoRA at the same rank (because only `A` is trained, the parameter count is roughly halved for matching `r`).

Because `A == 0` at initialization, the adapter contribution `B · A == 0` and the model's forward output is preserved exactly at step 0 — no residual subtraction is needed on the base weight.

## Quick Start

```python
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id

lora_config = LoraConfig(
    init_lora_weights="mica",
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(model, lora_config)
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
peft_model.save_pretrained("mica-llama-2-7b")
```

To reload the trained adapter:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", dtype=torch.bfloat16, device_map="auto"
)
peft_model = PeftModel.from_pretrained(model, "mica-llama-2-7b")
```

## Notes and limitations

- MiCA currently supports `nn.Linear` and `nn.Embedding` target modules.
- The chosen rank must satisfy `r <= min(in_features, out_features)` for linear layers and `r <= min(num_embeddings, embedding_dim)` for embedding layers; otherwise initialization raises `ValueError`.
- MiCA performs a full SVD per target weight at initialization. For 7B-scale models this is a one-time cost of seconds; for substantially larger weight matrices (e.g. 70B-scale) the cost grows.
- Combining MiCA with `use_dora=True` or other LoRA variants is not supported in this initial integration.

## Citation

```
@article{rudiger2026mica,
  title={MiCA Learns More Knowledge Than LoRA and Full Fine-Tuning},
  author={R{\"u}diger, Sten and Raschka, Sebastian},
  journal={arXiv preprint arXiv:2604.01694},
  year={2026}
}
```
