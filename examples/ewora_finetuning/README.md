# EWoRA: Expert Weighted Low-Rank Adaptation

## Introduction

[EWoRA](https://aclanthology.org/2025.findings-ijcnlp.108/) is a LoRA variant designed for finetuning on heterogeneous data. Instead of a single low-rank adapter, EWoRA uses `num_experts` independent low-rank expert adapters and learns a lightweight routing matrix that dynamically weights the experts for each input. This lets a single adapter capture the diverse expertise needed across a heterogeneous corpus (e.g. mixed domains or tasks) while keeping a LoRA-like parameter budget.

Note that `r` is the rank of each individual expert, so the LoRA-equivalent total rank is `r * num_experts`. Because the expert weighting depends on the input, EWoRA adapters cannot be merged into the base weights; use `unload()` to recover the base model without the adapter.

## Quick start

With respect to your standard PEFT training procedure with LoRA, simply swap your `LoraConfig` for an `EworaConfig`:

```python
import torch
from peft import EworaConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token_id = tokenizer.eos_token_id
ewora_config = EworaConfig(r=8, num_experts=4, task_type="CAUSAL_LM")

peft_model = get_peft_model(model, ewora_config)
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
peft_model.save_pretrained("ewora-mistral-7b")
```

To use the fine-tuned EWoRA adapter, simply run:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", dtype=torch.bfloat16, device_map="auto"
)
peft_model = PeftModel.from_pretrained(model, "ewora-mistral-7b")
```

## Reproducing the paper's mixed-task setting

`ewora_finetuning.py` reproduces the "Mixed" setting of Table 1 in the [paper](https://aclanthology.org/2025.findings-ijcnlp.108/), where EWoRA outperforms LoRA and other baselines across all test splits. Mistral 7B is finetuned on a simple concatenation (shuffled at train time) of four datasets spanning three domains:

- Code: [Magicoder-Evol-Instruct-110K](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K) (evaluated on HumanEval)
- Math: [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) (evaluated on GSM8K)
- General reasoning: [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag) and [Winogrande](https://huggingface.co/datasets/allenai/winogrande) (evaluated on their test splits)

The default hyperparameters of the script follow the paper:

| Hyperparameter | Value |
|---|---|
| Total rank (`r * num_experts`) | 32 (paper's `r=32`, partitioned into `n=4` experts, i.e. `r=8`, `num_experts=4`) |
| Target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| Epochs | 4 |
| Learning rate | 1e-4 (mixed setting), cosine annealing with `min_lr_rate=0.1` |
| Warmup ratio | 0.1 |
| Weight decay | 0 |
| Sequence length | 1024 (with packing) |
| Optimizer | AdamW |
| Precision | bf16 |

To launch the run (defaults reproduce the paper setting):

```bash
python ewora_finetuning.py --output_dir outputs/ewora_mixed
```

For multi-GPU training with DDP (the paper's runs used 4 GPUs):

```bash
torchrun --nproc_per_node 4 ewora_finetuning.py --output_dir outputs/ewora_mixed --gradient_checkpointing
```

Evaluation in the paper uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for GSM8K/HellaSwag/Winogrande (5-shot, temperature 0) and [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) for HumanEval (0-shot, temperature 0.2, top_p 0.95, n_samples 50).

## Additional Notes

- EWoRA shines when the finetuning corpus is heterogeneous (mixed domains, tasks, or styles); on a single narrow task it generally matches LoRA at the same parameter budget.
- There is no `alpha` parameter: the learned routing scores act as an input-dependent scaling of the expert outputs, taking the place of LoRA's static `alpha/r` factor.
- EWoRA adapters cannot be merged into the base model (`merge_and_unload()` raises an error); keep the adapter loaded for inference, or use `unload()` to drop it.

## Citation

```
@inproceedings{kohli-etal-2025-ewora,
    title = "{EW}o{RA}: Expert Weighted Low-Rank Adaptation for Heterogeneous Data",
    author = "Kohli, Harsh  and
      Feng, Helian  and
      Minorics, Lenon  and
      Vasani, Bhoomit  and
      He, Xin  and
      Kebarighotbi, Ali",
    booktitle = "Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics",
    year = "2025",
    url = "https://aclanthology.org/2025.findings-ijcnlp.108/",
    doi = "10.18653/v1/2025.findings-ijcnlp.108",
}
```
