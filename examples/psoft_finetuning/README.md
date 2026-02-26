# Efficient Orthogonal Fine-Tuning with Principal Subspace Adaptation (PSOFT) 
## Introduction ([Paper](https://huggingface.co/papers/2505.11235), [code](https://github.com/fei407/PSOFT))
PSOFT aims to preserve the geometric relationships among pre-trained weight column vectors—a core principle of OFT—while achieving a balanced trade-off across parameter, computation, and memory efficiency. Unlike existing OFT variants (e.g., OFTv2, BOFT, and GOFT) that rely on sparsity-based designs, PSOFT adopts a low-rank principal subspace perspective, bridging the gap between LoRA and OFT. PSOFT confines orthogonal fine-tuning to a principal subspace, offering theoretical guarantees via orthogonality constraints on the down-projection matrix, while enabling practical adaptability through two low-dimensional tunable vectors.


## Quick Start
```python
import torch
from peft import PsoftConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

model_name = "facebook/opt-125m"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

psoft_config = PsoftConfig(
    r=32,
    psoft_alpha=32,
)

peft_model = get_peft_model(model, psoft_config)
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
peft_model.save_pretrained("psoft-opt-125m")
```

## Best Practices
1. **Rank Choice**: Smaller ranks (e.g., `32–128`) are suitable for simpler tasks, while larger ranks (e.g., `64–256`) provide greater expressiveness for more complex tasks at the cost of increased parameters and computation.
2. **Scaling Factor**: The scaling factor is typically set to $r$ in PSOFT.
3. **Learning Rate**: Use standard learning rates (e.g., `1e-4` to `5e-3`) for stable training.
4. **SVD Initialization**: The `lowrank` option is more memory- and compute-efficient than `full`, making it more suitable for large models.
5. **Cayley–Neumann Approximation**: When the rank is large, enabling the Cayley–Neumann approximation can significantly improve computational efficiency, while the benefit is less pronounced for small ranks. In practice, a small number of Neumann series terms (typically `5`) usually provides a good balance between accuracy and efficiency.

```shell
python examples/psoft_finetuning/psoft_finetuning.py \
  --base_model_name_or_path meta-llama/Llama-3.2-3B \
  --output_dir ./outputs/psoft-llama3.2-3b-imdb \
  --data_path imdb \
  --dataset_split "train[:1%]" \
  --max_length 128 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --bits bf16 \
  --r 128 \
  --psoft_alpha 128 \
  --target_modules q_proj v_proj
```

## Citation
```
@inproceedings{wu2026efficient,
title={Efficient Orthogonal Fine-Tuning with Principal Subspace Adaptation},
author={Wu, Fei and Hu, Jia and Min, Geyong and Wang, Shiqiang},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=FSHrinMArK}
}
```



