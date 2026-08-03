# DEFT: Decompositional Efficient Fine-Tuning

## Introduction
[DEFT](https://proceedings.neurips.cc/paper_files/paper/2025/hash/93a34a7138bdad95e874018d5f491cc6-Abstract-Conference.html) adapts a frozen weight `W` by **removing** a learned rank-`r` sub-space and **injecting** a low-rank update in its place: `W' = (I - P_proj) @ W + Q_P @ R`. Unlike a purely additive update (LoRA's `W + B @ A`), the removal term lets DEFT re-purpose existing weight directions, which helps it learn new data/concepts while keeping the base model's capabilities (low forgetting). With the default identity initialization the adapter is an exact no-op at the start of training, and the update merges into the base weights for inference.

## Quick start

With respect to your standard PEFT training procedure with LoRA, simply swap your `LoraConfig` for a `DeftConfig`. DEFT uses `alpha` for the LoRA-style injection scaling (`alpha / r`) and `decomposition_method` (`"relu"` default, or `"qr"`) to derive the projector.

```python
import torch
from peft import DeftConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer.pad_token_id = tokenizer.eos_token_id
deft_config = DeftConfig(r=32, alpha=64, decomposition_method="relu")

peft_model = get_peft_model(model, deft_config)
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
peft_model.save_pretrained("deft-llama-3-8b")
```

To utilize the fine-tuned DEFT modules, simply run the following command:
```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B", dtype=torch.bfloat16, device_map="auto"
)
peft_model = PeftModel.from_pretrained(model, "deft-llama-3-8b")
```

## Advanced Usage
By default DEFT is applied to the query and value layers. Adding adapters on more layers will increase memory usage. To choose a different set of layers:
```bash
python examples/deft_finetuning/deft_finetuning.py --base_model meta-llama/Meta-Llama-3-8B --target_modules "q_proj,k_proj,v_proj,o_proj"
```

DEFT supports `torch.nn.Linear` and `Conv1D` (e.g. gpt-2) layers. The `qr` decomposition gives an orthogonal projection, and `para=True` selects the removal-only [PaRa](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f09e8dd9274cb7c2dd0dc65ffc6f427a-Abstract-Conference.html) variant.

### Fine-tune
```bash
python deft_finetuning.py \
    --base_model "PATH_TO_MODEL" \
    --data_path "PATH_TO_DATASET" \
    --output_dir "PATH_TO_OUTPUT_DIR" \
    --batch_size 1 \
    --num_epochs 3 \
    --learning_rate 3e-4 \
    --cutoff_len 512 \
    --val_set_size 500 \
    --eval_step 10 \
    --save_step 100 \
    --device "auto" \
    --rank 32 \
    --alpha 64 \
    --decomposition_method "relu" \
    --deft_dropout 0.05 \
    --target_modules "q_proj,v_proj" \
    --hub_model_id "YOUR_HF_REPO" \
    --push_to_hub
```

## Citation
```
@article{kumar2026deft,
  title={DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models},
  author={Kumar, Komal and Anwer, Rao and Shahbaz Khan, Fahad and Khan, Salman and Laptev, Ivan and Cholakkal, Hisham},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={102009--102035},
  year={2026}
}
```
