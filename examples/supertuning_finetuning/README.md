# Super-Tuning / Supra

## Introduction
[Super-Tuning](https://huggingface.co/papers/2607.09287) is a sparse fine-tuning method that freezes the base weight and trains only a small support of individual scalar weight entries, selected by weight magnitude (data-free, no calibration pass). Unlike LoRA, the trainable set is not restricted to a low-rank subspace. Setting `r` additionally allocates a LoRA-style low-rank adapter composed additively on top of the sparse support — the paper's "Supra" hybrid.

## Quick start

With respect to your standard PEFT training procedure with LoRA, simply swap your `LoraConfig` for a `SupertuningConfig`. The `sparsity` argument controls the fraction of frozen entries: `sparsity=0.99` trains 1% of each target weight. Leave `r=None` for pure Super, or set it to a positive integer for the Supra hybrid.

```python
import torch
from peft import SupertuningConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token_id = tokenizer.eos_token_id
supertuning_config = SupertuningConfig(sparsity=0.99, target_modules=["q_proj", "v_proj"])

peft_model = get_peft_model(model, supertuning_config)
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
peft_model.save_pretrained("supertuning-llama-3.2-1b")
```

To utilize the fine-tuned Super-Tuning modules, simply run the following command:
```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", dtype=torch.bfloat16, device_map="auto"
)
peft_model = PeftModel.from_pretrained(model, "supertuning-llama-3.2-1b")
```

## Advanced Usage
By default this script applies Super-Tuning to the query and value layers of the model. To target a different set of layers, pass a comma-separated list:
```bash
python examples/supertuning_finetuning/supertuning_finetuning.py --base_model meta-llama/Llama-3.2-1B --target_modules "q_proj,k_proj,v_proj,o_proj"
```

To train the Supra hybrid (sparse support + LoRA), pass `--rank`; `--lora_alpha` defaults to `2 * rank` when omitted:
```bash
python examples/supertuning_finetuning/supertuning_finetuning.py --base_model meta-llama/Llama-3.2-1B --rank 8
```

### Fine-tune
```bash
python supertuning_finetuning.py \
    --base_model "PATH_TO_MODEL" \
    --data_path "PATH_TO_DATASET" \
    --output_dir "PATH_TO_OUTPUT_DIR" \
    --batch_size 1 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --eval_step 10 \
    --save_step 100 \
    --device "auto" \
    --sparsity 0.99 \
    --rank 8 \
    --target_modules "q_proj,v_proj" \
    --hub_model_id "YOUR_HF_REPO" \
    --push_to_hub
```

## Additional Notes
- `sparsity` must be in `[0.0, 1.0)`. Very high values leave very few trainable entries; a sparsity so high that no entry is selected raises an error.
- `select_top=True` (default) keeps the largest-magnitude entries (paper's Super/Supra); `select_top=False` keeps the smallest (the paper's `-bottom` variants). The best direction is model- and task-dependent.
- Only `nn.Linear` layers are currently supported.

## Citation
```
@article{ilin2026supertuning,
      title={Super-Tuning: From Activation-Aware Pruning to Sparse Fine-Tuning},
      author={Ivan Ilin and Philip Zmushko and Peter Richt\'arik},
      year={2026},
      eprint={2607.09287},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
}
```
