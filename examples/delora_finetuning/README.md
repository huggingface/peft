# DeLoRA: Decoupled Low-Rank Adaptation 

## Introduction
[DeLoRA](https://huggingface.co/papers/2503.18225) tackles finetuning in a Frobenius-norm bounded setup: this allows to prevent divergence from the pretrained model, effectively decoupling the learning of angles and magnitudes.

This is done by (i) normalization of the BA low-rank matrices, which bound the updates' Frobenius norm, (ii) learnable scaling lambda, which controls the update's boundary/magnitude, (iii) layer-wise scaling of ||W||, to adapt each update's norm to the original weights' norm.

## Quick start

With respect to your standard PEFT training procedure with LoRA, simply swap your `LoraConfig` for a `DeloraConfig`. Note however that `lora_alpha` parameter is replaced by `delora_lambda` parameter which sets an upper bound to the Frobenius norm of the weight change.

```python
import torch
from peft import DeloraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer.pad_token_id = tokenizer.eos_token_id
delora_config = DeloraConfig(r=32, delora_lambda=15)

peft_model = get_peft_model(model, delora_config)
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
peft_model.save_pretrained("delora-llama-3-8b")
```

To utilize the fine-tuned DeLoRA modules, simply run the following command:
```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B", dtype=torch.bfloat16, device_map="auto"
)
peft_model = PeftModel.from_pretrained(model, "delora-llama-3-8b")
```

## Advanced Usage
In this script the default DeLoRA layers are the query and value layers of the Llama model. Adding adapters on more layers will increase memory usage. If you wish to choose a different set of layers for DeLoRA to be applied on, you can simply define it using:
```bash
python examples/delora_finetuning/delora_finetuning.py --base_model meta-llama/Meta-Llama-3-8B --target_modules "q_proj,k_proj,v_proj,o_proj" 
```

Using different lambdas for different layers is also possible by setting `lambda_pattern`.

### Fine-tune
```bash
python delora_finetuning.py \
    --base_model "PATH_TO_MODEL" \
    --data_path "PATH_TO_DATASET" \
    --output_dir "PATH_TO_OUTPUT_DIR" \
    --batch_size 1 \
    --num_epochs 3 \
    --learning_rate 3e-3 \
    --cutoff_len 512 \
    --val_set_size 500 \
    --eval_step 10 \
    --save_step 100 \
    --device "auto" \
    --rank 32 \
    --delora_lambda 15 \
    --module_dropout 0.1 \
    --target_modules "q_proj,v_proj" \
    --hub_model_id "YOUR_HF_REPO" \
    --push_to_hub
```

## Additional Notes
### Best practices
- use 10-100x larger learning rate than standard LoRA variants (typical values from 1e-3/1e-2/..)
- do not set a too small initial boundary parameter lambda (typical values are around 10/15/..)


### DeLoRA vs DoRA
DeLoRA might feel quite similar to DoRA (given the similar target of decoupling angular from magnitude learning), however it presents key differences: (i) DoRA applies normalization and scaling operations on the fully finetuned weights ($W + \Delta W$), (ii) DoRA's normalization operation is performed on the column space of the weight matrices.

Conversely DeLoRA (i) introduces the normalization and scaling operations directly on the weight updates $\Delta W$, better preventing divergence from the pretrained model, and (ii) normalizes the inner low-dimensional space, which enforces a Frobenius-norm boundary to the weight updates.


## Citation
```
@inproceedings{bini2025decouplinganglesstrengthlowrank,
      title={Decoupling Angles and Strength in Low-rank Adaptation}, 
      author={Massimo Bini and Leander Girrbach and Zeynep Akata},
      year={2025},
  booktitle={International Conference on Learning Representations (ICLR)},
}
```
