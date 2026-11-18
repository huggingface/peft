# Astra: Activation-Space Tail-Eigenvector Low-Rank Adaptation of Large Language Models

## Introduction

Most LoRA initialization schemes are agnostic of the activation subspaces that a downstream task actually uses.
[Astra](https://arxiv.org/abs/2602.19111) builds task-aware LoRA adapters from the tail eigenvectors of the covariance
matrix of the module output activations, which are estimated from a small calibration set of the downstream task.

Concretely, Astra feeds a few data samples of the target task into the pre-trained LLM and collects the covariance
matrix of the **output** activations of each targeted linear layer, i.e. $C=YY^\top\in\mathbb{R}^{d_{out}\times
d_{out}}$, where $Y$ denotes the output activations. An eigendecomposition $C=Q\Lambda Q^\top$ is then performed.
The pretrained weight $W\in\mathbb{R}^{d_{out}\times d_{in}}$ is projected onto the subspace spanned by the tail
eigenvectors $Q_{[:,-r:]}$, and the projection is used to initialize the adapter while being subtracted from the
frozen residual weight, i.e. $A=Q_{[:,-r:]}^\top W$, $B=Q_{[:,-r:]}$ and $W_{res}=W-\frac{\alpha}{r}BA$. This keeps
the model output unchanged at the start of adaptation and constrains the update to the task-relevant activation
subspace, which speeds up convergence and improves downstream performance.

For more details, please refer to the [Astra paper](https://arxiv.org/abs/2602.19111) and the
[reference implementation](https://github.com/LyoAI/Astra).

## Quick Start
```py
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft.tuners.lora import AstraConfig, preprocess_astra
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id
dataset = load_dataset("imdb", split="train[:256]")


def run_model():
    for batch in dataset:
        input_ids = tokenizer(batch["text"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**input_ids)


astra_config = AstraConfig()
lora_config = LoraConfig(
    init_lora_weights="astra",
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    astra_config=astra_config,
)

# Call `preprocess_astra` first to collect the covariance matrix and build the eigendecomposition for the model
# For more details, please refer to documentation of `preprocess_astra`
preprocess_astra(model, lora_config, run_model=run_model)

# Call `get_peft_model` after preprocessing, or else you'll encounter error
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

training_args = SFTConfig(dataset_text_field="text", max_length=128)
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
peft_model.save_pretrained("astra-llama-2-7b")
```

The example scripts are adapted from the [CorDA example](https://github.com/huggingface/peft/tree/main/examples/corda_finetuning).
The whole pipeline can be run as follows:

```sh
# Preprocess the model and build the Astra initialization
python preprocess.py --model_id meta-llama/Llama-2-7b-hf --calib_dataset wikitext2 --r 128 --save_model --save_path ./astra-llama-2-7b

# Finetune with Astra
accelerate launch astra_finetuning.py --model_name_or_path ./astra-llama-2-7b --astra_mode True
```

### Convert Astra to a standard LoRA adapter

Astra preprocessing modifies the base weights, so a trained Astra adapter must be loaded with its residual model.
For sharing or deployment, convert the trained adapter to a standard LoRA adapter.
The converted adapter works with the original base model, other adapters, and LoRA-compatible tools such as llama.cpp.

The conversion uses the untrained adapter saved by `preprocess.py` before training:

```python
# Saved by preprocess.py before training.
model.peft_config["default"].init_lora_weights = True
model.save_pretrained(os.path.join(save_path, "astra_init"))
```

```python
# After training, create a standard LoRA adapter.
peft_model.save_pretrained(
    lora_output_dir,
    path_initial_model_for_weight_conversion=os.path.join(save_path, "astra_init"),
)
```

The converted adapter can be loaded on top of the original base model:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", dtype=torch.bfloat16, device_map="auto"
)
# No eigendecomposition is performed during this step, and the base model remains unaltered.
peft_model = PeftModel.from_pretrained(model, "astra-llama-2-7b-lora")
```

Note that this conversion is not supported if `rslora` is used in combination with `rank_pattern` or `alpha_pattern`.

## Citation
```
@inproceedings{liuastra,
  title={Astra: Activation-Space Tail-Eigenvector Low-Rank Adaptation of Large Language Models},
  author={Liu, Kainan and Zhang, Yong and Cheng, Ning and Zhu, Yun and Wang, Yanmeng and Wang, Shaojun and Xiao, Jing},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  year={2026},
}
```
