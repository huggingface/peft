# EVA: Efficient Vector Adaptation
## Introduction ([Paper](https://arxiv.org/abs/2410.07170), [code](https://github.com/ml-jku/EVA))
Explained Variance Adaptation (EVA) is a novel initialization method for LoRA style adapters which initializes adapter weights in a data driven manner and adaptively allocates ranks according to the variance they explain. EVA improves average performance on a multitude of tasks across various domains, such as Language generation and understanding, Image classification, and Decision Making.

The abstract from the paper is:

*Foundation models (FMs) are pre-trained on large-scale datasets and then fine-tuned on a downstream task for a specific application. The most successful and most commonly used fine-tuning method is to update the pre-trained weights via a low-rank adaptation (LoRA). LoRA introduces new weight matrices that are usually initialized at random with a uniform rank distribution across model weights. Recent works focus on weight-driven initialization or learning of adaptive ranks during training. Both approaches have only been investigated in isolation, resulting in slow convergence or a uniform rank distribution, in turn leading to sub-optimal performance. We propose to enhance LoRA by initializing the new weights in a data-driven manner by computing singular value decomposition on minibatches of activation vectors. Then, we initialize the LoRA matrices with the obtained right-singular vectors and re-distribute ranks among all weight matrices to explain the maximal amount of variance and continue the standard LoRA fine-tuning procedure. This results in our new method **E**xplained **V**ariance **A**daptation (EVA). We apply EVA to a variety of fine-tuning tasks ranging from language generation and understanding to image classification and reinforcement learning. EVA exhibits faster convergence than competitors and attains the highest average score across a multitude of tasks per domain.*

## Quick Start
```python
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import EvaConfig, LoraConfig, get_peft_model, initialize_lora_eva_weights


# config
model_name = "meta-llama/Llama-3.1-8B"
max_seq_len = 512
rank = 16
alpha = 1
rho = 1.0
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
svd_batch_size = 4 # can be different from the batch size used in finetuning

# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# load dataset
dataset = load_dataset("Rowan/hellaswag")
dataset = dataset.map(
    lambda x: tokenizer(x["ctx"], padding="max_length", truncation=True, max_length=max_seq_len),
    batched=True,
    remove_columns=dataset["train"].column_names,
)
dataset.set_format(type="torch")

# create dataloader for SVD
# typically this is the same as the dataloader used for finetuning
dataloader = DataLoader(
    dataset["train"],
    batch_size=svd_batch_size,
    collate_fn=lambda examples: {k: torch.stack([v[k] for v in examples], dim=0) for k in examples[0].keys()},
)

# setup peft config
peft_config = LoraConfig(
    r=rank,
    lora_alpha=alpha,
    target_modules=target_modules,
    init_lora_weights="eva",
    eva_config=EvaConfig(rho=rho)
)

# move model to GPU
model = model.cuda()

# to optimize memory usage during eva initialization, set low_cpu_mem_usage=True
peft_model = get_peft_model(model, peft_config, low_cpu_mem_usage=True)

initialize_lora_eva_weights(peft_model, peft_config, dataloader)
```
`initialize_lora_eva_weights` will compute the SVD and load the components into the model. After this continue with standard LoRA finetuning.

## Using EVA with Bitsandbytes
EVA is fully compatible with bitsandbytes. Simply initialize the pretrained model with a BitsAndBytesConfig and then use the peft model with EVA.
```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True)
)
model = prepare_model_for_kbit_training(model)
peft_model = get_peft_model(model, peft_config)
initialize_lora_eva_weights(peft_model, peft_config, dataloader)
```

## Getting eva_state_dict without loading the adapter weights
In some cases you might just want to get the eva_state_dict without loading the adapter weights. This can be useful for example if:
- you want to precompute and store the eva_state_dict for different downstream tasks.
- you need to quantize the model for finetuning but want to perform EVA initialization with model weights in full/half precision.
- you do not intend to use a peft model for LoRA finetuning.

You can do this by calling `get_eva_state_dict` directly:
```python
from peft import get_eva_state_dict

eva_state_dict = get_eva_state_dict(model, peft_config, dataloader)
```

## EvaConfig

[[autodoc]] tuners.lora.config.EvaConfig

## initialize_lora_eva_weights

[[autodoc]] tuners.lora.eva.initialize_lora_eva_weights

## get_eva_state_dict

[[autodoc]] tuners.lora.eva.get_eva_state_dict

## Citation
In case you find our work useful, please consider citing it.

```	
@article{paischer2024eva,
    title={One Initialization to Rule them All: Fine-tuning via Explained Variance Adaptation}, 
    author={Fabian Paischer, Lukas Hauzenberger, Thomas Schmied, Benedikt Alkin, Marc Peter Deisenroth, Sepp Hochreiter},
    journal={arXiv preprint arXiv:2410.07170},
    year={2024}
}
```
