# EVA: Explained Variance Adaptation
## Introduction ([Paper](https://huggingface.co/papers/2410.07170), [code](https://github.com/ml-jku/EVA))
Explained Variance Adaptation (EVA) is a novel initialization method for LoRA style adapters which initializes adapter weights in a data driven manner and adaptively allocates ranks according to the variance they explain. EVA improves average performance on a multitude of tasks across various domains, such as Language generation and understanding, Image classification, and Decision Making.

The abstract from the paper is:

*Foundation models (FMs) are pre-trained on large-scale datasets and then fine-tuned on a downstream task for a specific application. The most successful and most commonly used fine-tuning method is to update the pre-trained weights via a low-rank adaptation (LoRA). LoRA introduces new weight matrices that are usually initialized at random with a uniform rank distribution across model weights. Recent works focus on weight-driven initialization or learning of adaptive ranks during training. Both approaches have only been investigated in isolation, resulting in slow convergence or a uniform rank distribution, in turn leading to sub-optimal performance. We propose to enhance LoRA by initializing the new weights in a data-driven manner by computing singular value decomposition on minibatches of activation vectors. Then, we initialize the LoRA matrices with the obtained right-singular vectors and re-distribute ranks among all weight matrices to explain the maximal amount of variance and continue the standard LoRA fine-tuning procedure. This results in our new method **E**xplained **V**ariance **A**daptation (EVA). We apply EVA to a variety of fine-tuning tasks ranging from language generation and understanding to image classification and reinforcement learning. EVA exhibits faster convergence than competitors and attains the highest average score across a multitude of tasks per domain.*

## Quick Start
Below is an example of how to use EVA with a causal language model. For a more detailed example see [eva_finetuning.py](https://github.com/huggingface/peft/blob/main/examples/eva_finetuning/eva_finetuning.py).
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
rho = 2.0
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
eva_config = EvaConfig(
    rho=rho
)
peft_config = LoraConfig(
    r=rank,
    lora_alpha=alpha,
    target_modules=target_modules,
    init_lora_weights="eva",
    eva_config=eva_config
)

# move model to GPU
model = model.cuda()

# to optimize memory usage during EVA initialization, set low_cpu_mem_usage=True
peft_model = get_peft_model(model, peft_config, low_cpu_mem_usage=True)

initialize_lora_eva_weights(peft_model, dataloader)
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
initialize_lora_eva_weights(peft_model, dataloader)
```

## Getting the EVA state_dict without loading the adapter weights
In some cases you might just want to get the state_dict after EVA initialization without loading the adapter weights. This can be useful for example if:
- you want to precompute and store the state_dict for different downstream tasks.
- you need to quantize the model for finetuning but want to perform EVA initialization with model weights in full/half precision.
- you do not intend to use a peft model for LoRA finetuning.
- you would like to leverage multiple GPUs for EVA initialization. (At the moment this is not directly supported by `initialize_lora_eva_weights`)

You can do this by calling `get_eva_state_dict` directly (you only need to pass `peft_config` if `model` is not a PeftModel):
```python
from peft import get_eva_state_dict

eva_state_dict = get_eva_state_dict(model, dataloader, peft_config)
```
Later you can load the state_dict into a `PeftModel` by using the `eva_state_dict` argument in `initialize_lora_eva_weights`:
```python
initialize_lora_eva_weights(peft_model, eva_state_dict=eva_state_dict)
```

## Leveraging multiple GPUs

EVA initialization can be parallelized across multiple GPUs. In this case inputs from multiple GPUs are gathered before computing the SVD for the batch. This requires that the model is wrapped in a `torch.nn.DataParallel` or `torch.nn.DistributedDataParallel` class. An example of how to use this can be found in [eva_finetuning_multi_gpu.py](https://github.com/huggingface/peft/blob/main/examples/eva_finetuning/eva_finetuning_multi_gpu.py).

## Customizing EVA

By default, EVA is designed to work with standard transformer language models. However we integrated three different parameters which can be used to customize EVA for other types of models.
1. `forward_fn`: Defines how the forward pass during EVA initialization should be computed.
2. `prepare_model_inputs_fn`: Can be used if it is necessary to use information contained in the original model_input to prepare the input for SVD in individual layers.
3. `prepare_layer_inputs_fn`: Defines how layer inputs should be prepared for SVD.

All three parameters can be passed to `initialize_lora_eva_weights` and `get_eva_state_dict`.

### forward_fn

`forward_fn` defines how the forward pass during EVA initialization should be computed. `forward_fn` receives two arguments: `model` and `inputs`. By default this is set to `forward_fn_dict` which simply returns `model(**inputs)`.

### prepare_model_inputs_fn

`prepare_model_inputs_fn` can be used if it is necessary to use information contained in the original model_input to prepare the input for SVD in individual layers. `prepare_model_inputs_fn` receives two arguments: `model_input` and `peft_config`. This component is separate from `prepare_layer_inputs_fn` as the output only needs to be computed once per batch. By default this parameter is set to `prepare_model_inputs_fn_language_modeling` which is used get a subset of indices based on attention and label mask to avoid including padding tokens in the SVD computation. If you would like to not use this component set `prepare_model_inputs_fn` to None. The default logic is:
```python
def prepare_model_inputs_fn_language_modeling(model_input, peft_config: LoraConfig):
    mask = model_input.get("attention_mask", torch.ones_like(model_input["input_ids"])).bool()
    if peft_config.eva_config.use_label_mask and hasattr(model_input, "labels"):
        mask = torch.logical_and(mask, model_input["labels"] != peft_config.eva_config.label_mask_value)
    return mask.nonzero()
```

### prepare_layer_inputs_fn

`prepare_layer_inputs_fn` can be used to preprocess the layer inputs before passing them to the SVD algorithm. `prepare_layer_inputs_fn` receives three arguments: `layer_input`, `model_input` and `layer_name`. It can either be a callable or a dictionary where the keys are the layer names and the values are callables. If it is a dictionary, functions are assigned to adapter layers based on the layer names. By default a language modeling setting is assumed where model_inputs are the outputs of `prepare_model_inputs_fn_language_modeling` which is a mask of indices. If this parameter is set to None, only two modifications are made to the layer inputs
- take the first element incase of a tuple or list. 
- if the input has more than 2 dimensions, we flatten all but the last dimension.

Must always return a tensor. The default logic is:
```python
def prepare_layer_inputs_fn_default(layer_input, model_input, layer_name) -> torch.Tensor:
    if isinstance(layer_input, (tuple, list)):
        layer_input = layer_input[0]
    return layer_input[model_input.T.unbind()]
```

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
