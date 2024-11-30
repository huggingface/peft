# CorDA: Context-Oriented Decomposition Adaptation of Large Language Models for Task-Aware Parameter-Efficient Fine-tuning

## Introduction


Existing PEFT methods are mostly agnostic of the context of a task of concern, e.g., a downstream task to learn or some pre-trained world knowledge to maintain.
[CorDA](https://openreview.net/pdf?id=Gi00NVru6n) builds task-aware LoRA adapters from weight decomposition oriented by the context of the task concerned. 

Concretely, CorDA randomly collects a few (by default 256) data samples from a target task, e.g. questions from a QA dataset or instructions to write a code or solve a math problem, and feeds these samples into a pre-trained LLM. We can obtain the covariance matrix of the input activation of each linear layer, i.e., $C=XX^T\in\mathcal{R}^{d_{in}\times d_{in}}$. 
We then perform singular value decomposition (SVD) for the weight $W\in \mathcal{R}^{d_{out}\times d_{in}}$ multiplied by the covariance matrix, i.e., $\verb|SVD|(WC) = U\Sigma V^T$. In this way, the context expressed by these representative covariance matrices is able to orientate the decomposition, such that the principal components (the singular vectors with the largest singular values) are most associated with the task of concern (please refer to Fig.2 of our paper for the advantage of our decomposition over the plain SVD). To ensure the same inference result with the pre-trained model at the start of adaptation, we multiply the inverse of these covariance matrices with the decomposed components, i.e., $\hat{W}=U\Sigma V^T C^{-1}$. 

Thanks to the task-awareness, you can choose how to utilize the task-specific principal components. For examples, if you want to adapt a model to a new task without losing the knowledge of a question-answering dataset, e.g., TriviaQA and NQopen, you can sample questions from this dataset to collect covariance matrices, and keep the principal components frozen because they compact the ability of this dataset, while using the lowest components with the smallest $r$ singular values to initialize the learnable LoRA adapters. This is achieved by the **knowledge-preserved mode (KPM)** of CorDA, which learns new tasks effectively while keeping the world knowledge you are concerned about as sound as possible. Alternatively, when your primary objective is to maximize performance on the finetuning task, disregarding the preservation of world knowledge, the **instruction-previewed mode (IPM**) will be favored. In this mode, CorDA uses the instruction and response from the fine-tuning task (e.g., Math or Code) to produce the covariance matrices. The principal components with the largest $r$ singular values, capturing the characteristics of the finetuning task in advance, can better adapt to the new ability, so they are used to initialize the LoRA adapters, with the remaining components frozen. IPM can further accelerate convergence to enhance the fine-tuning performance on downstream tasks.


The implementations of KPM and IPM are compared as follows:

| Mode | Collect covariance from | LoRA $A$ | LoRA $B$ |
|---|---|---|---
|KPM | questions from the knowledge benchmark to maintain | $A=\sqrt{\Sigma}\_{[-r:]}(V^T C^{-1})\_{[-r:,:]}$ | $B=U_{[:,-r:]}\sqrt{\Sigma}_{[-r:]}$ |
IPM | instructions and responses from the downstream task to learn | $A= \sqrt{\Sigma}\_{[:r]} (V^T C^{-1})\_{[:r,:]}$ | $B =U_{[:,:r]} \sqrt{\Sigma}_{[:r]}$ |

### Comparison with alternative methods

The distinction between CorDA with other similar LoRA initialization methods is summarized as follows:

| Method | Initialization for | SVD on | Data-driven | Support knowledge maintenance |
| - | - | - | - | - |
| PiSSA | $A$ and $B$ | weights | no | no |
| EVA | $A$ | activations | yes | no |
|CorDA |  $A$ and $B$ | weights (oriented by covariance) | yes | yes |

### Some Results

- Performance with knowledge-preserved mode (sample from NQopen, fine-tune on Math)

| Method | Model | NQ open | GSM8k | Math | Avg. |
|---|---|---|---|---|---|
|Pre-trained|Llama-2-7b| 14.99 | -| - | - |
|LoRA|Llama-2-7b|1.27| 42.68 | 5.88 | 16.61 |
|**CorDA (KPM)** |Llama-2-7b| **8.20** | **46.32**	| **7.00** | **20.51** |
|Pre-trained|Llama-2-13b|23.63|-|-|-|
|LoRA|Llama-2-13b| 16.26 | 57.24 | 8.92 | 27.47 |
|**CorDA (KPM)** |Llama-2-13b| **19.86** | **59.29** | **9.62** | **29.59** |
|Pre-trained|Llama-3-8b|13.41|-|-|-|
|LoRA|Llama-3-8b| 8.75 | 72.33 | 24.04| 35.04 |
|**CorDA (KPM)** |Llama-3-8b| **9.61** | **74.68** | **25.34** | **36.54** |
|Pre-trained|Gemma-2-9b|12.85|-|-|-|
|LoRA|Gemma-2-9b| 9.28 | 83.47 | 42.30| 45.02 |
|**CorDA (KPM)** |Gemma-2-9b|**10.17** | **84.08** | **42.64** | **45.63** |

- Performance with instruction-previewed mode (sample from Math, fine-tune on Math)

| Method | Model | GSM8k | Math |
| --- | --- | --- | ---|
|LoRA| Llama-2-7b | 42.68 | 5.88 |
|PiSSA | Llama-2-7b | 51.63 | 7.32 |
| **CorDA (IPM)** | Llama-2-7b | **53.45** | **8.64** |
|LoRA| Llama-2-13b | 57.24 | 8.92 |
|PiSSA | Llama-2-13b |60.88	| 11.08|
| **CorDA (IPM)** | Llama-2-13b | **62.47** |**11.54** |
|LoRA| Gemma-2-9b | 83.47 |	42.30 |
|PiSSA | Gemma-2-9b | 84.23	| 43.52|
| **CorDA (IPM)** | Gemma-2-9b | **84.45** | **43.88** |


## Quick Start

- Knowledge-preserved adaptation mode

```py
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft.tuners.lora.config import CordaConfig, CordaInitConfig
from peft.tuners.lora.corda import preprocess_corda
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id
sampled_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:256]")
dataset = load_dataset("imdb", split="train[:256]")


def run_model():
    for batch in sampled_dataset:
        input_ids = batch["text"]
        input_ids = input_ids.to(model.device)
        with torch.no_grad():
            model(input_ids)


init_config = CordaInitConfig(
    run_model=run_model,
)
corda_config = CordaConfig(
    sample_count=256,
    corda_method="kpm",
)
lora_config = LoraConfig(
    init_lora_weights="corda",
    corda_config=corda_config,
)
preprocess_corda(model, lora_config, init_config)
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

training_args = SFTConfig(dataset_text_field="text", max_seq_length=128)
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
peft_model.save_pretrained("corda-llama-2-7b")
```

- Instruction-previewed adaptation mode

```py
# Get model and dataset identically as KPM...

# Different from KPM, we run the model on dataset of the downstream task to collect covariance matrices
def run_model():
    for batch in dataset:
        input_ids = batch["text"]
        input_ids = input_ids.to(model.device)
        with torch.no_grad():
            model(input_ids)

# Different from KPM, we set `corda_method` to `"ipm"`
corda_config = CordaConfig(
    corda_method="ipm",
)

# The rest of training process is identical to KPM...
```

## Advanced Usage

`preprocess.py`: This script builds CorDA adapters for a model, and saves the adapters initial weights and residual model weights to a specified directory. Example usage:

```bash
CUDA_VISIBLE_DEVICES=0 python -u preprocess.py --model_id="meta-llama/Llama-2-7b-hf" \
    --r 128 --seed 233 \
    --save_model --save_path {your_save_path} \
    --first_eigen --calib_dataset "MetaMATH"
```

`corda_finetuning.py`: This script fine-tunes the preprocessed model built above on a downstream task.

Example usage:

```bash
python corda_finetuning.py \
    --model_name_or_path {path_to_residual_model} \
    --output_dir {path_to_output_model} \
    --corda_mode True \
    --data_path meta-math/MetaMathQA \
    --dataset_split "train[:100000]" \
    --dataset_field query response \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --report_to none
```

## Citation
```
@inproceedings{yangcorda,
  title={CorDA: Context-Oriented Decomposition Adaptation of Large Language Models for Task-Aware Parameter-Efficient Fine-tuning},
  author={Yang, Yibo and Li, Xiaojie and Zhou, Zhongzhu and Song, Shuaiwen Leon and Wu, Jianlong and Nie, Liqiang and Ghanem, Bernard},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
}
```