# CorDA: Context-Oriented Decomposition Adaptation of Large Language Models for Task-Aware Parameter-Efficient Fine-tuning

## Introduction


Existing PEFT methods are mostly agnoistic of the context of a task of concern, e.g., a downstream task to learn or some pre-trained world knowledge to maintain.
[CorDA](https://openreview.net/pdf?id=Gi00NVru6n) builds task-aware LoRA adapters from weight decomposition oriented by the context of the task concerned. 

Concretely, CorDA randomly collect a few (usually 256) data samples from a target task, e.g. questions from a QA dataset or instructions to write a code or solve a math problem, and feed these samples into a pre-trained LLM. We can obtain the covariance matrix of the input activation of each linear layer, i.e., $C=XX^T\in\mathcal{R}^{d_{in}\times d_{in}}$, where $X$ is the input of this linear layer. 
We then perform singular value decomposition (SVD) for the weight $W\in \mathcal{R}^{d_{out}\times d_{in}}$ multiplied by the covariance matrix, i.e., $\verb|SVD|(WC) = U\Sigma V^T$, where $U$ and $V$ are singular vectors and $\Sigma$ is the diagonal matrix with the singular values arranged in descending order. In this way, the context expressed by these representative covariance matrices is able to orientate the decomposition, such that the principal components are most associated with the task of concern. To ensure the same inference result with the pre-trained model at the start of adaptation, we multiply the inverse of these covariance matrices with the decomposed components, $\hat{W}=U\Sigma V^T C^{-1}$, where $\hat{W}$ is the weight after decomposition and reconstruction. 

Thanks to the task-awareness, CorDA enables two optional modes, **knowledge-preserving adaptation mode (KPM)** and **instruction-previewed adaptation mode (IPM**). In KPM, we use questions from question-answering dataset whose knowledge needs to be preserved, such as TriviaQA and NQopen, to obtain the covariance matrices. After our context-oriented decomposition, we use the components with the smallest $r$ singular values, $U_{[:,-r:]}$, $\Sigma_{[-r:]}$, and $(V^T C^{-1})_{[-r:,:]}$ to initialize the learnable LoRA adapters $A=\sqrt{\Sigma}_{[-r:]}(V^T C^{-1})_{[-r:,:]}$ and $B=U_{[:,-r:]}\sqrt{\Sigma}_{[-r:]}$. The other components that compact the question-answering ability are frozen during adaptation. 
KPM enables to learn new tasks effectively while keeping the world knowledge associated with $C$ as sound
as possible.
Alternatively, when one only aims to achieve performance as high as possible on the finetuning task without concern for world knowledge maintenance, our IPM will be favored. 
In this mode, CorDA uses the instruction and response from the fine-tuning task (e.g., Math or Code) to produce the covariance matrices. The principal components with large singular values capturing the characteristics of the finetuning task in advance can better accommodate the new ability. So we initialize adapters as $A= \sqrt{\Sigma}_{[:r]} (V^T C^{-1})_{[:r,:]}$ and $B =U_{[:,:r]} \sqrt{\Sigma}_{[:r]}$, and freeze the remaining components. The implementations of KPM and IPM are compared as follows:

| Mode | Collect covariance from | LoRA $A$ | LoRA $B$ |
|---|---|---|---
|KPM | questions from a knowledge benchmark to maintain | $A=\sqrt{\Sigma}_{[-r:]}(V^T C^{-1})_{[-r:,:]}$ | $B=U_{[:,-r:]}\sqrt{\Sigma}_{[-r:]}$ |
IPM | instructions and responses from a downstream task to learn | $A= \sqrt{\Sigma}_{[:r]} (V^T C^{-1})_{[:r,:]}$ | $B =U_{[:,:r]} \sqrt{\Sigma}_{[:r]}$ |



## Quick Start

- Knowledge-preserving adaptation mode

```py
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft.tuners.lora.config import CordaConfig
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


corda_config = CordaConfig(
    run_model=run_model,
    sample_count=256,
    corda_method="kpm",
)
lora_config = LoraConfig(
    init_lora_weights="corda",
)
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
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft.tuners.lora.config import CordaConfig
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id
dataset = load_dataset("imdb", split="train[:256]")


def run_model():
    for batch in dataset:
        input_ids = batch["text"]
        input_ids = input_ids.to(model.device)
        with torch.no_grad():
            model(input_ids)


corda_config = CordaConfig(
    run_model=run_model,
    sample_count=256,
    corda_method="ipm",
)
lora_config = LoraConfig(
    init_lora_weights="corda",
)
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

## Citation
```
@inproceedings{yangcorda,
  title={CorDA: Context-Oriented Decomposition Adaptation of Large Language Models for Task-Aware Parameter-Efficient Fine-tuning},
  author={Yang, Yibo and Li, Xiaojie and Zhou, Zhongzhu and Song, Shuaiwen Leon and Wu, Jianlong and Nie, Liqiang and Ghanem, Bernard},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
}
```