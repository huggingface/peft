# PiSSA: Principal Singular values and Singular vectors Adaptation
## Introduction ([Paper](https://arxiv.org/abs/2404.02948), [code](https://github.com/GraphPKU/PiSSA))
PiSSA represents a matrix $W\in\mathbb{R}^{m\times n}$ within the model by the product of two trainable matrices $A \in \mathbb{R}^{m\times r}$ and $B \in \mathbb{R}^{r\times n}$, where $r \ll \min(m, n)$, plus a residual matrix $W^{res}\in\mathbb{R}^{m\times n}$ for error correction. Singular value decomposition (SVD) is employed to factorize $W$, and the principal singular values and vectors of $W$ are utilized to initialize $A$ and $B$. The residual singular values and vectors initialize the residual matrix $W^{res}$, which keeps frozen during fine-tuning. This straightforward modification allows PiSSA to converge more rapidly than LoRA and ultimately attain superior performance. Moreover, PiSSA reduces the quantization error compared to QLoRA, leading to further enhancements.

## Quick Start
```python
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id
lora_config = LoraConfig(
    # init_lora_weights="pissa", # Configure the initialization method to "pissa", which may take several minutes to execute SVD on the pre-trained model.
    init_lora_weights="pissa_niter_4", # Initialize the PiSSA with fast SVD, which completes in just a few seconds.
)
peft_model = get_peft_model(model, lora_config)

peft_model.print_trainable_parameters()

dataset = load_dataset("imdb", split="train[:1%]")

trainer = SFTTrainer(
    model=peft_model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=128,
    tokenizer=tokenizer,
)
trainer.train()
peft_model.save_pretrained("pissa-llama-2-7b")
```
When utilizing fast SVD, reducing the rank and the number of iterations decreases the time required. However, this approach leads to higher errors in the computed matrices $A$ and $B$. To preserve the model's initial capabilities, we calculate the residual matrix by $W^{res} = W - BA$. Even with potential errors in $A$ and $B$, the sum of $W^{res}$ and $BA$ accurately equals $W$.


To utilize the fine-tuned PiSSA modules, simply run the following command:
```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16, device_map="auto"
)
# Performs SVD again to initialize the residual model and loads the state_dict of the fine-tuned PiSSA modules.
peft_model = PeftModel.from_pretrained(model, "pissa-llama-2-7b")
```

## Advanced Usage

### Access the preprocessed models
We recommend downloading decomposed models directly from the [Hugging Face Collections](https://huggingface.co/collections/fxmeng/pissa-661ce700721235e542a5d7a8) instead of performing SVD every time.
If the existing models do not meet your needs, apply PiSSA initialization to a pre-trained model and store the decomposed model locally:
```bash
python preprocess.py \
    --base_model_name_or_path meta-llama/Llama-2-7b-hf \
    --init_lora_weights pissa \
    --output_dir pissa-llama-2-7b-r32-alpha-32 \
    --lora_r 32 \
    --lora_alpha 32 \
    --lora_dropout 0 \
    --bits bf16
```

### Convert PiSSA to LoRA
The main advantage of PiSSA is concentrated during the training phase. For a trained PiSSA adapter, we recommend converting it equivalently to the LoRA adapter for using and sharing.
```python
# The fine-tuned matrices $A$ and $B$ in PiSSA adapter is saved and should be combined with the residual model.
peft_model.save_pretrained(output_dir) 
# Given the matrices $A_0$ and $B_0$, initialized by PiSSA and untrained, and the trained matrices $A$ and $B$, 
# we can convert these to LoRA by setting $\Delta W = A \times B - A_0 \times B_0 = [A \mid A_0] \times [B \mid -B_0]^T = A'B'$.
peft_model.save_pretrained(output_dir, convert_pissa_to_lora="pissa_init")

```
This conversion enables the loading of LoRA on top of a standard base model:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16, device_map="auto"
)
# No SVD is performed during this step, and the base model remains unaltered.
peft_model = PeftModel.from_pretrained(model, "pissa-llama-2-7b-lora")
```
Utilizing the converted LoRA does not require modifying the parameters of the base model. When multiple converted LoRAs are needed simultaneously, each adapter operates independently without interference, allowing for the adapters to be freely deleted or added.



### Fine-tune in 4-bit or 8-bit
If quantization fine-tuning is desired, it is necessary to first decompose the original model at full precision and then reload the residual model in either 4-bit or 8-bit configurations.
```shell
python pissa_finetuning.py \
    --residual_model_name_or_path fxmeng/pissa-llama-2-7b-r16-alpha-16 \
    --output_dir output/pissa-llama-2-7b-r16-alpha-16-metamath-10k \
    --bits nf4 \
    --data_path meta-math/MetaMathQA \
    --dataset_split train[:100000] \
    --dataset_field query response \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --tf32 True \
    --report_to none \
    --convert_pissa_to_lora
```

This approach ensures the preservation of high-frequency, out-of-distribution parameters in the low-rank PiSSA modules, resulting in reduced quantization errors during the quantization of the residual model.

## Citation
```
@article{meng2024pissa,
  title={PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models},
  author={Meng, Fanxu and Wang, Zhaohui and Zhang, Muhan},
  journal={arXiv preprint arXiv:2404.02948},
  year={2024}
}
```