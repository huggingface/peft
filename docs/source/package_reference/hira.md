# HiRA

High-Rank Adaptation ([HiRA](https://openreview.net/pdf?id=TwJrTz9cRS)) is a PEFT method that extends the LoRA approach by applying an element-wise modulation on the original weight matrix. Instead of adding a low-rank update directly, HiRA computes:

$$
W' = W_0 + W_0 \odot (B A)
$$

where $W_0$ is the base weight, and $A, B$ are low-rank factors with rank $r \ll \min(	\text{in_features}, \text{out_features})$. This formulation allows HiRA to adapt existing weights with a multiplicative, input-dependent modulation, often improving fine-tuning efficiency on downstream tasks.

The abstract from the HiRA paper is:

> *We propose Hadamard High-Rank Adaptation (HiRA), a parameter-efficient fine-tuning (PEFT) method that enhances the adaptability of Large Language Models (LLMs). While Low-rank Adaptation (LoRA) is widely used to reduce resource demands, its low-rank updates may limit its expressiveness for new tasks. HiRA addresses this by using a Hadamard product to retain high-rank update parameters, improving the model capacity. Empirically, HiRA outperforms LoRA and its variants on several tasks, with extensive ablation studies validating its effectiveness.*


## Examples

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model
from peft.tuners.hira import HiraConfig

# Example 1: HiRA on opt-125m for causal language modeling
model_id = "facebook/opt-125m"
base_model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Define HiRA configuration: apply to the MLP dense layers in each transformer block
hira_config = HiraConfig(
    r=32,
    target_modules=["k_proj", "q_proj", "v_proj", "fc1", "fc2"],
    hira_dropout=0.0,
    init_weights=True,
)
peft_model = get_peft_model(base_model, hira_config)

peft_model.print_trainable_parameters()
# trainable params: 4,718,592 || all params: 129,957,888 || trainable%: 3.6309
```

## HiraConfig

[[autodoc]] tuners.hira.config.HiraConfig

## Core Layers

### HiraLayer

[[autodoc]] tuners.hira.layer.HiraLayer

### Linear Adapter

[[autodoc]] tuners.hira.layer.Linear

### Embedding Adapter

[[autodoc]] tuners.hira.layer.Embedding

### Convolutional Adapters

[[autodoc]] tuners.hira.layer.Conv1d [[autodoc]] tuners.hira.layer.Conv2d [[autodoc]] tuners.hira.layer.ConvNd

## BitsAndBytes Integration

* **8-bit Quantized**: [[autodoc]] tuners.hira.bnb.Linear8bitLt
* **4-bit Quantized**: [[autodoc]] tuners.hira.bnb.Linear4bit
* **Dispatch Utilities**:

  * [[autodoc]] tuners.hira.bnb.dispatch_bnb_8bit
  * [[autodoc]] tuners.hira.bnb.dispatch_bnb_4bit

## Dispatch Handler

Default layer replacement for HiRA adapters:

[[autodoc]] tuners.hira.dispatch.dispatch_default


## Citation:
If you found HiRA is useful, please cite HiRA as:
```
@inproceedings{
huang2025hira,
title={Hi{RA}: Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models},
author={Qiushi Huang and Tom Ko and Zhan Zhuang and Lilian Tang and Yu Zhang},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=TwJrTz9cRS}
}
```
