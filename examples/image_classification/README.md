# Fine-tuning for image classification using LoRA and 🤗 PEFT

## Vision Transformer model from transformers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/peft/blob/main/examples/image_classification/image_classification_peft_lora.ipynb) 

We provide a notebook (`image_classification_peft_lora.ipynb`) where we learn how to use [LoRA](https://arxiv.org/abs/2106.09685) from 🤗 PEFT to fine-tune an image classification model by ONLY using **0.7%** of the original trainable parameters of the model. 

LoRA adds low-rank "update matrices" to certain blocks in the underlying model (in this case the attention blocks) and ONLY trains those matrices during fine-tuning. During inference, these update matrices are _merged_ with the original model parameters. For more details, check out the [original LoRA paper](https://arxiv.org/abs/2106.09685). 

## PoolFormer model from timm

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/peft/blob/main/examples/image_classification/image_classification_timm_peft_lora.ipynb) 

The notebook `image_classification_timm_peft_lora.ipynb` showcases fine-tuning an image classification model using from the [timm](https://huggingface.co/docs/timm/index) library. Again, LoRA is used to reduce the numberof trainable parameters to a fraction of the total.
