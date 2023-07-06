1. Setup env - train all models (the technique is specialize and this task is run with example model name user can explore more, see more in code).
```
bash semantic_segmentation_peft_lora.sh
```
* To track top GPU vRAM usage by device, open a new terminal and run this script, pass device ID, for ex device 0:
```
bash ../memory_record_moreh.sh 0
```

# Model
## Fine-tuning for semantic segmentation using LoRA and 🤗 PEFT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/peft/blob/main/examples/semantic_segmentation/semantic_segmentation_peft_lora.ipynb) 

We provide a notebook (`semantic_segmentation_peft_lora.ipynb`) where we learn how to use [LoRA](https://arxiv.org/abs/2106.09685) from 🤗 PEFT to fine-tune an semantic segmentation by ONLY using **14%%** of the original trainable parameters of the model. 

LoRA adds low-rank "update matrices" to certain blocks in the underlying model (in this case the attention blocks) and ONLY trains those matrices during fine-tuning. During inference, these update matrices are _merged_ with the original model parameters. For more details, check out the [original LoRA paper](https://arxiv.org/abs/2106.09685). 
