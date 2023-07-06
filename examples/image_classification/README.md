1. Setup env - train all models
```
bash run_all_models.sh model_batchsize.txt
```
* To track top GPU vRAM usage by device, open a new terminal and run this script, pass device ID, for ex device 2:
```
bash ../memory_record_moreh.sh 2
```

* You can also pick one and train as you want
```
#setup (skip if you did)
bash setup.sh image_classification
image_classification_peft_lora.sh
```


# Task infos:
## Fine-tuning for image classification using LoRA and 🤗 PEFT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/peft/blob/main/examples/image_classification/image_classification_peft_lora.ipynb) 

We provide a notebook (`image_classification_peft_lora.ipynb`) where we learn how to use [LoRA](https://arxiv.org/abs/2106.09685) from 🤗 PEFT to fine-tune an image classification model by ONLY using **0.7%** of the original trainable parameters of the model. 

LoRA adds low-rank "update matrices" to certain blocks in the underlying model (in this case the attention blocks) and ONLY trains those matrices during fine-tuning. During inference, these update matrices are _merged_ with the original model parameters. For more details, check out the [original LoRA paper](https://arxiv.org/abs/2106.09685). 
