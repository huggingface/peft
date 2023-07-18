# Fine-tuning a multilayer perceptron using LoRA and 🤗 PEFT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/peft/blob/main/examples/multilayer_perceptron/multilayer_perceptron_lora.ipynb)

PEFT supports fine-tuning any type of model as long as the layers being used are supported. The model does not have to be a transformers model, for instance. To demonstrate this, the accompanying notebook `multilayer_perceptron_lora.ipynb` shows how to apply LoRA to a simple multilayer perceptron and use it to train a model to perform a classification task.
