# RandLora: Full-rank parameter-efficient fine-tuning of large models 

## Introduction
[RandLora](https://huggingface.co/papers/2502.00987) is a parameter-efficient fine-tuning technique that is similar to LoRA and VeRA but performs full rank updates to improve performance. RandLora can be particulary usefull when adapting large model to hard tasks that require complex updates while preserving the parameter efficiency of LoRA. The full rank update of RandLora is acheived by linearly scaling random bases. The random bases are a collection of multiple low rank matrices such that the summation of their ranks if greater or equal to the full rank of the parameter matrices. The trainable parameters of RandLora are two diagonal matrices (vectors) that get multiplied with the right hand low rank random bases, in a similar way to VeRA's update. To maintain low memory usage, RandLora uses a custom function that prevents storing unnecessary bases in memory for backpropagation.

## Quick start
```python
import torch
from peft import RandLoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b", device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
randlora_config = RandLoraConfig()

peft_model = get_peft_model(model, lora_config)
trainer = transformers.Trainer(
    model=peft_model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    processing_class=tokenizer,
)
trainer.train()
peft_model.save_pretrained("randlora-llama-7b")
```

There is no additional change needed to your standard PEFT training procedure, simply swap your `LoraConfig` for a `RandLoraConfig`. Note however that RandLora's trainable parameter count is **inversely proportional** to the rank parameter `r`. Lower `r` to increase and increase it to reduce trainable parameters of RandLora.

Run the finetuning script simply by running:
```bash
python examples/randlora_finetuning/randlora_finetuning.py --base_model meta-llama/Meta-Llama-3-8B --data_path timdettmers/openassistant-guanaco
```
This 👆🏻 by default will load the model in peft set up with RandLora config. Now if you wanna quickly compare it with Lora, all you need to do is to input ` --use_lora` in the command line and reduce `--randlora_alpha` to 2x the rank. So same above example would be 👇🏻;

```bash
python examples/randlora_finetuning/randlora_finetuning.py --base_model meta-llama/Meta-Llama-3-8B --data_path timdettmers/openassistant-guanaco --use_lora --rank 32 --randlora_alpha 64
```

RandLora can be made to use sparse or very sparse random bases. These sparse matrices can help reduce overfitting. Add `--very_sparse` to run with very sparse matrices or `--sparse` for sparse matrices:

```bash
python examples/randlora_finetuning/randlora_finetuning.py --base_model meta-llama/Meta-Llama-3-8B --sparse
```

RandLora also supports quantization. To use 4-bit quantization try:

```bash
python examples/randlora_finetuning/randlora_finetuning.py --base_model meta-llama/Meta-Llama-3-8B --quantize
```

By default the RandLora layers are the key and value layers of LLama model. Adding adapters on more layers will increase memory usage. If you wish to choose a different set of layers for RandLora to be applied on, you can simply define it using:
```bash
python examples/randlora_finetuning/randlora_finetuning.py --randlora_target_modules "q_proj,k_proj,v_proj" 
```

### Full example of the script 
```bash
python randlora_finetuning.py \
    --base_model "PATH_TO_MODEL" \
    --data_path "PATH_TO_DATASET" \
    --output_dir "PATH_TO_OUTPUT_DIR" \
    --batch_size 1 \
    --num_epochs 3 \
    --learning_rate 3e-4 \
    --cutoff_len 512 \
    --val_set_size 500 \
    --quantize \
    --eval_step 10 \
    --save_step 100 \
    --device "cuda:0" \
    --rank 32 \
    --randlora_alpha 640 \
    --randlora_dropout 0.05 \
    --randlora_target_modules "k_proj,v_proj" \
    --hub_model_id "YOUR_HF_REPO" \
    --push_to_hub
```

## RandLora vs. LoRA
RandLora differs from LoRA and other related low rank approximation algorithms by chanllenging the low rank paradigm. RandLora adapters learn **full-rank** updates as the [paper](https://huggingface.co/papers/2502.00987) shows that the low rank constraint of LoRA can constrain performance gains as trainable parameters increase (with higher ranks). As a result, using RandLora is specifically recommended for difficult tasks that are underfit by LoRA. RandLoRA however also often improves performance for common tasks. If increasing LoRA's rank improves performance for your task, RandLora will most likely outperform.

RandLora is expected to increase performance over LoRA for equivalent amounts of trainable parameters, mostly for larger equivalent amounts (> LoRA rank 4).

RandLora's performance increase comes with two limitations:

1. Performance is dependent on using a large `randlora_alpha` scaling parameter (usually 20x the basis rank). This large parameter can sometimes make training the update unstable, reduce the learning rate or the scaling parameter if this is the case.

2. Increase training time over LoRA when using very low RandLora basis ranks.

## RandLora vs. VeRA
RandLora shares similarities with VeRA in that both algorithms use random basis combinations to address some of LoRA's limitations. The limitations addressed by each algorithm is however different.
VeRA aims to reduce trainable parameters beyond rank 1 LoRAs while RandLoRA reduces the performance limitation due to the low rank of the update as the trainable parameter count increases.

RandLora is expected to:

1. Improve performance over VeRA when more trainable parameters are required (hard tasks)

2. Reduce memory usage over VeRA thanks to RandLora's random base sharing strategy


## Citation
```
@inproceedings{2025_ICLR_RandLoRA,
  title="{RandLoRA: Full rank parameter-efficient fine-tuning of large models}",
  author="Albert, Paul and Zhang, Frederic Z. and Saratchandran, Hemanth and Rodriguez-Opazo, Cristian and van den Hengel, Anton and Abbasnejad, Ehsan",
  booktitle="{International Conference on Learning Representations (ICLR)}",
  year="2025"
}
```