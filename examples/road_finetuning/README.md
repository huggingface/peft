# RoAd: 3-in-1: 2D Rotary Adaptation for Efficient Finetuning, Efficient Batching and Composability


## Introduction

[RoAd](https://huggingface.co/papers/2409.00119) is a novel method that adapts LLMs using simple 2D rotations. It is highly parameter-efficient,
achieving strong performance with less than 0.1% trainable parameters.
RoAd also supports efficient serving of mixed-adapter requests within a batch, incurring only element-wise computation overhead rather than costly batch matrix multiplications.
Additionally, it improves model interpretability through structured and composable transformations.

## Quick start
```python
import torch
from peft import RoadConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b", device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
road_config = RoadConfig(
    variant="1",
)
peft_model = get_peft_model(model, road_config)
trainer = transformers.Trainer(
    model=peft_model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_length=2048,
    tokenizer=tokenizer,
)
trainer.train()
peft_model.save_pretrained("road-llama-3-8b")
```

RoAd requires a higher learning rate compared to LoRa and similar approaches, set it to around 1e-3.

Run the finetuning script simply by running:

```bash
python examples/road_finetuning/road_finetuning.py --base_model meta-llama/Meta-Llama-3-8B --data_path timdettmers/openassistant-guanaco
```

RoAd also supports quantization. To use 4-bit quantization try:

```bash
python examples/road_finetuning/road_finetuning.py --base_model meta-llama/Meta-Llama-3-8B --quantize
```

### Full example of the script 
```bash
python road_finetuning.py \
    --base_model "PATH_TO_MODEL" \
    --data_path "PATH_TO_DATASET" \
    --output_dir "PATH_TO_OUTPUT_DIR" \
    --batch_size 1 \
    --num_epochs 3 \
    --learning_rate 1e-3 \
    --cutoff_len 512 \
    --val_set_size 500 \
    --quantize \
    --eval_step 10 \
    --save_step 100 \
    --device "cuda:0" \
    --variant 1 \
    --road_target_modules "q_proj,k_proj,v_proj,o_proj" \
    --hub_model_id "YOUR_HF_REPO" \
    --push_to_hub
```
## Use the model on 🤗
You can load and use the model as any other 🤗 models.
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("ppetrushkov/llama-2-7b-sql-road-test")
```


## Citation
```
@inproceedings{
    liao2024in,
    title={3-in-1: 2D Rotary Adaptation for Efficient Finetuning, Efficient Batching and Composability},
    author={Baohao Liao and Christof Monz},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=rYjYwuM6yH}
}
```
