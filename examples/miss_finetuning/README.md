# MiSS: Balancing LoRA Performance and Efficiency with Simple Shard Sharing
## Introduction ([Paper](https://huggingface.co/papers/2409.15371), [code](https://github.com/JL-er/MiSS))
MiSS (Matrix Shard Sharing) is a novel PEFT method that adopts a low-rank structure, requires only a single trainable matrix, and introduces a new update mechanism distinct from LoRA, achieving an excellent balance between performance and efficiency.


## Quick Start
```python
import torch
from peft import MissConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id

miss_config = MissConfig(
    r = 64
)
#bat: In this mode, you can enable nonlinear updates across different shards.
# miss_config = MissConfig(
#     r = 64,
#     init_weights="bat"
# )

# mini: In this mode, you can set a smaller rank to use fewer trainable parameters, but it is recommended to keep `out_features % mini_r == 0`.
# miss_config = MissConfig(
#     r = 64,
#     init_weights="mini",
#     mini_r = 8
# )
peft_model = get_peft_model(model, miss_config)

peft_model.print_trainable_parameters()

dataset = load_dataset("imdb", split="train[:1%]")

training_args = SFTConfig(dataset_text_field="text", max_seq_length=128)
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
peft_model.save_pretrained("miss-llama-2-7b")
```


To utilize the fine-tuned MiSS modules, simply run the following command:
```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16, device_map="auto"
)
peft_model = PeftModel.from_pretrained(model, "miss-llama-2-7b")
```

## Advanced Usage

### Fine-tune 
```shell
#Bat performs better than MiSS, but it uses more memory and is twice as slow. If you want to use the Bat method, you only need to add the parameter init_weights="bat".
python miss_finetuning.py \
    --base_model_name_or_path meta-llama/Llama-2-7b-hf \
    --output_dir output/miss-llama-2-7b-metamath-10k \
    --miss_r 64 \
    --init_weights True \
    --bits bf16 \
    --data_path meta-math/MetaMathQA \
    --dataset_split train[:100000] \
    --dataset_field query response \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --tf32 True \
    --report_to none
```



# Citation
```bib
@misc{kang2025balancingloraperformanceefficiency,
      title={Balancing LoRA Performance and Efficiency with Simple Shard Sharing}, 
      author={Jiale Kang and Qingyu Yin},
      year={2025},
      eprint={2409.15371},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.15371}, 
}
