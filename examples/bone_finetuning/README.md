# DiSHA: Dimension-Sharding Adaptation with Fast Convergence and Fast Computation
## Introduction ([Paper](https://arxiv.org/pdf/2409.15371), [code](https://github.com/JL-er/DiSHA))
Low-Rank Adaptation (LoRA) leverages the low intrinsic rank of weight updates in Large Language Models (LLMs), establishing a Parameter-Efficient Fine-Tuning (PEFT) paradigm. However, LoRA suffers from slow convergence. We introduce Dimension-Sharding Adaptation (DiSHA), which expands the PEFT design space to unlock lower intrinsic ranks and faster convergence by default. Within DiSHA's design space, we propose Block Affine Adaptation (Bone), a computationally efficient structure that delivers both high performance and efficiency. While certain DiSHA configurations may result in colinear updates to weight shards, we address this with Block Affine Transformation Adaptation (BAT), a nonlinear variant of DiSHA. BAT introduces nonlinearity by combining trainable matrices with original weight shards in a nonlinear manner, inducing nonlinearity in matrix updates without introducing additional parameters. Empirical results show that Bone, under the DiSHA framework, consistently outperforms LoRA variants in both NLG and NLU tasks, with significantly improved computational efficiency. Further analysis demonstrates that BAT enhances model capabilities by leveraging its nonlinear design.


## Quick Start
```python
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id
bone_config = BoneConfig(
    r = 64
)
#Bat performs better than Bone, but it uses more memory and is twice as slow. If you want to use the Bat method, you only need to add the parameter init_weights="bat".
# bone_config = BoneConfig(
#     r = 64,
#     init_weights="bat"
# )
peft_model = get_peft_model(model, bone_config)

peft_model.print_trainable_parameters()

dataset = load_dataset("imdb", split="train[:1%]")

training_args = SFTConfig(dataset_text_field="text", max_seq_length=128)
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
peft_model.save_pretrained("bone-llama-2-7b")
```


To utilize the fine-tuned Bone modules, simply run the following command:
```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16, device_map="auto"
)
peft_model = PeftModel.from_pretrained(model, "bone-llama-2-7b")
```

## Advanced Usage

### Fine-tune 
```shell
#Bat performs better than Bone, but it uses more memory and is twice as slow. If you want to use the Bat method, you only need to add the parameter init_weights="bat".
python bone_finetuning.py \
    --base_model_name_or_path meta-llama/Llama-2-7b-hf \
    --output_dir output/bone-llama-2-7b-metamath-10k \
    --bone_r 64 \
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
@misc{kang2025dishadimensionshardingadaptationlarge,
      title={DiSHA: Dimension-Sharding Adaptation of Large Language Models with Fast Convergence and Fast Computation}, 
      author={Jiale Kang},
      year={2025},
      eprint={2409.15371},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.15371}, 
}