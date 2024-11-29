# BONE: BLOCK AFFINE TRANSFORMATION AS PARAMETER EFFICIENT FINE-TUNING METHODS FOR LARGE LANGUAGE MODELS
## Introduction ([Paper](https://arxiv.org/pdf/2409.15371), [code](https://github.com/JL-er/Bone))
Low-Rank Adaptation (LoRA) has achieved remarkable training results by freezing the original weights and training only low-rank matrices, establishing itself as the predominant fine-tuning method for LLMs. In pursuit of performance closer to full-parameter training, a series of LoRA variants have emerged, such as LoRA+, PISSA, Olora, and LoRA-GA. This paper introduces a novel PEFT technique distinct from LoRA, called Block-Affine Adaptation (Bone). By dividing the original weights into multiple subspaces that share a single matrix for weight updates, Bone simplifies the process by requiring the trainable matrix to be initialized to zero, eliminating the need for complex initialization as in some LoRA variants. Compared to LoRA, Bone significantly reduces memory usage and achieves faster computation. Evaluation of both NLU and NLG tasks demonstrates that Bone substantially outperforms LoRA and its variants. Inspired by Pissa, we further proposed the 'Weight Guide' theory to better utilize the information from the original weights. By integrating 'Weight Guide' with Bone, we developed a new structure called Block-Affine Transformation (Bat), and ablation experiments confirmed the effectiveness of 'Weight Guide'.

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
@misc{kang2024boneblockaffineadaptationlarge,
      title={Bone: Block-Affine Adaptation of Large Language Models}, 
      author={Jiale Kang},
      year={2024},
      eprint={2409.15371},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.15371}, 
}