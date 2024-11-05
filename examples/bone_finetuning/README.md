# BONE: BLOCK AFFINE TRANSFORMATION AS PARAMETER EFFICIENT FINE-TUNING METHODS FOR LARGE LANGUAGE MODELS
## Introduction ([Paper](https://arxiv.org/pdf/2409.15371), [code](https://github.com/JL-er/Bone))
Low-Rank Adaptation (LoRA) has achieved remarkable training results by freezing the original weights and training only low-rank matrices, establishing itself as the predominant fine-tuning method for LLMs. In pursuit of performance closer to full-parameter training, a series of LoRA variants have emerged, such as LoRA+, PISSA, Olora, and LoRA-GA. However, these improvements complicate the initial setup of model training and increase initialization time. More importantly, they overlook the internal interactions of the original weight information. To address these issues, we introduce a novel theory, ``Weight Guide'' aimed at continuously guiding trainable matrices through the original weights during training to enhance the utilization of weight information. Based on this theory, we designed a new PEFT technique called Bone Block Affine, which not only enhances the utilization of original weight information but also emphasizes the internal connections between weights, leading to faster convergence and better data fitting. Experimental comparisons across two different LLM architectures (LLaMA2, RWKV6) and various parameter scales demonstrate that the Bone structure can achieve rapid convergence and superior data fitting without the need for complex initialization. For example, when fine-tuning LLaMA2-7B on the MetaMathQA dataset and validating on GSM8k and math benchmarks, Bone achieved fine-tuning scores of 49.36 and 8.8, respectively, outperforming PISSA by 5.84% and 1.96%.

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
python bone_finetuning.py \
    --base_model_name_or_path meta-llama/Llama-2-7b-hf \
    --output_dir output/bone-llama-2-7b-metamath-10k \
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
@misc{kang2024boneblockaffinetransformation,
      title={Bone: Block Affine Transformation as Parameter Efficient Fine-tuning Methods for Large Language Models}, 
      author={Jiale Kang},
      year={2024},
      eprint={2409.15371},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.15371}, 
}