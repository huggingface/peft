# QALoRA: Quantization-Aware Low-Rank Adaptation

## Introduction
[QALoRA](https://huggingface.co/papers/2309.14717) is a quantization-aware version of Low-Rank Adaptation that enables efficient fine-tuning of quantized large language models. 
QALoRA uses input feature pooling and a specialized grouping technique to work with quantized weights, significantly reducing memory requirements while preserving performance. 
QALoRA enables fine-tuning of models that would otherwise be too large for consumer GPUs. In PEFT it only works for GPTQ.

## Quick start
```python
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
from datasets import load_dataset

# Load a quantized model (example with GPTQ quantization)
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7b-GPTQ", 
    revision="gptq-4bit-32g-actorder_True", 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7b-GPTQ")
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

# Configure QALoRA parameters
lora_config = LoraConfig(
    use_qalora=True,
    qalora_group_size=8,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
)

# Create the PEFT model
peft_model = get_peft_model(model, lora_config)

# Set up trainer and train
trainer = Trainer(
    model=peft_model,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=3e-4,
        output_dir="qalora-llama-2-7b"
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()
peft_model.save_pretrained("qalora-llama-2-7b")
```

To use QALoRA, simply set `use_qalora = True` and specify a `qalora_group_size` in your LoRA configuration. The group size controls the memory/performance tradeoff - smaller values use less memory but may affect performance.

## Command Line Examples

Run the finetuning script with a GPTQ quantized model:

You can customize the pooling group size (default is 16):
```bash
python examples/qalora_finetuning/qalora_gptq_finetuning.py \
    --base_model TheBloke/Llama-2-7b-GPTQ \
    --use_qalora \
    --qalora_group_size 32
```

### Full example of the script 
```bash
python qalora_gptq_finetuning.py \
    --base_model "TheBloke/Llama-2-13b-GPTQ" \
    --output_dir "PATH_TO_OUTPUT_DIR" \
    --batch_size 1 \
    --num_epochs 3 \
    --learning_rate 3e-4 \
    --cutoff_len 512 \
    --use_qalora \
    --qalora_group_size 32 \
    --eval_step 10 \
    --save_step 100 \
    --device "cuda:0" \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --push_to_hub
```

## Use the model on 🤗
You can load and use the finetuned model like any other PEFT model:
```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the base quantized model
base_model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7b-GPTQ",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7b-GPTQ")

# Load the PEFT adapter
peft_model_id = "YOUR_HF_REPO"
model = PeftModel.from_pretrained(base_model, peft_model_id)

# Generate text
input_text = "Hello, I'm a language model"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## QALoRA vs. LoRA

QALoRA offers several advantages over standard LoRA:

1. **Memory efficiency**: QALoRA works directly with quantized models, reducing memory usage by up to 60-70% compared to standard LoRA.

2. **Hardware accessibility**: Enables fine-tuning of larger models (13B, 70B) on consumer GPUs that would be impossible with standard LoRA.

3. **Performance preservation**: Despite quantization, QALoRA can achieve comparable performance to full-precision LoRA in many tasks.


## Implementation Details: Merging with Quantized Models

> **Note:** The current implementation differs from the original QA-LoRA paper's approach.

While the QA-LoRA paper describes a direct weight modification technique using "beta shift" to modify quantized weights without full dequantization, this implementation uses a different approach:

1. The quantized model is first dequantized to full precision
2. The QALoRA adapter weights are then merged with the dequantized model
3. The merged model must be re-quantized if quantization is still desired


### Memory Considerations

This process requires significant memory (enough to hold the full dequantized model) and additional computation for the re-quantization step. For large models, this may not be possible on consumer hardware.

For most use cases, we recommend keeping the base quantized model and the QALoRA adapter separate, loading them with `PeftModel.from_pretrained()` as shown in the usage example above. This approach maintains the memory efficiency benefits of quantization throughout the deployment pipeline.


## Citation
```
@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}

@article{xu2023qalora,
  title={QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models},
  author={Xu, Yuhui and Liu, Lingxi and Rao, Longhui and Zhao, Teng and Xiong, Zhiwei and Gao, Mingkui},
  journal={arXiv preprint arXiv:2309.14717},
  year={2023}
}
```