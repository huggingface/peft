#Safe LoRA 

The official code of Safe LoRA:the Silver Lining of Reducing Safety Risks when Fine-tuning Large Language Models


## Quick Start

Please import the SafeLoraConfig and apply_safelora first.
Then, fill in the paths for the base, aligned, and PEFT models according to your needs. There are two types of `select_layers_type`: `threshold` and `number`. The `threshold` type will determine how many layers will be projected based on the value you set. The `number` type directly specifies the number of projected layers.

```python

from peft.utils.safelora import SafeLoraConfig, apply_safelora

config = SafeLoraConfig(base_model_path='../LLM_Models/llama-2-7b-hf/',\
                            aligned_model_path='../LLM_Models/llama-2-7b-chat-fp16/',
                            peft_model_path = '../finetuneLLM/finetuned_models/samsumBad-7b-fp16-peft-seed-42',
                            devices='cuda',
                            select_layers_type='threshold',
                            save_weights=True)

final_lora_weight = apply_safelora(config)

```
