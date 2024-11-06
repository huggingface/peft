from peft import PeftConfig
from peft.utils.safelora import SafeLoraConfig, apply_safelora

config = SafeLoraConfig(base_model_path='../LLM_Models/llama-2-7b-hf/',\
                            aligned_model_path='../LLM_Models/llama-2-7b-chat-fp16/',
                            peft_model_path = '../finetuneLLM/finetuned_models/samsumBad-7b-fp16-peft-seed-42',
                            devices='cuda',
                            select_layers_type='threshold',
                            save_weights=True)

final_lora_weight = apply_safelora(config)