import os
from peft import PeftModel

def save_and_share(peft_model: PeftModel, save_path: str, push_to_hub_user: str = None):
    # No need for SVD when we load the PiSSA and residual model saved locally.
    peft_model.peft_config['default'].init_lora_weights = True
    # Save PiSSA adapter.
    peft_model.save_pretrained(os.path.join(save_path,"pissa_init"))
    
    # Directly delete default adapter with `peft_model.delete_adapter("default")` and save as base model with `peft_model.get_base_model()` will raise an error (At least one adapter is needed):
    # File ~/peft/src/peft/peft_model.py:926, in PeftModel.active_peft_config(self)
    #     925 def active_peft_config(self):
    # --> 926     return self.peft_config[self.active_adapter]
    # KeyError: 'default'
    
    # Therefore, we merging the residual with a zero initialized lora, which will not influence its value.
    peft_model.add_adapter(adapter_name='zero_initialized_lora', peft_config=peft_model.peft_config['default'])
    peft_model.merge_and_unload(adapter_names=['zero_initialized_lora'])
    model = peft_model.get_base_model()
    
    # Save the residual model and the tokenizer.
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # (Optional) shareing the residual model to huggingface hub.
    if push_to_hub_user is not None:
        model.push_to_hub(os.path.join(push_to_hub_user, save_path))
        tokenizer.push_to_hub(os.path.join(push_to_hub_user, save_path))
    
    
if __name__ == "__main__":
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    lora_config = LoraConfig(r=16,lora_alpha=16,init_lora_weights="pissa", lora_dropout=0, target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    save_and_share(model, save_path="pissa-llama-2-7b-r16-alpha-16")
    
