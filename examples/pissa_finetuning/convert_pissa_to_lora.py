import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import json


# A PiSSA of rank r can be equivalently represented by a LoRA of rank 2r.
# The advantage of PiSSA lies in the training phase. Upon completion of training, when sharing with others, it is recommended to convert PiSSA into LoRA. 
# LoRA does not modify the parameters of the base model during use. 
# When multiple PiSSAs/LoRAs are needed simultaneously, each adapter works independently without interference, allowing for the adapters to be freely deleted or added.

def pissa_to_lora(init_path, finetuned_path, output_path, device='cpu', tensors_name="adapter_model.safetensors", config_name="adapter_config.json"):
    tensors_init = {}
    with safe_open(os.path.join(init_path, tensors_name), framework="pt", device=device) as f:
        for k in f.keys():
            tensors_init[k] = f.get_tensor(k)
            
    tensors_finetune = {}
    with safe_open(os.path.join(finetuned_path, tensors_name), framework="pt", device=device) as f:
        for k in f.keys():
            tensors_finetune[k] = f.get_tensor(k)
            
    tensors_delta_w = {}
    for name in tensors_init.keys():
        ## W = W^res + A_0 \times B_0,
        ## W + \Delta W = W^res + A \times B,
        ## \Delta W = A \times B - A_0 \times B_0 = [A + A_0] \times [B - B_0]^T.
        tensors_delta_w[name] = torch.cat([tensors_finetune[name], tensors_init[name]], dim=0) if 'lora_A' in name else torch.cat([tensors_finetune[name], -tensors_init[name]], dim=1)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    save_file(tensors_delta_w, os.path.join(output_path, tensors_name))
    
    with open(os.path.join(init_path, config_name))as f:
        adapter_config = json.load(f)
    adapter_config['init_lora_weights']=True
    adapter_config['r']*=2
    adapter_config['lora_alpha']*=2
    with open(os.path.join(output_path, config_name),'w')as f:
        json.dump(adapter_config, f)

if __name__ == "__main__":
    
    # Download the llama-2-7b model from huggingface
    
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b', device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b')
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Configure PiSSA with Fast SVD:

    from peft import LoraConfig, get_peft_model
    peft_config = LoraConfig(
        r = 16,
        lora_alpha = 16,
        lora_dropout = 0,
        init_lora_weights='pissa_niter_4', # Fast initialization with "_niter_xx"
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()


    ############################## It's essential to save initial PiSSA parameters for conversion to LoRA. ##############################

    model.save_pretrained('pissa-r16-llama-2-7b-alpaca-init')

    # Finetuning with PiSSA

    from trl import SFTTrainer
    from datasets import load_dataset
    dataset = load_dataset("fxmeng/alpaca_in_mixtral_format", split="train")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer
    )
    trainer.train()

    ############################## Upon completion, save final PiSSA parameters ##############################
    model.save_pretrained('pissa-r16-llama-2-7b-alpaca-finetuned')

    ############################## The different of the PiSSA parameters before and after the training corresponding to delta W in LoRA. ##############################
    pissa_to_lora('pissa-r16-llama-2-7b-alpaca-init', 'pissa-r16-llama-2-7b-alpaca-finetuned', "lora-r32-llama-2-7b-alpaca", device='cpu')