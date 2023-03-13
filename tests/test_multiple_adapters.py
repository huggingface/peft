import unittest
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM

class PeftAdapterTester(unittest.TestCase):
    
    
    def test_single_adapter_8bit_model():
        lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=None,  #handled automatically by peft
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                n_adapters=2
        )
        model_name = "EleutherAI/gpt-neo-125M"
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
        
        model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, lora_config)
        out = model(torch.LongTensor([1,2,3,4]).to("cuda"))
        loss = out[0].mean()
        loss.backward()

        for n, p in model.named_parameters():
            print(n, p.grad)


    def test_multi_adapters_8bit_model():
        lora_config = LoraConfig(
        r=[16,8],
        lora_alpha=32,
        target_modules=None,  #handled automatically by peft
        lora_dropout=[0.05, 0.1],
        bias="none",
        task_type="CAUSAL_LM",
        n_adapters=2
        )        
        
        model_name = "EleutherAI/gpt-neo-125M"
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
        
        model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, lora_config)

        for i in range(2):
            model.enable_adapter_index(i)
            model.train()
            out = model(torch.LongTensor([1,2,3,4]).to("cuda"))
            loss = out[0].mean()
            loss.backward()

            for n, p in model.named_parameters():
                print(n, p.grad)


    def test_single_adapter_model():
        lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=None,  #handled automatically by peft
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                n_adapters=2
        )
        model_name = "EleutherAI/gpt-neo-125M"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = get_peft_model(model, lora_config)
        out = model(torch.LongTensor([1,2,3,4]).to("cuda"))
        loss = out[0].mean()
        loss.backward()

        for n, p in model.named_parameters():
            print(n, p.grad)


    def test_multi_adapters_model():
        lora_config = LoraConfig(
        r=[16,8],
        lora_alpha=32,
        target_modules=None,  #handled automatically by peft
        lora_dropout=[0.05, 0.1],
        bias="none",
        task_type="CAUSAL_LM",
        n_adapters=2
        )        
        
        model_name = "EleutherAI/gpt-neo-125M"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = get_peft_model(model, lora_config)

        for i in range(2):
            model.enable_adapter_index(i)
            model.train()
            out = model(torch.LongTensor([1,2,3,4]).to("cuda"))
            loss = out[0].mean()
            loss.backward()

            for n, p in model.named_parameters():
                print(n, p.grad)


    def test_single_adapter_8bit_model_merged_linear():
        lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=None,  #handled automatically by peft
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                n_adapters=2
        )
        model_name = "bigscience/bloom-560m"
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
        
        model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, lora_config)
        out = model(torch.LongTensor([1,2,3,4]).to("cuda"))
        loss = out[0].mean()
        loss.backward()

        for n, p in model.named_parameters():
            print(n, p.grad)


    def test_multi_adapters_8bit_model_merged_linear():
        lora_config = LoraConfig(
        r=[16,8],
        lora_alpha=32,
        target_modules=None,  #handled automatically by peft
        lora_dropout=[0.05, 0.1],
        bias="none",
        task_type="CAUSAL_LM",
        n_adapters=2
        )        
        
        model_name = "bigscience/bloom-560m"
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
        
        model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, lora_config)

        for i in range(2):
            model.enable_adapter_index(i)
            model.train()
            out = model(torch.LongTensor([1,2,3,4]).to("cuda"))
            loss = out[0].mean()
            loss.backward()

            for n, p in model.named_parameters():
                print(n, p.grad)


    def test_single_adapter_model_merged_linear():
        lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=None,  #handled automatically by peft
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                n_adapters=2
        )
        model_name = "bigscience/bloom-560m"
        model = AutoModelForCausalLM.from_pretrained(model_name)

        model = get_peft_model(model, lora_config)
        out = model(torch.LongTensor([1,2,3,4]).to("cuda"))
        loss = out[0].mean()
        loss.backward()

        for n, p in model.named_parameters():
            print(n, p.grad)


    def test_multi_adapters_model_merged_linear():
        lora_config = LoraConfig(
        r=[16,8],
        lora_alpha=32,
        target_modules=None,  #handled automatically by peft
        lora_dropout=[0.05, 0.1],
        bias="none",
        task_type="CAUSAL_LM",
        n_adapters=2
        )        
        
        model_name = "bigscience/bloom-560m"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        model = get_peft_model(model, lora_config)

        for i in range(2):
            model.enable_adapter_index(i)
            model.train()
            out = model(torch.LongTensor([1,2,3,4]).to("cuda"))
            loss = out[0].mean()
            loss.backward()

            for n, p in model.named_parameters():
                print(n, p.grad)