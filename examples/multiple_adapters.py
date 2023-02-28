import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training


from transformers import AutoModelForCausalLM

model_name = "edbeeching/gpt-neo-125M-imdb"
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")


"""### Apply LoRA
Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.
"""
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


lora_config = LoraConfig(
    r=(16,16),
    lora_alpha=(32,32),
    target_modules=None,  #handled automatically by peft
    lora_dropout=(0.05,0.01),
    bias="none",
    task_type="CAUSAL_LM",
    n_adapters=2
)

model = prepare_model_for_int8_training(model)
model = get_peft_model(model, lora_config)

model.enable_adapter_index(1)
model.train()
out = model(torch.LongTensor([1,2,3,4]).to("cuda"))
loss = out[0].mean()
loss.backward()


for n, p in model.named_parameters():
    print(n, p.grad)

# model.enable_adaptor(0)
# model.disable_adaptor(0)