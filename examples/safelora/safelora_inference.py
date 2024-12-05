import os

from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from peft import PeftModel
from peft.utils.safelora import SafeLoraConfig, apply_safelora


peft_path = "../finetuneLLM/finetuned_models/samsumBad-7b-fp16-peft-seed-42"
config = SafeLoraConfig(
    base_model_path="meta-llama/Llama-2-7b-hf",
    aligned_model_path="TheBloke/Llama-2-7B-Chat-fp16",
    peft_model_path=peft_path,
    device="cuda",
    select_layers_type="threshold",
    save_weights=True,
)

final_lora_weight = apply_safelora(config)

save_file(
    final_lora_weight,
    f"{os.path.join('../finetuneLLM/finetuned_models/samsumBad-7b-fp16-peft-seed-42', 'adapter_model.safetensors')}",
)

model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-fp16")
model = PeftModel.from_pretrained(model, peft_path)
