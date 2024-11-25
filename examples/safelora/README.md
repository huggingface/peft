# Safe LoRA 

The official code of Safe LoRA: The Silver Lining of Reducing Safety Risks when Fine-tuning Large Language Models

## Quick Start

### Get Weights with SafeLoRA
Please import the `SafeLoraConfig` and `apply_safelora` first.
Then, fill in the paths for the base, aligned, and PEFT models according to your needs. There are two types of `select_layers_type`: `threshold` and `number`. The `threshold` type will determine how many layers will be projected based on the value you set. The `number` type directly specifies the number of projected layers. `save_weights=True` will save and replace your original peft model weights.

```python

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

```
### Save SafeLoRA's Weights
If you set `save_weights=False`, but still want to save the weights, you can use the following code.

```python
from safetensors.torch import save_file

path = ...  # your PEFT model path
save_file(final_lora_weight, os.path.join(path, "adapter_model.safetensors"))
```

### Use SafeLoRA Model
Next, you can load the base model of the Peft Model along with the Peft model itself to use a model that has both downstream task utility and alignment.

```python
from transformers import AutoModelForCausalLM
from peft import PeftConfig, PeftModel

model = AutoModelForCausalLM.from_pretrained(<base-model-id>)
model = PeftModel.from_pretrained(model, <SafeLoRA-path>)
```
