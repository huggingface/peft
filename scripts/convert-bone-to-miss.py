import json
import os

input_dir = "xxx/bone"
output_dir = "xxx/miss"


output_path = os.path.join(output_dir, "adapter_config.json")

os.makedirs(output_dir, exist_ok=True)
input_path = os.path.join(input_dir, "adapter_config.json")
with open(input_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

config["peft_type"] = "MISS"

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)


import os
from safetensors import safe_open
from safetensors.torch import save_file

input_path = os.path.join(input_dir, 'adapter_model.safetensors')
output_path = os.path.join(output_dir, 'adapter_model.safetensors')

os.makedirs(output_dir, exist_ok=True)
new_data = {}
with safe_open(input_path, framework='pt') as f:
    for old_key in f.keys():
        tensor = f.get_tensor(old_key)
        print(f"{old_key}: {tensor.shape}")
        
        new_key = old_key.replace("bone", "miss")
        new_data[new_key] = tensor

save_file(new_data, output_path)


print(f"\nfile save: {output_path}")
