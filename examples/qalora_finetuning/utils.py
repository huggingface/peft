"""Memory requirement utility

The memory requirement in the tables of this presentation were calculated by
using the included `utils.py` script. To run the script, make sure that
`accelerate` is installed in your Python environment (`python -m pip install
accelerate`). Executing the script does _not_ download the model or load it into
memory. Therefore, you can all this for very large models without the risk to
run out of memory.

```bash
# return memory estimate of Llama3 8B
python utils.py "meta-llama/Meta-Llama-3-8B"
# the same, but using rank 32 for LoRA
python utils.py "meta-llama/Meta-Llama-3-8B" --rank 32
# the same, but loading the model with 4bit quantization
python utils.py "meta-llama/Meta-Llama-3-8B" --dtype int4
```

Note that for gated models, you need to have a Hugging Face account, accept the
terms of the model, and log in to your Hugging Face account:
https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command.

The size of the activations is not included in the estimate.

"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
import os
import torch
from safetensors import safe_open
import transformers
from accelerate.commands.estimate import create_empty_model
from accelerate.utils.other import convert_bytes


# suppress all warnings and logs
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

dtype_to_bytes_linear = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int4": 0.5}
# no quantization if not Linear, assume 16 bit instead
dtype_to_bytes_other = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 2, "int4": 2}
LORA = "lora"
QALORA = "qalora"


def get_num_params(param):  # from PEFT
    """Get the number of parameters from an nn.Parameter"""
    num_params = param.numel()
    # if using DS Zero 3 and the weights are initialized empty
    if num_params == 0 and hasattr(param, "ds_numel"):
        num_params = param.ds_numel

    # Due to the design of 4bit linear layers from bitsandbytes
    # one needs to multiply the number of parameters by 2 to get
    # the correct number of parameters
    if param.__class__.__name__ == "Params4bit":
        if hasattr(param, "element_size"):
            num_bytes = param.element_size()
        elif not hasattr(param, "quant_storage"):
            num_bytes = 1
        else:
            num_bytes = param.quant_storage.itemsize
        num_params = num_params * 2 * num_bytes
    return num_params


def get_param_count(model_id, rank, group_size):
    """Get the number of parameters in a model, including LoRA parameters"""
    # this is only an approximation because we ignore buffers
    model = create_empty_model(model_id, "transformers")

    count_params = defaultdict(int)
    for module in model.modules():
        if len(list(module.children())) > 0:
            continue  # not leaf

        module_name = str(module).split("(", 1)[0]
        for param_name, param in module.named_parameters():
            key = f"{module_name}.{param_name}"
            count_params[key] += get_num_params(param)
            if key == "Linear.weight":
                m, n = param.shape
                count_params[LORA] += m * rank + n * rank
                count_params[QALORA] += m * rank + (n // group_size) * rank
    count_params = dict(count_params)

    # checking against transformers count
    assert 4 * sum(v for k, v in count_params.items() if k != LORA and k != QALORA) == model.get_memory_footprint(
        False
    )
    return count_params


def get_param_bytes(count_params, dtype):
    """Get the number of bytes in a model, including LoRA parameters"""
    num_bytes = defaultdict(int)
    for key, val in count_params.items():
        if key == "Linear.weight":
            num_bytes[key] = int(val * dtype_to_bytes_linear[dtype])
        elif key == LORA:
            # we assume that LoRA is always loaded in float32
            num_bytes[key] = int(val * dtype_to_bytes_other["float32"])
        elif key == QALORA:
            # we assume that QALORA is always loaded in bfloat16
            num_bytes[key] = int(val * dtype_to_bytes_other["bfloat16"])
        else:
            num_bytes[key] = int(val * dtype_to_bytes_other[dtype])
    num_bytes = dict(num_bytes)

    return num_bytes


def get_training_memory_estimate(num_bytes):
    """Get the memory estimate for fine-tuning a model

    Simplified assumptions: don't include activation size, automatic mixed precision, etc.

    We assume that Adam is used, which gives us:

    - size of model itself
    - size of gradients (trainable parameters only)
    - size of 1st and 2nd momentum of Adam (trainable parameters only)

    """
    adapter_keys = [LORA, QALORA]  # Define known adapter keys
    total_size_base = sum(v for k, v in num_bytes.items() if k not in adapter_keys)
    factor = 1 + 1 + 1  # model_params + gradients + optimizer_states
    estimates = {
        "memory full fine-tuning": total_size_base + factor * total_size_base,
    }

    if LORA in num_bytes:
        trainable_bytes_lora = num_bytes[LORA]
        estimates["memory LoRA fine-tuning"] = total_size_base + factor * trainable_bytes_lora

    if QALORA in num_bytes:
        trainable_bytes_qalora = num_bytes[QALORA]
        estimates["memory QALoRA fine-tuning"] = total_size_base + factor * trainable_bytes_qalora

    return estimates


def main(model_id, rank, group_size, dtype, sink=print):
    """Main function to calculate memory requirements of a model.

    Outputs the results in JSON format.

    Args:
        model_id (str): Model name (on Hugging Face)
        rank (int): Rank of LoRA adapter
        dtype (str): Data type, one of float32, float16, bfloat16, int8, int4
        sink (function): Function to print the result with (default: print).
    """
    count_params = get_param_count(model_id, rank=rank, group_size=group_size)
    num_bytes = get_param_bytes(count_params, dtype=dtype)
    num_bytes_readable = {k: convert_bytes(v) for k, v in num_bytes.items()}

    adapter_keys = [LORA, QALORA]  # Define known adapter keys

    # Calculate total parameters/size for the base model (excluding all adapters)
    total_params_base = sum(v for k, v in count_params.items() if k not in adapter_keys)
    total_size_base = sum(v for k, v in num_bytes.items() if k not in adapter_keys)
    total_size_base_readable = convert_bytes(total_size_base)

    # Calculate total parameters/size including all adapters
    total_params_with_adapters = sum(count_params.values())
    total_size_with_adapters = sum(num_bytes.values())
    total_size_with_adapters_readable = convert_bytes(total_size_with_adapters)

    training_bytes = get_training_memory_estimate(num_bytes)
    training_bytes_readable = {k: convert_bytes(v) for k, v in training_bytes.items()}

    result = {
        "number of parameters": count_params,
        "number of bytes": num_bytes,
        "number of bytes (readable)": num_bytes_readable,
        "total number of parameters (base model)": total_params_base,
        "total number of parameters (with adapters)": total_params_with_adapters,
        "total size (base model)": total_size_base,
        "total size (with adapters)": total_size_with_adapters,
        "total size (base model, readable)": total_size_base_readable,
        "total size (with adapters, readable)": total_size_with_adapters_readable,
    }

    # Dynamically add training memory estimates to the result
    if "memory full fine-tuning" in training_bytes_readable:
        result["memory required for full fine-tuning"] = training_bytes_readable["memory full fine-tuning"]
    if "memory LoRA fine-tuning" in training_bytes_readable:
        result["memory required for LoRA fine-tuning"] = training_bytes_readable["memory LoRA fine-tuning"]
    if "memory QALoRA fine-tuning" in training_bytes_readable:
        result["memory required for QALoRA fine-tuning"] = training_bytes_readable["memory QALoRA fine-tuning"]

    if dtype.startswith("int"):
        if "memory required for full fine-tuning" in result:  # Check if key exists before trying to append
            result["memory required for full fine-tuning"] += "*"

    sink(json.dumps(result, indent=2))

    if dtype.startswith("int"):
        print("*Note that quantized models cannot be fine-tuned without PEFT", file=sys.stderr)
    return result



def load_all_lr_layers(adapter_path):
    """
    Lädt einen gespeicherten LoRA-Adapter, extrahiert lora_A und lora_B,
    berechnet den Skalierungsfaktor und rekonstruiert die Low-Rank-Update-Matrix für alle Layer.
    """
    # 1. Pfade zur Konfigurations- und Gewichtsdatei definieren
    config_path = os.path.join(adapter_path, "adapter_config.json")
    weights_path = os.path.join(adapter_path, "adapter_model.safetensors")

    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        print(f"❌ Fehler: Konnte 'adapter_config.json' oder 'adapter_model.safetensors' nicht in {adapter_path} finden.")
        return

    # 2. Konfiguration laden, um 'r' und 'lora_alpha' zu erhalten
    print(f"🔍 Lade Konfiguration von: {config_path}")
    with open(config_path, "r") as f:
        adapter_config = json.load(f)

    r = adapter_config.get("r")
    lora_alpha = adapter_config.get("lora_alpha")
    use_rslora = adapter_config.get("use_rslora", False)

    if r is None or lora_alpha is None:
        print("❌ Fehler: 'r' oder 'lora_alpha' nicht in der Konfiguration gefunden.")
        return

    # 3. Den 'scaling'-Parameter berechnen
    if use_rslora:
        scaling = lora_alpha / torch.sqrt(torch.tensor(r))
        print(f"✅ Konfiguration geladen: r={r}, lora_alpha={lora_alpha} (rsLoRA-Skalierung verwendet)")
    else:
        scaling = lora_alpha / r
        print(f"✅ Konfiguration geladen: r={r}, lora_alpha={lora_alpha}")
    
    print(f"   Berechneter Skalierungsfaktor: {scaling:.4f}\n")

    # 4. Das State Dictionary des Adapters laden
    print(f"🔍 Lade Gewichte von: {weights_path}")
    # Verwende safe_open für safetensors
    adapter_state_dict = {}
    lr_layers = {}
    with safe_open(weights_path, framework="pt") as f:
        for key in f.keys():
            adapter_state_dict[key] = f.get_tensor(key)
    print(f"✅ {len(adapter_state_dict)} Tensoren geladen.\n")

    # 5. Durch alle Layer iterieren und die Low-Rank-Matrix rekonstruieren
    # Finde alle lora_A Gewichte, um die Layer zu identifizieren
    lora_a_keys = [key for key in adapter_state_dict if key.endswith(".lora_A.weight")]

    for lora_a_key in lora_a_keys:
        base_key = lora_a_key.replace(".lora_A.weight", "")
        lora_b_key = base_key + ".lora_B.weight"

        if lora_b_key in adapter_state_dict:
            lora_A = adapter_state_dict[lora_a_key]
            lora_B = adapter_state_dict[lora_b_key]
            
            # Dies ist die Kernlogik Ihrer Anfrage
            # LR = scaling * (B @ A)
            low_rank_update = scaling * (lora_B @ lora_A)
            
            # Speichere im Dict (base_key ist der Layer-Name, z.B. "q_proj")
            cleaned_keys = base_key + ".base_layer.weight"
            lr_layers[cleaned_keys] = low_rank_update
            
            print(f"--- Rekonstruiert für Layer: {base_key} ---")
            print(f"  - lora_A Form: {lora_A.shape}")
            print(f"  - lora_B Form: {lora_B.shape}")
            print(f"  - Rekonstruierte LR-Matrix Form: {low_rank_update.shape}\n")

    print(f"✅ Alle LR-Layer geladen und in lr_layers gespeichert: {list(lr_layers.keys())}")
    
    return lr_layers

def apply_lr_weights(model, lr_layers):
    """
    Wendet die Low-Rank-Gewichte auf das Modell an.
    """
    for name, param in model.named_parameters():
        if name in lr_layers:
            print(f"🔄 Wende LR-Gewichte auf Layer '{name}' an.")
            param.data = lr_layers[name].to(param.device, dtype=param.dtype)

def compare_weights(model, lr_layers):
    """
    Vergleicht die ursprünglichen Gewichte des Modells mit den aktualisierten Gewichten.
    Gibt die maximale, minimale und durchschnittliche Abweichung aus.
    """
    total_error = 0.0
    total_params = 0

    for name, param in model.named_parameters():
        if name in lr_layers:
            original_weights = param.data.clone()
            updated_weights = original_weights + lr_layers[name].to(param.device, dtype=param.dtype)
            error = torch.abs(updated_weights - original_weights).sum().item()
            total_error += error
            total_params += param.numel()

            print(f"🔍 Layer '{name}':")
            print(f"  - Max Error: {torch.abs(updated_weights - original_weights).max().item():.6f}")
            print(f"  - Min Error: {torch.abs(updated_weights - original_weights).min().item():.6f}")
            print(f"  - Mean Error: {error / param.numel():.6f}\n")

    print(f"✅ Gesamtfehler: {total_error:.6f}")
    print(f"✅ Durchschnittlicher Fehler pro Parameter: {total_error / total_params:.6f}")

def show_gradients_of_model(model):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"❌ Gradient disabled for parameter: {name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str, help="Model name (on Hugging Face)")
    parser.add_argument("--rank", type=int, default=8, help="Rank of LoRA adapter")
    parser.add_argument(
        "--dtype", type=str, default="float32", help="Data type, one of float32, float16, bfloat16, int8, int4"
    )
    parser.add_argument("--group_size", type=int, default=8, help="QA_Lora_groupsize")
    args = parser.parse_args()
    main(args.model_id, rank=args.rank, dtype=args.dtype, group_size=args.group_size)

