import json
import os
from datetime import datetime
import random

def generate_peft_config(peft_type):
    if peft_type == "LORA":
        return {
            "peft_type": "LORA",
            "r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj"]
        }
    elif peft_type == "ADAPTION_PROMPT":
        return {
            "peft_type": "ADAPTION_PROMPT",
            "adapter_len": 10,
            "adapter_layers": 2,
            "task_type": "CAUSAL_LM"
        }
    elif peft_type == "PREFIX_TUNING":
        return {
            "peft_type": "PREFIX_TUNING",
            "num_virtual_tokens": 20,
            "encoder_hidden_size": 512
        }
    else:
        return {
            "peft_type": peft_type,
            "task_type": "CAUSAL_LM"
        }

def generate_sample_json(peft_type, experiment_num):
    # Create sample data with some randomization for realistic variation
    sample_data = {
        "run_info": {
            "experiment_name": f"{peft_type.lower()}_experiment_{experiment_num}",
            "train_config": {
                "model_id": "meta-llama/Llama-2-7b-hf"
            },
            "peft_config": generate_peft_config(peft_type),
            "total_time": random.uniform(3000, 4000),  # 50-66 minutes
            "created_at": datetime.now().isoformat(),
            "peft_branch": "main"
        },
        "train_info": {
            "status": "success",
            "cuda_memory_reserved_avg": random.randint(10000, 15000),  # MB
            "cuda_memory_max": random.randint(12000, 18000),  # MB
            "cuda_memory_reserved_99th": random.randint(11000, 16000),  # MB
            "train_time": random.uniform(2800, 3800),  # seconds
            "file_size": random.randint(4000000, 6000000),  # bytes
            "metrics": [
                {
                    "test accuracy": random.uniform(0.7, 0.9),
                    "train loss": random.uniform(0.1, 0.3),
                    "train samples": random.randint(800, 1200),
                    "train total tokens": random.randint(40000, 60000)
                }
            ]
        },
        "meta_info": {
            "package_info": {
                "peft-version": "0.7.0",
                "transformers-version": "4.36.0",
                "datasets-version": "2.14.0",
                "torch-version": "2.1.0",
                "bitsandbytes-version": "0.41.0"
            },
            "system_info": {
                "gpu": "NVIDIA A100",
                "cuda_version": "12.1",
                "python_version": "3.12.0"
            }
        }
    }

    # Create directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), "MetaMathQA", "temporary_results")
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{peft_type.lower()}_experiment_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Save to file
    with open(filepath, "w") as f:
        json.dump(sample_data, f, indent=2)

    print(f"Generated sample JSON file: {filepath}")

def generate_multiple_samples():
    peft_types = ["LORA", "ADAPTION_PROMPT", "PREFIX_TUNING"]
    for peft_type in peft_types:
        for i in range(3):  # Generate 3 samples for each PEFT type
            generate_sample_json(peft_type, i+1)

if __name__ == "__main__":
    generate_multiple_samples() 