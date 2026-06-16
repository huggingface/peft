from peft import IA3Config

# Create an IA3 Configuration
# Target modules are the specific parts of the neural network we want to modify.
# For IA3, we scale the activations of the attention and feed-forward layers.
config = IA3Config(
    target_modules=["q_proj", "k_proj", "v_proj", "down_proj"],
    feedforward_modules=["down_proj"],
    task_type="CAUSAL_LM"
)

# We save the config to the required directory inside the benchmarking framework
output_path = "method_comparison/MetaMathQA/experiments/ia3/llama-3.2-3B-test"
import os
os.makedirs(output_path, exist_ok=True)
config.save_pretrained(output_path)

print(f"IA3 Config successfully generated and saved to: {output_path}")
