## Base Model Inference Caching

The benchmarking suite uses a separate script, `run_base.py`, to measure base model inference times and save results for reuse. This should be run once per model configuration to avoid redundant computations and ensure consistent baseline metrics for all PEFT experiments.

**Usage:**
```bash
python run_base.py
```
This will cache the base model inference results for the specified configuration. Subsequent runs of `run.py` will automatically load these cached results.

# PEFT Benchmarking Suite

This directory contains a comprehensive benchmarking framework for Parameter-Efficient Fine-Tuning (PEFT) methods. For the task of text generation, the suite measures inference performance, memory usage, and other key metrics across different PEFT configurations.

## Overview

The benchmarking suite provides:
- **Inference time measurement** across different prompt categories
- **Memory usage during inference** (RAM and GPU)
- **Parameter efficiency metrics** (trainable vs total parameters)
- **Time per token analysis** for fair comparison across different generation lengths
- **Structured result logging** with detailed metadata

## Architecture

The suite follows a clean separation between:
1. **Default benchmark configuration** - shared settings for consistent comparison
2. **Individual adapter configurations** - PEFT-specific parameters for each experiment

This ensures that all experiments are comparable while allowing flexibility in adapter parameters.

## Quick Start

### Running a Single Experiment

```bash
# From the peft_bench directory
python run.py experiments/lora/lora_r8 --verbose
```

## Configuration Structure

The benchmarking suite uses a hierarchical configuration system:

1. **Default benchmark parameters** (`default_benchmark_params.json`) - Base configuration shared by all experiments
2. **Experiment-specific overrides** (`benchmark_params.json` in each experiment) - Optional overrides for specific experiments  
3. **Adapter configuration** (`adapter_config.json` in each experiment) - PEFT method parameters

This structure ensures consistent comparison while allowing flexibility where needed.

### Default Configuration (`default_benchmark_params.json`)

Contains shared benchmark settings that apply to all experiments. Here are the key configuration fields:

- `model_id`: The Hugging Face model ID to use as the base model (e.g., "facebook/opt-350m")
- `dtype`: Model precision ("float16", "float32", or "bfloat16")
- `seed`: Random seed for reproducibility
- `max_new_tokens`: Maximum number of tokens to generate during inference
- `num_inference_runs`: Number of inference runs per prompt for statistical reliability
- `use_4bit`: Whether to use 4-bit quantization (bool)
- `use_8bit`: Whether to use 8-bit quantization (bool)

Each experiment can override these settings by providing its own `benchmark_params.json` file.

### Experiment Structure

Each experiment directory should contain:

1. `adapter_config.json`: PEFT adapter configuration. For details on available parameters and their meanings, refer to the [PEFT documentation](https://huggingface.co/docs/peft/main/en/developer_guides/adapters).

2. (Optional) `benchmark_params.json`: Override specific benchmark parameters for this experiment.

Example directory structure:
```
experiments/
└── lora/
    ├── lora_r8/                # LoRA rank 8 experiment
    │   ├── adapter_config.json # PEFT adapter configuration
    │   └── benchmark_params.json # Optional benchmark overrides
    └── lora_r16/               # LoRA rank 16 experiment
        └── adapter_config.json
```

### Experiment-Specific Overrides Example

If an experiment needs different benchmark settings, create `benchmark_params.json`:
```json
{
    "_comment": "Override settings for this specific experiment",
    "max_new_tokens": 50,
    "num_inference_runs": 15,
    "num_prompt_samples": 2
}
```

These parameters will override the defaults from `default_benchmark_params.json`. However, the defaults should generally not be changed to keep the results from the individual experiments comparable.

### Create a New Experiment Adapter Configuration

To create a new experiment, follow these steps:

1. **Create the experiment directory**
   ```bash
   mkdir -p experiments/lora/lora_r8
   ```

2. **Generate the adapter configuration programmatically**
   Use the PEFT library to create and save your adapter config:

   ```python
   from peft import LoraConfig

   config = LoraConfig(
       lora_alpha=16,
       lora_dropout=0.1,
       r=8,
       target_modules=["q_proj", "v_proj"],
       task_type="CAUSAL_LM"
   )
   config.save_pretrained("experiments/lora/lora_r8")
   ```

   This will create an `adapter_config.json` in your experiment directory. Adjust parameters as needed for your experiment.

3. **(Optional) Add benchmark overrides**
   If you need to override default benchmark settings, create a `benchmark_params.json` in the same directory.

4. **Run the benchmark**
   ```bash
   python run.py experiments/lora/lora_r8 --verbose
   ```

## Prompt Categories

The benchmark automatically runs across all prompt categories for consistent comparison:
- **short** - Brief prompts (1-2 sentences)
- **medium** - Moderate length prompts (paragraph-level)
- **long** - Extended prompts (multiple paragraphs)

Results are tracked separately for each category, allowing analysis of how different PEFT methods perform across varying input lengths.

## Results Structure

Results are saved in a structured JSON format with three main sections:

### `run_info`
- Execution metadata (timestamp, duration, status)
- Hardware information (GPU type, CUDA version, etc.)
- Error information (if applicable)
- PEFT and benchmark configurations

### `generation_info`
- Memory usage logs at different stages
- Per-category metrics (inference time, time per token, etc.)
- Overall aggregated metrics
- Individual sample results for detailed analysis

### `meta_info`
- Model information (ID, PEFT method)
- Parameter counts (adapter, total, ratio)
- Model size information (base model, adapter)
- System and package information

## Key Metrics

### Inference Performance
- **Inference Time**: Total time for generation per category
- **Time Per Token**: Normalized time accounting for different generation lengths
- **Inference Overhead**: Percentage increase compared to base model

### Memory Usage
- **Peak GPU Memory**: Maximum GPU memory during benchmark
- **Peak RAM Memory**: Maximum RAM usage
- **Memory Logs**: Detailed tracking at each stage

### Parameter Efficiency
- **Adapter Parameters**: Number of parameters in the PEFT adapter
- **Parameter Ratio**: Percentage of total model parameters that are in the adapter
- **Adapter Size**: Memory footprint of the adapter in MB
