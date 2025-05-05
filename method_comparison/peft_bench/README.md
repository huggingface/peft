# PEFT Benchmark Suite

This directory contains tools to benchmark different Parameter-Efficient Fine-Tuning (PEFT) methods on various models and tasks.

## Features

- Measure performance metrics including inference speed, training throughput, and memory usage
- Compare multiple PEFT methods with different configurations
- Easily extensible to add new methods and models
- Supports custom prompts and dataset integration

## Usage

```bash
python -m method_comparison.peft_bench.run --config configs/benchmark_config.json
```

## Configuration

The benchmark is configured through JSON files specifying:
- Models to test
- PEFT methods and configurations
- Prompt categories (short, medium, long)
- Number of inference and training runs
- Hardware settings (quantization options)

See `configs/` directory for example configurations.

## Output

Results are saved in structured JSON format with:
- `run_info`: Basic information about the run (timestamp, hardware, etc.)
- `train_info`: Training metrics (throughput, memory usage, etc.)
- `meta_info`: Information about the model and PEFT method
- Detailed metrics for each prompt category

## Adding New PEFT Methods

To add a new PEFT method, update the configuration file to include the new method's parameters. 