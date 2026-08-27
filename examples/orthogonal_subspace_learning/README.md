# Orthogonal Subspace Fine-tuning (OSF) - Continual Learning Example

This example demonstrates OSF's ability to learn multiple tasks sequentially while preventing catastrophic forgetting, a key challenge in continual learning.

## Introduction

**Orthogonal Subspace Fine-tuning (OSF)** is a parameter-efficient fine-tuning method designed specifically for continual learning scenarios. Unlike traditional fine-tuning which suffers from catastrophic forgetting when learning new tasks, OSF constrains parameter updates to be orthogonal to previously important directions, effectively preserving knowledge from earlier tasks.

### Key Features

- **Prevents Catastrophic Forgetting**: Maintains performance on previous tasks while learning new ones
- **Full Model Capacity**: Unlike LoRA-based methods, OSF allows full-rank updates within the trainable subspace
- **Progressive Budget Allocation**: Gradually allocates more capacity to preserve previous knowledge
- **No Additional Parameters**: Modifies weights in-place without adding extra parameters per task

## Quick Start

### Installation

```bash
pip install -e ".[dev]"
```

### Basic Usage

Run the continual learning example with OSF:

```bash
python osf_continual_learning.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --num_train 1000 \
    --num_eval 200 \
    --num_epochs 2 \
    --output_dir ./outputs
```

To compare with full fine-tuning baseline:

```bash
python osf_continual_learning.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --run_baseline \
    --output_dir ./outputs
```

## Continual Learning Scenario

This example trains a model on three different tasks sequentially:

1. **ScienceQA** - Science question answering across natural, language, and social sciences
2. **NumGLUE** - Mathematical reasoning and numerical understanding
3. **FOMC** - Financial sentiment classification (Dovish/Hawkish/Neutral)

### Progressive Capacity Allocation

OSF uses a progressive budget allocation strategy where each task gets decreasing trainable capacity while preserving more knowledge from previous tasks:

| Task | Effective Rank | Preserved | Trainable | Description |
|------|----------------|-----------|-----------|-------------|
| Task 1 (ScienceQA) | 0.3 | 30% | 70% | Maximum capacity for first task |
| Task 2 (NumGLUE) | 0.5 | 50% | 50% | Balanced capacity allocation |
| Task 3 (FOMC) | 0.7 | 70% | 30% | Minimal capacity, maximum preservation |

This allocation ensures:
- Early tasks get sufficient capacity to learn effectively
- Later tasks can still learn new patterns
- Previous knowledge is progressively protected from interference

## How OSF Works

OSF decomposes each weight matrix using SVD into high-rank (preserved) and low-rank (trainable) components:

```
W = U_high @ S_high @ V_high^T + U_low @ S_low @ V_low^T
    └─────────┬─────────┘        └──────┬──────┘
          frozen                    trainable
     (previous tasks)              (current task)
```

During training:
1. **Initialization**: Perform SVD on each weight matrix
2. **Partitioning**: Split singular values based on `effective_rank`
3. **Freezing**: Freeze top-k singular directions (high-rank subspace)
4. **Training**: Update remaining directions (low-rank subspace)
5. **Gradient Projection**: Ensure updates are orthogonal to frozen subspace

Between tasks:
1. **Unload**: Merge OSF components back into base model
2. **Re-initialize**: Perform fresh SVD with increased `effective_rank`
3. **Continue**: Train on next task with larger frozen subspace

## Command Line Arguments

```
--model_name              Model to use (default: meta-llama/Llama-3.1-8B-Instruct)
--num_train              Number of training samples per task (default: 1000)
--num_eval               Number of evaluation samples per task (default: 200)
--output_dir             Directory for outputs (default: ./osf_continual_learning_outputs)
--num_epochs             Training epochs per task (default: 2)
--learning_rate          Learning rate (default: 5e-6)
--batch_size             Batch size per device (default: 32)
--gradient_accumulation_steps  Gradient accumulation (default: 1)
--max_length             Maximum sequence length (default: 512)
--seed                   Random seed (default: 42)
--run_baseline           Also run full fine-tuning baseline for comparison
```

## Expected Results

### OSF Performance

When using OSF (with 2 epochs per task), you should observe:
- **Reduced catastrophic forgetting**: Performance on earlier tasks degrades less compared to full fine-tuning
- **Continued learning**: Model successfully learns each new task
- **Better retention**: OSF maintains higher average accuracy across all tasks

### Full Fine-tuning Baseline

Standard full fine-tuning typically shows:
- **Catastrophic forgetting**: Significant performance degradation on earlier tasks
- **Last task bias**: Model performs well only on the most recent task
- **Task interference**: New task learning overwrites previous knowledge

## Understanding the Results

### Forgetting Analysis

The script prints a forgetting analysis showing how much earlier task performance changes.

**Example results from training with 2 epochs per task:**

```
SUMMARY METRICS
================================================================================

1. Average Accuracy Across All 3 Tasks (After Final Task):
   OSF:     53.42%
   Full FT: 46.26%
   Difference: +7.17% (OSF better)

2. Average Forgetting (Task 1 & 2):
   Forgetting = Final Accuracy - Initial Accuracy (negative is worse)

   ScienceQA:
     OSF:     +30.50% (initial: 55.00% → final: 85.50%)
     Full FT: -13.00% (initial: 84.50% → final: 71.50%)
     Difference: +43.50% (OSF better)

   NumGLUE:
     OSF:     +30.00% (initial: 16.00% → final: 46.00%)
     Full FT: +1.00% (initial: 37.50% → final: 38.50%)
     Difference: +29.00% (OSF better)

   Average Forgetting:
     OSF:     +30.25%
     Full FT: -6.00%
     Difference: +36.25% (OSF better)
```

**Interpreting Forgetting Metrics:**
- **Negative values** = Forgetting occurred (performance decreased)
- **Positive values** = Backward transfer occurred (performance improved)
- **Values closer to 0** = Better retention

In this example, OSF shows significant positive backward transfer (+30.25% average), while Full FT shows slight forgetting (-6.00% average). This demonstrates OSF's ability to not only prevent catastrophic forgetting but also enable beneficial knowledge transfer across tasks.

## Advanced Usage

### Custom Task Configuration

You can modify the tasks and capacity allocation in the script:

```python
tasks = [
    {
        "name": "Task1",
        "train": task1_train,
        "eval": task1_eval,
        "effective_rank": 0.2,  # Freeze 20%, train 80%
    },
    {
        "name": "Task2",
        "train": task2_train,
        "eval": task2_eval,
        "effective_rank": 0.6,  # Freeze 60%, train 40%
    },
]
```

### Using Different Models

OSF works with any transformer-based model:

```bash
# Smaller model for faster experimentation
python osf_continual_learning.py --model_name gpt2

# Different LLaMA variant
python osf_continual_learning.py --model_name meta-llama/Llama-2-7b-hf
```

### Adjusting Target Modules

In the script, you can modify which modules to apply OSF to:

```python
config = OSFConfig(
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention only
    effective_rank=task["effective_rank"],
)
```

Common configurations:
- **Attention only**: `["q_proj", "k_proj", "v_proj", "o_proj"]`
- **Attention + MLP**: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- **All linear**: `target_modules="all-linear"`

## Customization

### Adding Your Own Tasks

To add custom tasks, create data loading and formatting functions in `utils.py`:

```python
def load_my_task(num_train=1000, num_eval=200, seed=42):
    """Load your custom dataset."""
    dataset = load_dataset("your/dataset")
    # ... split and return
    return train_dataset, eval_dataset

def format_my_task_for_llama(examples, tokenizer, max_length=512):
    """Format your task for instruction following."""
    prompts = []
    labels_text = []

    for i in range(len(examples)):
        prompt = f"Your instruction template: {examples['input'][i]}"
        label = examples['output'][i]

        prompts.append(prompt)
        labels_text.append(label)

    # ... tokenization logic
    return formatted_examples
```

Then add to the tasks list in `osf_continual_learning.py`.

## Performance Tips

### Memory Optimization

For large models, consider:
- Reducing `batch_size` and increasing `gradient_accumulation_steps`
- Using smaller `max_length`
- Enabling gradient checkpointing (add to model before OSF):
  ```python
  model.gradient_checkpointing_enable()
  ```

### Training Speed

To speed up training:
- Reduce `num_train` and `num_eval` for initial testing
- Use smaller models (e.g., `gpt2` or `Llama-2-7b`)
- Reduce `max_length` for shorter sequences

### Better Results

For improved continual learning performance:
- Play around with `num_epochs` per task (try 2-3 epochs)
- Adjust `learning_rate`
- Experiment with different capacity allocation strategies

## Citation

If you use OSF in your research, please cite:

```bibtex
@misc{nayak2025sculptingsubspacesconstrainedfinetuning,
      title={Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning}, 
      author={Nikhil Shivakumar Nayak and Krishnateja Killamsetty and Ligong Han and Abhishek Bhandwaldar and Prateek Chanda and Kai Xu and Hao Wang and Aldo Pareja and Oleg Silkin and Mustafa Eyceoz and Akash Srivastava},
      year={2025},
      eprint={2504.07097},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.07097}, 
}
```

## Additional Resources

- [OSF Documentation](../../docs/source/package_reference/osf.md)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Original Paper](https://huggingface.co/papers/2504.07097)

## License

This example is licensed under Apache 2.0. See the PEFT repository for full license details.
