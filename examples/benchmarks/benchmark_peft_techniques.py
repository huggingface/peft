"""
Benchmark script comparing PEFT fine-tuning techniques for conversational AI.

Compares: LoRA, AdaLoRA, IA3, VeRA, LoHa
Base model: meta-llama/Llama-3.2-1B (or any causal LM)
Dataset: OpenAssistant/oasst1

Usage:
    python benchmark_peft_techniques.py --model_name meta-llama/Llama-3.2-1B
    python benchmark_peft_techniques.py --model_name microsoft/phi-2 --techniques lora adalora
    python benchmark_peft_techniques.py --techniques lora ia3 vera --max_samples 500

Outputs:
    - results/peft_benchmark/benchmark_results.json
    - results/peft_benchmark/benchmark_summary.md

Related issue: https://github.com/huggingface/peft/issues/2310
"""

import argparse
import json
import time
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import (
    get_peft_model,
    LoraConfig,
    AdaLoraConfig,
    IA3Config,
    VeraConfig,
    LoHaConfig,
    TaskType,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TECHNIQUE_CONFIGS = {
    "lora": lambda: LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    ),
    "adalora": lambda: AdaLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    ),
    "ia3": lambda: IA3Config(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["k_proj", "v_proj", "down_proj"],
        feedforward_modules=["down_proj"],
    ),
    "vera": lambda: VeraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=256,
        target_modules=["q_proj", "v_proj"],
    ),
    "loha": lambda: LoHaConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        alpha=32,
        target_modules=["q_proj", "v_proj"],
    ),
}


@dataclass
class BenchmarkResult:
    technique: str
    trainable_params: int
    total_params: int
    trainable_pct: float
    peak_gpu_memory_mb: float
    training_time_sec: float
    train_loss: float
    eval_loss: float
    samples_per_second: float


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_dataset(tokenizer, max_length=512, max_samples=2000):
    """Load and tokenize OpenAssistant conversation dataset."""
    dataset = load_dataset("OpenAssistant/oasst1", split="train")

    # Filter to only English messages
    dataset = dataset.filter(lambda x: x["lang"] == "en")
    dataset = dataset.select(range(min(max_samples, len(dataset))))

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    split = tokenized.train_test_split(test_size=0.1, seed=42)
    return split["train"], split["test"]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def run_benchmark(
    technique_name,
    model_name,
    train_dataset,
    eval_dataset,
    tokenizer,
    output_dir,
    num_epochs=1,
    batch_size=4,
    gradient_accumulation_steps=4,
):
    """Run a single PEFT technique benchmark."""
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {technique_name.upper()}")
    print(f"{'='*60}")

    # Reset GPU memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Load fresh base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Apply PEFT
    peft_config = TECHNIQUE_CONFIGS[technique_name]()
    model = get_peft_model(model, peft_config)

    trainable, total = count_parameters(model)
    trainable_pct = 100.0 * trainable / total

    print(f"  Trainable params: {trainable:,} / {total:,} ({trainable_pct:.4f}%)")
    model.print_trainable_parameters()

    # Training arguments
    run_output = os.path.join(output_dir, technique_name)
    training_args = TrainingArguments(
        output_dir=run_output,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        bf16=True,
        report_to="none",
        dataloader_pin_memory=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train and measure
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time

    # Evaluate
    eval_result = trainer.evaluate()

    # GPU memory
    peak_memory_mb = 0.0
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    result = BenchmarkResult(
        technique=technique_name,
        trainable_params=trainable,
        total_params=total,
        trainable_pct=trainable_pct,
        peak_gpu_memory_mb=round(peak_memory_mb, 2),
        training_time_sec=round(training_time, 2),
        train_loss=round(train_result.training_loss, 4),
        eval_loss=round(eval_result["eval_loss"], 4),
        samples_per_second=round(train_result.metrics.get("train_samples_per_second", 0), 2),
    )

    print(f"\n  Results for {technique_name}:")
    print(f"    Train Loss:     {result.train_loss}")
    print(f"    Eval Loss:      {result.eval_loss}")
    print(f"    Peak GPU Mem:   {result.peak_gpu_memory_mb:.0f} MB")
    print(f"    Training Time:  {result.training_time_sec:.0f}s")
    print(f"    Samples/sec:    {result.samples_per_second}")

    # Cleanup
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------


def generate_markdown_table(results, model_name):
    """Generate a markdown comparison table."""
    lines = [
        f"# PEFT Techniques Benchmark",
        f"",
        f"**Model:** `{model_name}`  ",
        f"**Dataset:** OpenAssistant/oasst1 (English subset)  ",
        f"**Training:** 1 epoch, lr=2e-4, bf16, batch_size=4, grad_accum=4",
        f"",
        "| Technique | Trainable Params | % of Total | Peak GPU (MB) | Train Time (s) | Train Loss | Eval Loss | Samples/s |",
        "|-----------|-----------------|------------|---------------|----------------|------------|-----------|-----------|",
    ]

    for r in sorted(results, key=lambda x: x.eval_loss):
        lines.append(
            f"| **{r.technique}** | {r.trainable_params:,} | {r.trainable_pct:.4f}% | "
            f"{r.peak_gpu_memory_mb:.0f} | {r.training_time_sec:.0f} | "
            f"{r.train_loss:.4f} | {r.eval_loss:.4f} | {r.samples_per_second:.1f} |"
        )

    lines.extend([
        "",
        "## Key Takeaways",
        "",
        "- **Lowest eval loss**: Best quality fine-tuning",
        "- **Lowest GPU memory**: Best for constrained hardware",
        "- **Fewest trainable params**: Most parameter-efficient",
        "- **Highest samples/s**: Fastest training throughput",
        "",
        "## Technique Configurations",
        "",
        "| Technique | Key Hyperparameters |",
        "|-----------|-------------------|",
        "| LoRA | r=16, alpha=32, target=[q_proj, v_proj] |",
        "| AdaLoRA | r=16, alpha=32, adaptive rank allocation |",
        "| IA3 | target=[k_proj, v_proj, down_proj], learned rescaling |",
        "| VeRA | r=256, shared random projection matrices |",
        "| LoHa | r=16, alpha=32, Hadamard product low-rank adapters |",
        "",
        "## How to Reproduce",
        "",
        "```bash",
        "# Run all techniques",
        "python examples/benchmarks/benchmark_peft_techniques.py \\",
        f"    --model_name {model_name}",
        "",
        "# Run specific techniques",
        "python examples/benchmarks/benchmark_peft_techniques.py \\",
        f"    --model_name {model_name} \\",
        "    --techniques lora ia3 vera",
        "```",
        "",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark PEFT fine-tuning techniques")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--techniques",
        nargs="+",
        default=list(TECHNIQUE_CONFIGS.keys()),
        choices=list(TECHNIQUE_CONFIGS.keys()),
        help="Techniques to benchmark",
    )
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="results/peft_benchmark")
    args = parser.parse_args()

    print(f"PEFT Benchmark")
    print(f"Model: {args.model_name}")
    print(f"Techniques: {args.techniques}")

    # Setup
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, eval_ds = prepare_dataset(tokenizer, args.max_length, args.max_samples)
    print(f"Dataset: {len(train_ds)} train, {len(eval_ds)} eval samples")

    # Run benchmarks
    results = []
    for technique in args.techniques:
        try:
            result = run_benchmark(
                technique_name=technique,
                model_name=args.model_name,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                tokenizer=tokenizer,
                output_dir=args.output_dir,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
            )
            results.append(result)
        except Exception as e:
            print(f"\n  FAILED: {technique} -- {e}")
            continue

    if not results:
        print("No techniques completed successfully.")
        return

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    results_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {results_path}")

    md_table = generate_markdown_table(results, args.model_name)
    md_path = os.path.join(args.output_dir, "benchmark_summary.md")
    with open(md_path, "w") as f:
        f.write(md_table)
    print(f"Summary saved to {md_path}")

    print("\n" + md_table)


if __name__ == "__main__":
    main()
