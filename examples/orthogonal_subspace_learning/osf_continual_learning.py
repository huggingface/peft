# Copyright 2025-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
OSF Continual Learning Example

This script demonstrates OSF's ability to learn multiple tasks sequentially while preventing
catastrophic forgetting, compared to standard full fine-tuning.

Tasks:
1. ScienceQA - Science question answering
2. NumGLUE - Mathematical reasoning
3. FOMC - Financial sentiment classification

OSF Configuration:
- Task 1: effective_rank=0.3 (train 70%, freeze 30%)
- Task 2: effective_rank=0.5 (train 50%, freeze 50%)
- Task 3: effective_rank=0.7 (train 30%, freeze 70%)
"""

import argparse
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from utils import (
    DataCollatorForCompletionOnly,
    format_fomc_for_llama,
    format_numglue_for_llama,
    format_scienceqa_for_llama,
    load_fomc,
    load_numglue,
    load_scienceqa,
)

from peft import OSFConfig, get_peft_model


def compute_accuracy_scienceqa(model, eval_dataset, tokenizer, data_collator):
    """Compute accuracy for ScienceQA (extract predicted letter)."""
    model.eval()
    correct = 0
    total = 0

    # Create a simple dataloader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(eval_dataset, batch_size=8, collate_fn=data_collator)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"]

            # Generate predictions
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=5,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )

            # Extract predictions and ground truth
            for i in range(len(outputs)):
                # Decode the generated text
                generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)

                # Extract the answer (last letter in the generated text)
                # Look for single capital letters A, B, C, D
                matches = re.findall(r"\b([A-D])\b", generated_text)
                pred = matches[-1] if matches else "X"

                # Get ground truth (find the label that's not -100)
                label_ids = labels[i][labels[i] != -100]
                if len(label_ids) > 0:
                    gt = tokenizer.decode(label_ids, skip_special_tokens=True).strip()
                    if pred == gt:
                        correct += 1
                    total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def compute_accuracy_numglue(model, eval_dataset, tokenizer, data_collator):
    """Compute accuracy for NumGLUE (extract predicted number)."""
    model.eval()
    correct = 0
    total = 0

    from torch.utils.data import DataLoader

    dataloader = DataLoader(eval_dataset, batch_size=8, collate_fn=data_collator)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"]

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=20,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )

            for i in range(len(outputs)):
                generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)

                # Extract number from generated text
                numbers = re.findall(r"-?\d+\.?\d*", generated_text)
                pred = numbers[-1] if numbers else "-999"

                # Get ground truth
                label_ids = labels[i][labels[i] != -100]
                if len(label_ids) > 0:
                    gt = tokenizer.decode(label_ids, skip_special_tokens=True).strip()
                    if pred == gt:
                        correct += 1
                    total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def compute_accuracy_fomc(model, eval_dataset, tokenizer, data_collator):
    """Compute accuracy for FOMC (extract predicted sentiment)."""
    model.eval()
    correct = 0
    total = 0

    from torch.utils.data import DataLoader

    dataloader = DataLoader(eval_dataset, batch_size=8, collate_fn=data_collator)

    valid_labels = ["Dovish", "Hawkish", "Neutral"]

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"]

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )

            for i in range(len(outputs)):
                generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)

                # Extract sentiment label
                pred = None
                for label in valid_labels:
                    if label in generated_text:
                        pred = label
                        break

                # Get ground truth
                label_ids = labels[i][labels[i] != -100]
                if len(label_ids) > 0:
                    gt = tokenizer.decode(label_ids, skip_special_tokens=True).strip()
                    if pred == gt:
                        correct += 1
                    total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def evaluate_model(model, eval_dataset, data_collator, tokenizer, task_name, task_type):
    """Evaluate model on a dataset and return loss and accuracy."""
    # Compute loss
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        eval_dataset=eval_dataset,
        args=TrainingArguments(
            label_names=["labels"],
        ),
    )
    results = trainer.evaluate()
    loss = results["eval_loss"]

    # Compute accuracy based on task type
    if task_type == "scienceqa":
        accuracy = compute_accuracy_scienceqa(model, eval_dataset, tokenizer, data_collator)
    elif task_type == "numglue":
        accuracy = compute_accuracy_numglue(model, eval_dataset, tokenizer, data_collator)
    elif task_type == "fomc":
        accuracy = compute_accuracy_fomc(model, eval_dataset, tokenizer, data_collator)
    else:
        accuracy = 0.0

    print(f"  {task_name}: Loss = {loss:.4f}, Accuracy = {accuracy * 100:.2f}%")
    return loss, accuracy


def train_with_osf(
    model_name,
    num_train,
    num_eval,
    output_dir,
    num_epochs,
    learning_rate,
    batch_size,
    gradient_accumulation_steps,
    max_length,
    seed,
):
    """Train using OSF with progressive rank allocation."""
    print("\n" + "=" * 80)
    print("TRAINING WITH OSF (Orthogonal Subspace Fine-tuning)")
    print("=" * 80)

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    # Load all datasets with task-specific sizes
    # FOMC only has 496 samples total, so we use 350 train + 146 eval for it
    print("\nLoading datasets...")
    scienceqa_train, scienceqa_eval = load_scienceqa(1000, 200, seed)
    numglue_train, numglue_eval = load_numglue(1000, 200, seed)
    fomc_train, fomc_eval = load_fomc(350, 146, seed)

    # Store original eval datasets for later
    scienceqa_eval_original = scienceqa_eval
    numglue_eval_original = numglue_eval
    fomc_eval_original = fomc_eval

    # Format datasets
    scienceqa_train = scienceqa_train.map(
        lambda x: format_scienceqa_for_llama(x, tokenizer, max_length),
        batched=True,
        remove_columns=scienceqa_train.column_names,
    )
    scienceqa_eval = scienceqa_eval.map(
        lambda x: format_scienceqa_for_llama(x, tokenizer, max_length),
        batched=True,
        remove_columns=scienceqa_eval.column_names,
    )

    numglue_train = numglue_train.map(
        lambda x: format_numglue_for_llama(x, tokenizer, max_length),
        batched=True,
        remove_columns=numglue_train.column_names,
    )
    numglue_eval = numglue_eval.map(
        lambda x: format_numglue_for_llama(x, tokenizer, max_length),
        batched=True,
        remove_columns=numglue_eval.column_names,
    )

    fomc_train = fomc_train.map(
        lambda x: format_fomc_for_llama(x, tokenizer, max_length), batched=True, remove_columns=fomc_train.column_names
    )
    fomc_eval = fomc_eval.map(
        lambda x: format_fomc_for_llama(x, tokenizer, max_length), batched=True, remove_columns=fomc_eval.column_names
    )

    data_collator = DataCollatorForCompletionOnly(tokenizer, max_length)

    # Task configurations
    tasks = [
        {
            "name": "ScienceQA",
            "train": scienceqa_train,
            "eval": scienceqa_eval,
            "eval_original": scienceqa_eval_original,
            "effective_rank": 0.3,  # Freeze 30%, train 70%
            "type": "scienceqa",
        },
        {
            "name": "NumGLUE",
            "train": numglue_train,
            "eval": numglue_eval,
            "eval_original": numglue_eval_original,
            "effective_rank": 0.5,  # Freeze 50%, train 50%
            "type": "numglue",
        },
        {
            "name": "FOMC",
            "train": fomc_train,
            "eval": fomc_eval,
            "eval_original": fomc_eval_original,
            "effective_rank": 0.7,  # Freeze 70%, train 30%
            "type": "fomc",
        },
    ]

    # Store evaluation history: {task_name: [(loss, accuracy), ...]}
    eval_history = {
        "ScienceQA": [],
        "NumGLUE": [],
        "FOMC": [],
    }

    # Sequential task training
    model = base_model
    for task_idx, task in enumerate(tasks):
        print(f"\n{'=' * 80}")
        print(f"TASK {task_idx + 1}: {task['name']}")
        print(f"Effective Rank: {task['effective_rank']} (preserving {task['effective_rank'] * 100:.0f}%)")
        print(f"{'=' * 80}")

        # Configure OSF for this task
        config = OSFConfig(
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            effective_rank=task["effective_rank"],
        )

        # Apply OSF to the model
        model = get_peft_model(model, config)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/osf_{task['name'].lower()}",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            logging_steps=10,
            save_strategy="no",
            bf16=True,
            remove_unused_columns=False,
        )

        # Train on current task
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=task["train"],
            data_collator=data_collator,
        )

        print(f"\nTraining on {task['name']}...")
        trainer.train()

        # Evaluate on all tasks seen so far
        print(f"\nEvaluating on all tasks after training on {task['name']}:")
        for eval_task_idx in range(task_idx + 1):
            eval_task = tasks[eval_task_idx]
            loss, accuracy = evaluate_model(
                model, eval_task["eval"], data_collator, tokenizer, eval_task["name"], eval_task["type"]
            )
            eval_history[eval_task["name"]].append((loss, accuracy))

        # Unload OSF to get the updated base model for next task (if not last task)
        if task_idx < len(tasks) - 1:
            print("\nUnloading OSF adapter to prepare for next task...")
            model = model.unload()

    # Save final model
    final_model_path = f"{output_dir}/osf_final"
    model.save_pretrained(final_model_path)
    print(f"\nFinal OSF model saved to {final_model_path}")

    return eval_history


def train_full_finetuning(
    model_name,
    num_train,
    num_eval,
    output_dir,
    num_epochs,
    learning_rate,
    batch_size,
    gradient_accumulation_steps,
    max_length,
    seed,
):
    """Train using standard full fine-tuning (baseline for comparison)."""
    print("\n" + "=" * 80)
    print("TRAINING WITH FULL FINE-TUNING (Baseline)")
    print("=" * 80)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load all datasets with task-specific sizes
    # FOMC only has 496 samples total, so we use 350 train + 146 eval for it
    print("\nLoading datasets...")
    scienceqa_train, scienceqa_eval = load_scienceqa(1000, 200, seed)
    numglue_train, numglue_eval = load_numglue(1000, 200, seed)
    fomc_train, fomc_eval = load_fomc(350, 146, seed)

    # Store original eval datasets
    scienceqa_eval_original = scienceqa_eval
    numglue_eval_original = numglue_eval
    fomc_eval_original = fomc_eval

    # Format datasets
    scienceqa_train = scienceqa_train.map(
        lambda x: format_scienceqa_for_llama(x, tokenizer, max_length),
        batched=True,
        remove_columns=scienceqa_train.column_names,
    )
    scienceqa_eval = scienceqa_eval.map(
        lambda x: format_scienceqa_for_llama(x, tokenizer, max_length),
        batched=True,
        remove_columns=scienceqa_eval.column_names,
    )

    numglue_train = numglue_train.map(
        lambda x: format_numglue_for_llama(x, tokenizer, max_length),
        batched=True,
        remove_columns=numglue_train.column_names,
    )
    numglue_eval = numglue_eval.map(
        lambda x: format_numglue_for_llama(x, tokenizer, max_length),
        batched=True,
        remove_columns=numglue_eval.column_names,
    )

    fomc_train = fomc_train.map(
        lambda x: format_fomc_for_llama(x, tokenizer, max_length), batched=True, remove_columns=fomc_train.column_names
    )
    fomc_eval = fomc_eval.map(
        lambda x: format_fomc_for_llama(x, tokenizer, max_length), batched=True, remove_columns=fomc_eval.column_names
    )

    data_collator = DataCollatorForCompletionOnly(tokenizer, max_length)

    tasks = [
        {"name": "ScienceQA", "train": scienceqa_train, "eval": scienceqa_eval, "type": "scienceqa"},
        {"name": "NumGLUE", "train": numglue_train, "eval": numglue_eval, "type": "numglue"},
        {"name": "FOMC", "train": fomc_train, "eval": fomc_eval, "type": "fomc"},
    ]

    # Store evaluation history
    eval_history = {
        "ScienceQA": [],
        "NumGLUE": [],
        "FOMC": [],
    }

    # Load base model once
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    # Sequential task training
    for task_idx, task in enumerate(tasks):
        print(f"\n{'=' * 80}")
        print(f"TASK {task_idx + 1}: {task['name']}")
        print(f"{'=' * 80}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/full_{task['name'].lower()}",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            logging_steps=10,
            save_strategy="no",
            bf16=True,
            remove_unused_columns=False,
        )

        # Train on current task
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=task["train"],
            data_collator=data_collator,
        )

        print(f"\nTraining on {task['name']}...")
        trainer.train()

        # Evaluate on all tasks seen so far
        print(f"\nEvaluating on all tasks after training on {task['name']}:")
        for eval_task_idx in range(task_idx + 1):
            eval_task = tasks[eval_task_idx]
            loss, accuracy = evaluate_model(
                model, eval_task["eval"], data_collator, tokenizer, eval_task["name"], eval_task["type"]
            )
            eval_history[eval_task["name"]].append((loss, accuracy))

    # Save final model
    final_model_path = f"{output_dir}/full_final"
    model.save_pretrained(final_model_path)
    print(f"\nFinal full fine-tuning model saved to {final_model_path}")

    return eval_history


def print_results_comparison(osf_history, full_history):
    """Print comparison table of OSF vs Full Fine-tuning."""
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON: OSF vs Full Fine-tuning")
    print("=" * 80)

    tasks = ["ScienceQA", "NumGLUE", "FOMC"]

    # Print detailed results
    print("\n" + "-" * 80)
    print("DETAILED RESULTS (Accuracy %)")
    print("-" * 80)
    print(f"{'Task':<15} {'After Task':<15} {'OSF Acc %':<15} {'Full FT Acc %':<15} {'Difference':<15}")
    print("-" * 80)

    for task_idx, task in enumerate(tasks):
        for eval_after_idx in range(task_idx, len(tasks)):
            eval_after = tasks[eval_after_idx]
            osf_acc = osf_history[task][eval_after_idx - task_idx][1] * 100
            full_acc = full_history[task][eval_after_idx - task_idx][1] * 100
            diff = osf_acc - full_acc

            print(
                f"{task:<15} {eval_after:<15} {osf_acc:<15.2f} {full_acc:<15.2f} {diff:+15.2f}{'  (OSF better)' if diff > 0 else ''}"
            )

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY METRICS")
    print("=" * 80)

    # Final average accuracy across all 3 tasks
    osf_final_accs = [osf_history[task][-1][1] * 100 for task in tasks]
    full_final_accs = [full_history[task][-1][1] * 100 for task in tasks]

    osf_avg_final = sum(osf_final_accs) / len(osf_final_accs)
    full_avg_final = sum(full_final_accs) / len(full_final_accs)

    print("\n1. Average Accuracy Across All 3 Tasks (After Final Task):")
    print(f"   OSF:     {osf_avg_final:.2f}%")
    print(f"   Full FT: {full_avg_final:.2f}%")
    print(
        f"   Difference: {osf_avg_final - full_avg_final:+.2f}% {'(OSF better)' if osf_avg_final > full_avg_final else '(Full FT better)'}"
    )

    # Average forgetting (for tasks 1 and 2 only, since task 3 is the final task)
    print("\n2. Average Forgetting (Task 1 & 2):")
    print("   Forgetting = Final Accuracy - Initial Accuracy (negative is worse)\n")

    osf_forgetting_vals = []
    full_forgetting_vals = []

    for task_idx, task in enumerate(tasks[:-1]):  # Exclude last task
        osf_initial_acc = osf_history[task][0][1] * 100  # Right after learning task
        osf_final_acc = osf_history[task][-1][1] * 100  # After learning all tasks
        osf_forgetting = osf_final_acc - osf_initial_acc

        full_initial_acc = full_history[task][0][1] * 100
        full_final_acc = full_history[task][-1][1] * 100
        full_forgetting = full_final_acc - full_initial_acc

        osf_forgetting_vals.append(osf_forgetting)
        full_forgetting_vals.append(full_forgetting)

        print(f"   {task}:")
        print(f"     OSF:     {osf_forgetting:+.2f}% (initial: {osf_initial_acc:.2f}% → final: {osf_final_acc:.2f}%)")
        print(
            f"     Full FT: {full_forgetting:+.2f}% (initial: {full_initial_acc:.2f}% → final: {full_final_acc:.2f}%)"
        )
        print(
            f"     Difference: {osf_forgetting - full_forgetting:+.2f}% {'(OSF better)' if osf_forgetting > full_forgetting else '(Full FT better)'}\n"
        )

    osf_avg_forgetting = sum(osf_forgetting_vals) / len(osf_forgetting_vals)
    full_avg_forgetting = sum(full_forgetting_vals) / len(full_forgetting_vals)

    print("   Average Forgetting:")
    print(f"     OSF:     {osf_avg_forgetting:+.2f}%")
    print(f"     Full FT: {full_avg_forgetting:+.2f}%")
    print(
        f"     Difference: {osf_avg_forgetting - full_avg_forgetting:+.2f}% {'(OSF better)' if osf_avg_forgetting > full_avg_forgetting else '(Full FT better)'}"
    )

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="OSF Continual Learning Example")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or path",
    )
    parser.add_argument("--num_train", type=int, default=1000, help="Number of training samples per task")
    parser.add_argument("--num_eval", type=int, default=200, help="Number of evaluation samples per task")
    parser.add_argument("--output_dir", type=str, default="./osf_continual_learning_outputs", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs per task")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--run_baseline",
        action="store_true",
        help="Also run full fine-tuning baseline for comparison",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Train with OSF
    osf_history = train_with_osf(
        args.model_name,
        args.num_train,
        args.num_eval,
        args.output_dir,
        args.num_epochs,
        args.learning_rate,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.max_length,
        args.seed,
    )

    # Optionally train with full fine-tuning baseline
    if args.run_baseline:
        full_history = train_full_finetuning(
            args.model_name,
            args.num_train,
            args.num_eval,
            args.output_dir,
            args.num_epochs,
            args.learning_rate,
            args.batch_size,
            args.gradient_accumulation_steps,
            args.max_length,
            args.seed,
        )

        # Print comparison
        print_results_comparison(osf_history, full_history)
    else:
        print("\n" + "=" * 80)
        print("OSF TRAINING COMPLETE")
        print("=" * 80)
        print("\nTo compare with full fine-tuning baseline, run with --run_baseline flag")

        # Print OSF-only summary
        tasks = ["ScienceQA", "NumGLUE", "FOMC"]
        print("\n" + "=" * 80)
        print("OSF SUMMARY METRICS")
        print("=" * 80)

        # Final average accuracy
        osf_final_accs = [osf_history[task][-1][1] * 100 for task in tasks]
        osf_avg_final = sum(osf_final_accs) / len(osf_final_accs)

        print(f"\n1. Average Accuracy Across All 3 Tasks (After Final Task): {osf_avg_final:.2f}%")
        for task, acc in zip(tasks, osf_final_accs):
            print(f"   {task}: {acc:.2f}%")

        # Average forgetting
        print("\n2. Average Forgetting (Task 1 & 2):")
        osf_forgetting_vals = []
        for task_idx, task in enumerate(tasks[:-1]):
            osf_initial_acc = osf_history[task][0][1] * 100
            osf_final_acc = osf_history[task][-1][1] * 100
            osf_forgetting = osf_initial_acc - osf_final_acc
            osf_forgetting_vals.append(osf_forgetting)

            print(f"   {task}: {osf_forgetting:+.2f}% (initial: {osf_initial_acc:.2f}% → final: {osf_final_acc:.2f}%)")

        osf_avg_forgetting = sum(osf_forgetting_vals) / len(osf_forgetting_vals)
        print(f"   Average: {osf_avg_forgetting:+.2f}%")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
