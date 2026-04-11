"""Quick standalone eval of a saved checkpoint with adapter key remapping."""

import json
import os
import sys

import safetensors.torch as st
import torch
from data import load_rl_datasets
from reward import extract_boxed, safe_grade
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel


def remap_adapter_keys(checkpoint_dir: str) -> bool:
    adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    tensors = st.load_file(adapter_path)
    needs_remap = any("language_model." in k for k in tensors)
    if not needs_remap:
        print("Keys already correct, no remapping needed.")
        return False
    remapped = {}
    for k, v in tensors.items():
        new_key = k.replace(".language_model.", ".")
        remapped[new_key] = v
    st.save_file(remapped, adapter_path)
    print(f"Remapped {len(remapped)} keys (removed 'language_model.' prefix)")

    # Also remap rank_pattern / lora_alpha in adapter_config.json (AdaLoRA)
    config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        for field in ("rank_pattern", "lora_alpha"):
            mapping = cfg.get(field)
            if isinstance(mapping, dict) and any("language_model." in k for k in mapping):
                cfg[field] = {k.replace("language_model.", ""): v for k, v in mapping.items()}
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

    return True


@torch.no_grad()
def evaluate(model, tokenizer, dataset, max_new_tokens=1024):
    correct = 0
    total = 0
    for i, row in enumerate(dataset):
        inputs = tokenizer(row["prompt"], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        try:
            pred = extract_boxed(completion)
            is_correct = safe_grade(pred, row["ground_truth"])
        except ValueError:
            is_correct = False
        correct += int(is_correct)
        total += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(dataset)}] running accuracy: {correct}/{total} = {correct/total:.3f}")
    return correct / max(total, 1)


def main():
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/lora--qwen3.5-4b-gsm8k/eval_checkpoint"
    model_id = "Qwen/Qwen3.5-4B"
    eval_size = 50

    print(f"Checkpoint: {checkpoint_dir}")
    print("Step 1: Remap adapter keys if needed...")
    remap_adapter_keys(checkpoint_dir)

    print("Step 2: Load dataset...")
    _, test_ds = load_rl_datasets(
        dataset_name="gsm8k", dataset_config="main",
        train_split="train", test_split="test",
        train_subset_size=100, eval_subset_size=eval_size, seed=42,
    )
    test_ds = test_ds.select(range(min(eval_size, len(test_ds))))
    print(f"  Test samples: {len(test_ds)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Step 3: Load base model + merge adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model = model.merge_and_unload()
    model.eval()

    print("Step 4: Evaluate...")
    acc = evaluate(model, tokenizer, test_ds)
    print(f"\nAdapter model pass@1: {acc:.3f} ({eval_size} samples)")


if __name__ == "__main__":
    main()
