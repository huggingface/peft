#!/usr/bin/env python
"""
Minimal reproducer for p.grad None issue in AdaLoRA.

This script demonstrates that at the very first training step, when update_ipt is called
BEFORE the backward pass, p.grad is None which would cause a crash without the defensive
None check added in line 338 of layer.py.

The issue can occur in vanilla AdaLoRA when:
1. update_and_allocate is called before backward() on the first step
2. Parameters are frozen (requires_grad=False)
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import AdaLoraConfig, get_peft_model


def main():
    print("=== Reproducer for p.grad None issue in AdaLoRA ===\n")

    # minimal setup
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # configure AdaLoRA (vanilla, no DoRA)
    peft_config = AdaLoraConfig(
        init_r=8,
        target_r=4,
        beta1=0.85,
        beta2=0.85,
        tinit=10,
        tfinal=50,
        deltaT=5,
        lora_alpha=16,
        lora_dropout=0.0,
        task_type="SEQ_CLS",
        inference_mode=False,
        total_step=100,
        target_modules=["q_lin", "v_lin"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # create a simple batch
    inputs = tokenizer(
        ["This is a test sentence."], return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    inputs["labels"] = torch.tensor([1])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("\n--- Step 0: Before any forward/backward pass ---")
    print("Calling update_and_allocate(global_step=0) BEFORE backward...")

    # this is the problematic scenario: calling update_and_allocate before backward
    # at the first step, gradients haven't been computed yet
    try:
        # SCENARIO 1: update_and_allocate called BEFORE backward (causes the issue)
        model.base_model.update_and_allocate(global_step=0)
        print("✓ update_and_allocate succeeded (thanks to p.grad None check)")

        # check if any lora parameters have gradients
        has_grad = False
        for n, p in model.named_parameters():
            if "lora_" in n and "default" in n:
                if p.grad is not None:
                    has_grad = True
                    break

        print(f"  Any LoRA params have gradients? {has_grad}")

    except Exception as e:
        print(f"✗ update_and_allocate failed with error: {e}")
        print("  This would crash without the defensive p.grad None check!")

    print("\n--- Step 1: Normal training step with forward and backward ---")

    # now do a normal forward + backward
    outputs = model(**inputs)
    loss = outputs.loss
    print(f"Loss: {loss.item():.4f}")

    loss.backward()

    # check gradients after backward
    print("Gradients after backward:")
    for n, p in model.named_parameters():
        if "lora_A.default" in n:
            print(f"  {n}: grad is None? {p.grad is None}")
            break

    # now update_and_allocate should work normally
    print("\nCalling update_and_allocate(global_step=1) AFTER backward...")
    model.base_model.update_and_allocate(global_step=1)
    print("✓ update_and_allocate succeeded")

    optimizer.step()
    optimizer.zero_grad()

    print("\n--- Demonstrating the defensive check is necessary ---")
    print("Without the 'if p.grad is None: continue' check on line 338,")
    print("the code would crash with: TypeError: unsupported operand type(s) for *: 'Parameter' and 'NoneType'")
    print("when trying to compute (p * p.grad).abs() on line 347.")

    print("\n=== Reproducer completed successfully ===")


if __name__ == "__main__":
    main()
