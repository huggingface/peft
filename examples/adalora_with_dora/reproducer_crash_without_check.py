#!/usr/bin/env python
"""
Reproducer demonstrating the CRASH that occurs WITHOUT the p.grad None check.

This is a modified version of the RankAllocator.update_ipt method that REMOVES
the defensive 'if p.grad is None: continue' check to show the actual error that
would occur without it.

Run this to see the TypeError that the defensive check prevents.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import AdaLoraConfig, get_peft_model


def update_ipt_without_defensive_check(rank_allocator, model):
    """
    Modified version of RankAllocator.update_ipt WITHOUT the defensive p.grad None check.
    This will crash with TypeError when p.grad is None.
    """
    for n, p in model.named_parameters():
        if ("lora_" in n) and (rank_allocator.adapter_name in n) and ("lora_magnitude_vector" not in n):
            if n not in rank_allocator.ipt:
                rank_allocator.ipt[n] = torch.zeros_like(p)
                rank_allocator.exp_avg_ipt[n] = torch.zeros_like(p)
                rank_allocator.exp_avg_unc[n] = torch.zeros_like(p)

            # NOTE: The defensive check 'if p.grad is None: continue' is REMOVED here
            # This will cause a crash when p.grad is None

            with torch.no_grad():
                # This line will fail with TypeError when p.grad is None
                rank_allocator.ipt[n] = (p * p.grad).abs().detach()
                # sensitivity smoothing
                rank_allocator.exp_avg_ipt[n] = (
                    rank_allocator.beta1 * rank_allocator.exp_avg_ipt[n]
                    + (1 - rank_allocator.beta1) * rank_allocator.ipt[n]
                )
                # uncertainty quantification
                rank_allocator.exp_avg_unc[n] = (
                    rank_allocator.beta2 * rank_allocator.exp_avg_unc[n]
                    + (1 - rank_allocator.beta2) * (rank_allocator.ipt[n] - rank_allocator.exp_avg_ipt[n]).abs()
                )


def main():
    print("=== Demonstrating the CRASH without defensive p.grad None check ===\n")

    # minimal setup
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # configure vanilla AdaLoRA (no DoRA)
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

    # create a simple batch
    inputs = tokenizer(
        ["This is a test sentence."], return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    inputs["labels"] = torch.tensor([1])

    print("--- Calling update_ipt WITHOUT defensive None check BEFORE backward pass ---")
    print("Expected error: TypeError: unsupported operand type(s) for *: 'Parameter' and 'NoneType'\n")

    try:
        # get the rank allocator
        rank_allocator = model.base_model.rankallocator

        # call the modified update_ipt that doesn't have the defensive check
        update_ipt_without_defensive_check(rank_allocator, model)

        print("✗ UNEXPECTED: No error occurred!")

    except TypeError as e:
        print(f"✓ EXPECTED ERROR CAUGHT:")
        print(f"  {type(e).__name__}: {e}")
        print("\nThis is the exact error that the defensive 'if p.grad is None: continue' check prevents!")
        print("The error occurs because at the first step (before backward), p.grad is None,")
        print("and Python cannot multiply a Parameter by None: (p * None) fails.")

    except Exception as e:
        print(f"✗ Unexpected error type: {type(e).__name__}: {e}")

    print("\n=== Reproducer completed ===")


if __name__ == "__main__":
    main()
