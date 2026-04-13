#!/usr/bin/env python
"""Basic local smoke test for a PEFT development checkout."""

import torch

import peft
from peft import LoraConfig, TaskType, get_peft_model


def main() -> None:
    base_model = torch.nn.Sequential(
        torch.nn.Linear(8, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 2),
    )

    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["0", "2"],
        r=4,
        lora_alpha=8,
    )

    model = get_peft_model(base_model, config)
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())

    print(f"peft version: {peft.__version__}")
    print(f"loaded from: {peft.__file__}")
    print(f"model class: {model.__class__.__name__}")
    print(f"trainable params: {trainable}")
    print(f"total params: {total}")
    model.print_trainable_parameters()


if __name__ == "__main__":
    main()
