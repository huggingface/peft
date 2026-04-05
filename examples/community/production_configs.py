"""
Production LoRA Configuration Presets
Author: Rehan Malik

Tested configurations for common fine-tuning scenarios.
"""

from dataclasses import dataclass


@dataclass
class LoRAPreset:
    name: str
    r: int
    alpha: int
    lr: float
    epochs: int
    batch: int
    grad_accum: int
    note: str

    @property
    def effective_batch(self): return self.batch * self.grad_accum


PRESETS = {
    "7b_general": LoRAPreset("7B General", 16, 32, 2e-4, 3, 4, 8,
        "Good default for 7B models on A100/A10G"),
    "7b_code": LoRAPreset("7B Code", 32, 64, 1e-4, 2, 4, 8,
        "Higher rank for code generation tasks"),
    "13b_qlora": LoRAPreset("13B QLoRA", 16, 32, 1e-4, 2, 2, 16,
        "4-bit quantized for 13B on single 24GB GPU"),
    "70b_qlora": LoRAPreset("70B QLoRA", 16, 32, 5e-5, 1, 1, 32,
        "4-bit for 70B models on 2x A100"),
}


if __name__ == "__main__":
    for key, p in PRESETS.items():
        print(f"{p.name}: r={p.r} alpha={p.alpha} lr={p.lr} "
              f"batch={p.effective_batch} | {p.note}")
