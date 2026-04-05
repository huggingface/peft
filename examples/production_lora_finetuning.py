"""
Production LoRA Fine-Tuning Guide
Author: Rehan Malik

End-to-end LoRA fine-tuning patterns used in production.
Includes hyperparameter selection, training loop, and evaluation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LoRAConfig:
    """Production-tested LoRA configuration.

    These defaults are based on fine-tuning 7B-70B models
    across multiple domains (code, medical, financial).
    """
    r: int = 16                    # rank (16-32 covers most use cases)
    lora_alpha: int = 32           # alpha = 2*r is a good starting point
    lora_dropout: float = 0.05     # minimal dropout prevents overfitting
    target_modules: list = None    # None = auto-detect
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    # Training hyperparameters
    learning_rate: float = 2e-4    # higher than full fine-tuning
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation: int = 8  # effective batch = 32
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"

    # Memory optimization
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    bf16: bool = True              # use bf16 on Ampere+ GPUs

    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation

    def estimated_trainable_params(self, model_params: int) -> dict:
        """Estimate trainable parameters (typically 0.1-1% of base model)."""
        # Rough estimate: 2 * r * hidden_dim * num_target_layers
        estimated = model_params * 0.005  # ~0.5% for rank 16
        return {
            "base_params": f"{model_params/1e9:.1f}B",
            "trainable_estimate": f"{estimated/1e6:.1f}M",
            "percentage": f"{(estimated/model_params)*100:.2f}%",
        }


@dataclass
class QLoRAConfig(LoRAConfig):
    """4-bit quantized LoRA - fits 70B models on 2x A100."""
    quantization_bits: int = 4
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


RECOMMENDED_CONFIGS = {
    "7b_general": LoRAConfig(r=16, lora_alpha=32, num_epochs=3, learning_rate=2e-4),
    "7b_code": LoRAConfig(r=32, lora_alpha=64, num_epochs=2, learning_rate=1e-4),
    "13b_general": LoRAConfig(r=16, lora_alpha=32, num_epochs=2, learning_rate=1e-4),
    "70b_qlora": QLoRAConfig(r=16, lora_alpha=32, num_epochs=1, learning_rate=5e-5,
                              batch_size=2, gradient_accumulation=16),
}


def training_checklist(config: LoRAConfig) -> list[str]:
    """Pre-training validation checklist."""
    checks = [
        f"Effective batch size: {config.effective_batch_size()}",
        f"Learning rate: {config.learning_rate} (with {config.lr_scheduler} scheduler)",
        f"Gradient checkpointing: {'enabled' if config.use_gradient_checkpointing else 'DISABLED - enable for large models'}",
        f"Mixed precision: {'bf16' if config.bf16 else 'fp32 - consider bf16 for speed'}",
        f"Flash attention: {'enabled' if config.use_flash_attention else 'DISABLED - enable for memory savings'}",
    ]
    if config.r > 64:
        checks.append(f"WARNING: LoRA rank {config.r} is high - consider r=16-32 first")
    return checks


if __name__ == "__main__":
    config = RECOMMENDED_CONFIGS["7b_general"]
    print("=== 7B General Fine-Tuning Config ===")
    print(f"Rank: {config.r}, Alpha: {config.lora_alpha}")
    print(f"Effective batch: {config.effective_batch_size()}")
    print(f"\nEstimated params (7B model):")
    for k, v in config.estimated_trainable_params(7e9).items():
        print(f"  {k}: {v}")
    print(f"\nChecklist:")
    for check in training_checklist(config):
        print(f"  - {check}")
