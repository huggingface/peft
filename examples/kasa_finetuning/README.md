# KaSA: Knowledge-aware Singular-value Adaptation

## Introduction ([Paper](https://huggingface.co/papers/2412.06071))

KaSA (Knowledge-aware Singular-value Adaptation) is a parameter-efficient fine-tuning method closely related to LoRA. Like LoRA, KaSA inserts a low-rank update into a pretrained weight `W ∈ R^{out×in}`. Unlike LoRA, KaSA operates in the spectral domain of the base weight:

- Compute the SVD `W = U Σ V^T` and discard the `r` smallest singular components, leaving the rank-`(k - r)` approximation as the new frozen base weight (`k = min(in_features, out_features)`). The intuition is that the smallest singular components carry noisy or long-tail knowledge that can hinder adaptation.
- Parametrize the trainable update in SVD form: `ΔW = (α/r) · B · diag(ΔΣ) · A`, where `ΔΣ` (`lora_diag`) is a learnable `r`-vector of singular values inserted between the LoRA factors. `B` is zero-initialized as in vanilla LoRA, so the update is zero at step 0.
- Train with two auxiliary regularizers: an L2 penalty `β · ||ΔΣ||²` on the singular values and an orthogonal regularization `γ · (||B^T B - I||_F + ||A A^T - I||_F)` on the adapter factors, which softly enforces the semi-orthogonality assumed by the SVD parametrization.

## Quick Start

```python
import torch
from peft import KasaConfig, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id

lora_config = LoraConfig(
    kasa_config=KasaConfig(beta=1e-4, gamma=1e-3),
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()


class KasaSFTTrainer(SFTTrainer):
    """Adds the KaSA auxiliary regularization to the task loss."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        if return_outputs:
            loss, outputs = result
            return loss + model._get_kasa_loss(), outputs
        return result + model._get_kasa_loss()


dataset = load_dataset("imdb", split="train[:1%]")
training_args = SFTConfig(dataset_text_field="text", max_length=128)
trainer = KasaSFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
peft_model.save_pretrained("kasa-llama-2-7b")
```

To reload the trained adapter:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", dtype=torch.bfloat16, device_map="auto"
)
peft_model = PeftModel.from_pretrained(model, "kasa-llama-2-7b")
```

Loading the adapter re-applies the same SVD truncation to the freshly loaded base weight, so the reloaded model matches the one that was trained.

## Notes and limitations

- KaSA currently supports `nn.Linear` target modules only, and not `fan_in_fan_out=True` layers (e.g. transformers `Conv1D`).
- The SVD truncation of the base weight is **destructive**: adding a KaSA adapter permanently changes the layer's frozen weight. Disabling or unloading the adapter does not restore the original base weight, and `merge` followed by `unmerge` round-trips to the truncated weight, not the original one. This is inherent to the method. Keep the original checkpoint if you need the unmodified base model.
- KaSA performs a full SVD per target weight at initialization. For 7B-scale models this is a one-time cost of seconds; for substantially larger weight matrices the cost grows.
- The auxiliary regularizers are optional but recommended for faithfulness to the paper; without them the SVD interpretation of the update is only approximate. They only take effect if you add the model's `_get_kasa_loss()` to your loss as shown above.
- Combining KaSA with `use_dora=True` or other LoRA variants is not supported, and KaSA adapters cannot be mixed with non-KaSA adapters on the same model.

## Citation

```
@inproceedings{wang2025kasa,
  title={KaSA: Knowledge-Aware Singular-Value Adaptation of Large Language Models},
  author={Wang, Fan and Jiang, Juyong and Park, Chansung and Kim, Sunghun and Tang, Jing},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
