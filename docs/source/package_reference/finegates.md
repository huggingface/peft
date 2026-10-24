<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# FineGates

[FineGates](https://arxiv.org/abs/2602.09169) is a structured-sparsity PEFT method that learns row and column gates on
top of frozen linear layers. Instead of adding a low-rank residual like LoRA, FineGates directly rescales the base
weights through learned gates:

$$W \rightarrow \mathrm{Diag}(\omega_r)\,W\,\mathrm{Diag}(\omega_c)$$

This makes FineGates attractive when you want a PEFT method that can be merged into the base model without introducing
new inference-time matrices. After merging, the targeted weights contain structured zeros that can be inspected or
exported for downstream compression workflows.

Compared to LoRA:

* FineGates adds only gate vectors, not adapter matrices.
* FineGates can produce structured zero rows and columns after merging.
* FineGates typically benefits from an auxiliary sparsity loss during training.
* FineGates does not currently provide mixed-adapter batch inference like RoAd.

## Usage

```py
from peft import FineGatesConfig, FineGatesTrainerMixin, TaskType, get_peft_model
from transformers import Trainer


class FineGatesTrainer(FineGatesTrainerMixin, Trainer):
    pass


peft_config = FineGatesConfig(
    task_type=TaskType.SEQ_CLS,
    target_modules=["q_proj", "v_proj", "down_proj"],
    target_sparsity=0.2,
    sparsity_loss_weight=1e-2,
)

peft_model = get_peft_model(model, peft_config)
trainer = FineGatesTrainer(model=peft_model, args=training_args, train_dataset=train_dataset)
trainer.train()
```

## Sparsity statistics

`FineGatesModel` exposes a `get_finegates_compression_stats()` helper that reports how many rows, columns, and implied
weight parameters remain active for the current adapter.

## Benchmark overview

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=FineGates"
	frameborder="0"
	width="850"
	height="1000"
></iframe>

# API

## FineGatesConfig

[[autodoc]] tuners.finegates.config.FineGatesConfig

## FineGatesModel

[[autodoc]] tuners.finegates.model.FineGatesModel
