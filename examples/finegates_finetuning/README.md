# FineGates finetuning

This example shows how to finetune a sequence-classification model with FineGates and the auxiliary sparsity loss.

## Quick start

```bash
python examples/finegates_finetuning/finegates_glue.py \
  --model_name_or_path bert-base-uncased \
  --task_name mrpc \
  --output_dir ./finegates-mrpc
```

The script uses a custom `Trainer` class that mixes in `FineGatesTrainerMixin`, so the task loss and the FineGates
sparsity loss are optimized together.

Useful knobs:

* `--target_sparsity`: target structured sparsity level.
* `--sparsity_loss_weight`: weight of the auxiliary sparsity loss.
* `--target_modules`: comma-separated module suffixes to adapt.

After training, the adapter can be merged into the base model:

```python
merged_model = trainer.model.merge_and_unload()
```
