<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Helper methods

A collection of helper functions for PEFT.

## Training with additional PEFT losses and updates

[`helpers.PeftTrainerMixin`] integrates [`PeftModel.get_peft_loss`] and [`PeftModel.on_peft_optimizer_step`] with Transformers `Trainer` and compatible subclasses. It adds auxiliary losses during training and updates adapter state after successful optimizer updates, before gradients are cleared. Methods that require neither operation use the same interface.

```python
from peft.helpers import PeftTrainerMixin
from transformers import Trainer

class PeftTrainer(PeftTrainerMixin, Trainer):
    pass

trainer = PeftTrainer(model=peft_model, args=training_args, train_dataset=train_dataset)
trainer.train()
```

The mixin handles gradient accumulation and skips adapter updates when AMP skips an optimizer update. It requires a Transformers version with `TrainerCallback.on_optimizer_step`; upgrade Transformers if that callback is unavailable. Do not also add method-specific auxiliary losses or register callbacks that update the same adapters.

[[autodoc]] helpers.PeftTrainerMixin

## Checking if a model is a PEFT model

[[autodoc]] helpers.check_if_peft_model
    - all

## Temporarily Rescaling Adapter Scale in LoraLayer Modules

[[autodoc]] helpers.rescale_adapter_scale
    - all

## Context manager to disable input dtype casting in the `forward` method of LoRA layers

[[autodoc]] helpers.disable_input_dtype_casting
    - all

## Context manager to enable DoRA caching (faster at inference time but requires more memory)

[[autodoc]] helpers.DoraCaching
    - all

## KappaTune target selection

`KappaTuneSelector` and `find_kappa_target_modules` implement a general target selection process from the [KappaTune paper](https://arxiv.org/abs/2506.16289). 

The method identifies modules with higher flexibility (higher output differential entropy) and lower specialization (lower sensitivity to specific input directions).

These properties make the selected modules good candidates for mitigating catastrophic forgetting in any adaptation method that adds trainable parameters, including LoRA, DoRA, LoHa, AdaLoRA, and even direct fine-tuning of the original weights.

[[autodoc]] helpers.KappaTuneSelector
    - all

[[autodoc]] helpers.find_kappa_target_modules
    - all
