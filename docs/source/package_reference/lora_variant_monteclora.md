# MonteCLoRA (Monte Carlo Low-Rank Adaptation)

MonteCLoRA wraps a standard LoRA adapter with a small variational module that draws Monte Carlo samples of stochastic perturbations on top of the LoRA `A` matrix during training. Concretely, it learns variational parameters (a Wishart-based covariance, a per-sample multivariate-normal noise term, and a Dirichlet weighting over the samples) and adds the resulting averaged perturbation to `lora_A` at every forward pass. A KL-divergence + entropy term is added to the training loss to keep these variational parameters anchored to a sensible prior. At inference time the sampler is disabled and MonteCLoRA behaves exactly like a regular LoRA adapter, so there is **no extra inference cost or extra parameters to merge**. For the full method see https://huggingface.co/papers/2411.04358.

You may want to consider MonteCLoRA when:

- You are fine-tuning on a small or noisy dataset and want stronger regularization than vanilla LoRA. The Monte Carlo averaging and the KL term together act as a Bayesian-style regularizer.
- You want better uncertainty calibration / robustness from your adapter without paying extra cost at inference time (the variational machinery is training-only).
- Vanilla LoRA is overfitting and lowering `r` or increasing `lora_dropout` is not enough.

You probably do *not* need MonteCLoRA when you have a large, clean dataset and vanilla LoRA already trains stably — in that regime the extra variational parameters mostly add training overhead without much benefit.

To enable MonteCLoRA, pass a `MontecloraConfig` to `LoraConfig`:

```py
from peft import LoraConfig, MontecloraConfig

monteclora_config = MontecloraConfig(
    num_samples=8,         # number of Monte Carlo samples per forward pass
    sample_scaler=1e-4,    # magnitude of the variational perturbation
    kl_loss_weight=1e-5,   # weight of the KL term added to the training loss
)
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    monteclora_config=monteclora_config,
)
```

During training you must add the variational regularization loss to the task loss. The simplest way is to call [`LoraModel._get_monteclora_loss`] on the underlying `LoraModel`:

```py
task_loss = ...  # standard loss returned by your model
monteclora_loss = model._get_monteclora_loss()  # 0.0 if MonteCLoRA is not used
total_loss = task_loss + monteclora_loss
total_loss.backward()
```

If you train with the HF `Trainer`, you can simply mix in [`peft.helpers.MontecloraTrainerMixin`] which does this for you in `compute_loss`:

```py
from transformers import Trainer
from peft.helpers import MontecloraTrainerMixin


class MontecloraTrainer(MontecloraTrainerMixin, Trainer):
    pass
```

A complete working example is available at [`examples/monteclora_finetuning`](https://github.com/huggingface/peft/tree/main/examples/monteclora_finetuning).
