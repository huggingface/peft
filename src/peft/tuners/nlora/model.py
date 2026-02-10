# model.py
import torch.nn as nn
from peft.tuners.tuners_utils import BaseTuner

from .layer import NonlinearLoraLinear

class NonlinearLoraModel(BaseTuner):
    prefix = "nlora_"                 # unique prefix for state dict filtering
    tuner_layer_cls = NonlinearLoraLinear

    def _prepare_adapter_config(self, peft_config, model_config):
        if peft_config.target_modules is None:
            raise ValueError("NonlinearLoraConfig.target_modules must be set.")
        return peft_config

    def _create_and_replace(self, config, adapter_name, target, target_name, parent, **kwargs):
        # Only wrap Linear for now (extend: Conv1D, Embedding, etc.)
        if isinstance(target, nn.Linear):
            new_module = NonlinearLoraLinear(target)
            new_module.update_layer(
                adapter_name=adapter_name,
                r=config.r,
                alpha=config.alpha,
                dropout=config.dropout,
                activation_fn=config.activation_fn,
            )
            setattr(parent, target_name, new_module)

    def __getattr__(self, name: str):
        # forward missing attrs (generate(), config, etc.)
        return getattr(self.model, name)
