import torch.nn as nn
from peft import PeftConfig
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from .layer import CustomTokensLayer



class CustomTokensModel(BaseTuner):
    prefix: str = "custom_tokens"

    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)
        
    def _prepare_adapter_config(self, peft_config, model_config):
        return peft_config

    def _create_and_replace(
        self,
        peft_config: PeftConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        """
        A private method to create and replace the target module with the adapter module.
        """
        kwargs = peft_config.to_dict()

        if isinstance(target, CustomTokensLayer):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = self._create_new_module(peft_config, adapter_name, target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)

    def _check_target_module_exists(self, peft_config: PeftConfig, key: str) -> bool:
        return check_target_module_exists(peft_config, key)
    
    @staticmethod
    def _create_new_module(peft_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        new_module = CustomTokensLayer(target, adapter_name, **kwargs)

        return new_module
    
    def _replace_module(self, parent, child_name, new_module, child): # see https://github.com/huggingface/peft/blob/e5973883057b723b3f0fe3982bfa9d1e0c0fd8ec/src/peft/tuners/lycoris_utils.py#L300C4-L300C47
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if self.prefix in name:
                module.to(child.weight.device)
    
    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False
