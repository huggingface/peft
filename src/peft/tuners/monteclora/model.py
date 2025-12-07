# Copyright 2023-present the HuggingFace Inc. team.
import torch
import torch.nn as nn
from peft.tuners.lora import LoraModel
from .layer import MonteCLoraLinear

class MonteCLoraModel(LoraModel):
    """
    Creates a Monte Carlo Low Rank Adapter (MonteCLoRA) model.
    ```python
    from transformers import AutoModelForCausalLM
    from peft import MonteCLoraConfig, TaskType, get_peft_model, LoraConfig

    device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"

    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
    peft_config = MonteCLoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                use_monteclora=True,
                monteclora_n=8,
                sample_scaler=3e-4,
                kl_loss_weight=1e-5,
                task_type="CAUSAL_LM",
            )
    model = get_peft_model(model, peft_config)
    ```
    """
    

    def _create_new_module(self, lora_config, adapter_name, target, **kwargs):
        current_key = kwargs.get("current_key", "")
        
        monteclora_targets = lora_config.monteclora_config.monteclora_targets
        target_suffix = current_key.split('.')[-1]
        
        # Determine if this specific module should use MonteCLoRA
        if monteclora_targets is None:
            # If not specified, apply to all LoRA targets
            should_apply = True
        else:
            should_apply = target_suffix in monteclora_targets

        if should_apply and lora_config.monteclora_config.use_monteclora:
            kwargs.update({
                "monteclora_config": lora_config.monteclora_config,
            })

        if isinstance(target, torch.nn.Linear):
            if kwargs.get("fan_in_fan_out", False):
                kwargs["fan_in_fan_out"] = False
            new_module = MonteCLoraLinear(target, adapter_name, **kwargs)
            return new_module

        return super()._create_new_module(lora_config, adapter_name, target, **kwargs)