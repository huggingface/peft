# Copyright 2023-present the HuggingFace Inc. team.
import re
from itertools import chain

import torch

from peft.tuners.lora import LoraLayer, LoraModel
from peft.utils import get_quantization_config

from .layer import MonteCLoraLinear


class MonteCLoraModel(LoraModel):
    """
    Creates a Monte Carlo Low Rank Adapter (MonteCLoRA) model. Inherits from LoraModel but overrides creation logic to
    handle MonteCLoRA targeting.
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

    ### Important: MonteCLoraModel requires additional loss functions. Use the following code in Huggingface Trainer class.

    from transformers import Trainer


    class MonteCLoRATrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # 1. Compute the standard task loss (e.g., CrossEntropy)
            # We call the parent class's compute_loss to handle the forward pass and label smoothing
            if return_outputs:
                task_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            else:
                task_loss = super().compute_loss(model, inputs, return_outputs=False)
                outputs = None  # Placeholder if return_outputs is False
            # 2. Calculate Variational Loss (KLD + Entropy)
            # We iterate through modules to find MonteCLoRA layers
            var_loss_sum = 0.0
            num_monte_layers = 0
            # We assume the model might be wrapped (DDP, FSDP, etc.), so we look at named_modules
            for name, module in model.named_modules():
                # Using string checks prevents import errors if MonteCLoRA isn't imported globally
                if module.__class__.__name__ in ["MonteCLoRASampler", "MonteCLoRALinear"]:
                    if hasattr(module, "get_variational_loss"):
                        # Retrieve the losses
                        l1, l2 = module.get_variational_loss()

                        # Ensure they are on the correct device and connected to the graph
                        var_loss_sum += l1 + l2
                        num_monte_layers += 1
            # 3. Normalize the Variational Loss
            # Logic from your file: Average the var loss over the number of active MonteCLoRA layers
            regularization_loss = 0.0
            if num_monte_layers > 0:
                regularization_loss = var_loss_sum / num_monte_layers
            # 4. Combine losses
            total_loss = task_loss + regularization_loss

            return (total_loss, outputs) if return_outputs else total_loss


    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        # ... other args
    )

    # Instantiate the CUSTOM trainer
    trainer = MonteCLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # Note: 'tokenizer' arg is deprecated in HF v5+, use processing_class
    )

    trainer.train()
    ```
    """

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        """
        Overridden from LoraModel to explicitly pass `current_key` to `_create_new_module`. Standard LoraModel logic
        calculates rank/alpha patterns but does not forward the key.
        """
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # 1. Pattern matching logic (Copied from LoraModel)
        pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)

        # 2. Prepare kwargs, INCLUDING current_key
        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "ephemeral_gpu_offload": lora_config.runtime_config.ephemeral_gpu_offload,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
            # CRITICAL FIX: Pass current_key so _create_new_module can decide on MonteCLoRA application
            "current_key": current_key,
        }

        # 3. Quantization handling
        quant_methods = ["gptq", "aqlm", "awq"]
        for quant_method in quant_methods:
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                kwargs[f"{quant_method}_quantization_config"] = quantization_config

        # 4. Handle existing LoraLayers (updates) vs New Modules (creation)
        if isinstance(target, LoraLayer):
            # If target is already a LoraLayer (e.g. MonteCLoraLinear inherits LoraLayer), update it
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
                # If the existing layer is MonteCLoraLinear, it will accept this:
                monteclora_config=lora_config.monteclora_config,
            )
        else:
            # Create new module
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # Adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _create_new_module(self, lora_config, adapter_name, target, **kwargs):
        """
        Interprets `current_key` to check against `monteclora_targets` and instantiates MonteCLoraLinear.
        """
        current_key = kwargs.get("current_key", "")

        # Access config via the property backward-compatibility wrapper or direct attribute
        mc_config = getattr(lora_config, "monteclora_config", lora_config)
        monteclora_targets = mc_config.monteclora_targets
        target_suffix = current_key.split(".")[-1]

        # Determine if this specific module should use MonteCLoRA
        should_apply = False
        if mc_config.use_monteclora:
            if monteclora_targets is None:
                should_apply = True
            else:
                should_apply = target_suffix in monteclora_targets

        if should_apply:
            kwargs.update(
                {
                    "monteclora_config": mc_config,
                }
            )

        if isinstance(target, torch.nn.Linear):
            if kwargs.get("fan_in_fan_out", False):
                kwargs["fan_in_fan_out"] = False

            # Dispatch to MonteCLoraLinear
            new_module = MonteCLoraLinear(target, adapter_name, **kwargs)
            return new_module

        # Fallback to standard LoRA dispatch for other layer types
        return super()._create_new_module(lora_config, adapter_name, target, **kwargs)
