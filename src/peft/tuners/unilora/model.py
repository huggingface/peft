from __future__ import annotations
import warnings
import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING 
from .config import UniLoraConfig
from .layer import Linear, UniLoraLayer

class UniLoraModel(BaseTuner):
    """
    Creates UniLora model from a pretrained transformers model.
    """
    prefix: str = "unilora_"
    tuner_layer_cls = UniLoraLayer 
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False,state_dict=None) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage,state_dict=state_dict)
        
        # --- UniLora-specific initialization logic (global hash index assignment) ---
        # 1. Count the total number of required indices (using the new `indices` variable)
        lora_param_count = 0
        for name, module in model.named_modules():
             if isinstance(module, UniLoraLayer):
               lora_param_count += module.unilora_indices_A[adapter_name].numel()
               lora_param_count += module.unilora_indices_B[adapter_name].numel()
        

        # 2. Generate globally uniform-distributed indices
        adapter_cfg = self.peft_config[adapter_name]
        theta_d_length = adapter_cfg.theta_d_length
        proj_seed = adapter_cfg.proj_seed
        indices = self.generate_index(lora_param_count,theta_d_length,proj_seed)
        pointer = 0
        
        # 3. Assign the generated indices back to each layer (replacing the randomly initialized indices)
        for name, module in model.named_modules():
            if isinstance(module, UniLoraLayer):
             
                param_numel = module.unilora_indices_A[adapter_name].numel()
                chunk = indices[pointer: pointer + param_numel]
                module.unilora_indices_A[adapter_name] = chunk.view_as(module.unilora_indices_A[adapter_name]).clone()
                pointer += param_numel
                
           
                param_numel = module.unilora_indices_B[adapter_name].numel()
                chunk = indices[pointer: pointer + param_numel]
                module.unilora_indices_B[adapter_name] = chunk.view_as(module.unilora_indices_B[adapter_name]).clone()
                pointer += param_numel
        
        assert pointer == len(indices)
        
        # 4. Compute the usage frequency of each index for normalization (Scales)
        counts = torch.bincount(indices, minlength=theta_d_length) 
        sqrt_counts = 1/torch.sqrt(counts.float()) 
        
        index_ls = []
        for name, module in model.named_modules():
             if isinstance(module, UniLoraLayer):
               index_ls.append(module.unilora_indices_A[adapter_name].long())
               index_ls.append(module.unilora_indices_B[adapter_name].long())
        
        norm_factors = [sqrt_counts[t] for t in index_ls]
        
        # 5. Update the Scales of each layer (formerly Norm)
        uni_modules = [m for m in self.modules() if isinstance(m, UniLoraLayer)]
       
        for module, (scale_a, scale_b) in zip(uni_modules, zip(*[iter(norm_factors)] * 2)):
            module.update_scaling(adapter_name, scale_a, scale_b)

 
    # Assigns a deterministic index in `[0, theta_d_length - 1]` to each LoRA parameter.
    # This function maps the flattened LoRA parameter space of size `lora_param_count` 
    # (i.e., D) into `theta_d_length` buckets (i.e., d), where `d << D`. The resulting 
    # index tensor determines which entry of the global UniLoRA vector `theta_d` 
    # will be used to reconstruct each LoRA parameter.
    def generate_index(self, lora_param_count, theta_d_length,proj_seed):
        import numpy as np
        total_length = lora_param_count
        num_unique = theta_d_length
        base_count = total_length // num_unique
        remaining = total_length % num_unique
        rng = np.random.default_rng(proj_seed)
        data = np.repeat(np.arange(num_unique), base_count)
        if remaining > 0:
            extras = rng.choice(num_unique, size=remaining, replace=False)
            data = np.concatenate([data, extras])
        rng.shuffle(data)
        return torch.tensor(data)

    def _init_unilora_theta_d(self, config: UniLoraConfig, adapter_name: str) -> None:
        unilora_theta_d = torch.zeros(config.theta_d_length)
        torch.nn.init.uniform_(unilora_theta_d, -config.init_theta_d_bound, config.init_theta_d_bound)
        self.unilora_theta_d[adapter_name] = unilora_theta_d

    def _pre_injection_hook(self, model: nn.Module, config: UniLoraConfig, adapter_name: str) -> None:
        self.unilora_theta_d = nn.ParameterDict({})
        

    def _create_and_replace(
        self,
        unilora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "fan_in_fan_out": unilora_config.fan_in_fan_out,
            "bias": bias,
        }
        self._init_unilora_theta_d(unilora_config, adapter_name)
        
        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                unilora_theta_d=self.unilora_theta_d,
                r=unilora_config.r,
                theta_d_length=unilora_config.theta_d_length,
                unilora_dropout=unilora_config.unilora_dropout,
            )
        else:
            new_module = self._create_new_module(
                unilora_config=unilora_config,
                unilora_theta_d=self.unilora_theta_d,
                adapter_name=adapter_name,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, child):
       
        setattr(parent, child_name, new_module)
        if hasattr(child, "base_layer"):
            child = child.base_layer

        meta = torch.device("meta")
        # dispatch to correct device
        for name, module in new_module.named_modules():
            if (self.prefix in name) or ("ranknum" in name):
                if hasattr(child, "qweight"):
                    weight = child.qweight
                elif hasattr(child, "W_q"):
                    weight = child.W_q
                elif hasattr(child, "weight"):
                    weight = child.weight
                elif getattr(child, "in_proj_weight", None) is not None:  # MHA
                    weight = child.in_proj_weight
                else:
                    weight = next(child.parameters())
                if not any(p.device == meta for p in module.parameters()):
                    module.to(weight.device)

    @staticmethod
    def _create_new_module(unilora_config, unilora_theta_d, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target
        
        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = unilora_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = unilora_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `transformers.pytorch_utils.Conv1D`."
            )
        new_module = Linear(
            base_layer=target,
            unilora_theta_d=unilora_theta_d,
            adapter_name=adapter_name,
            r=unilora_config.r,
            theta_d_length=unilora_config.theta_d_length,
            unilora_dropout=unilora_config.unilora_dropout,
            **kwargs,
        )
        return new_module


    def print_savable_parameters(self) -> None:
        """
        Prints the number of savable Uni-LoRA parameters and total savable parameters.
        """
        unilora_params, other_params = self.get_nb_savable_parameters()
        print(
            f"Uni-LoRA params to-be-saved (float32-equivalent): {unilora_params:,d} "
            f"|| total params to-be-saved: {(unilora_params + other_params):,d}"
        )