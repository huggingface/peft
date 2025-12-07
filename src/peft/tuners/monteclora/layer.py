# Copyright 2023-present the HuggingFace Inc. team.
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lora import LoraLayer
from ..tuners_utils import MonteCLoRASampler

class MonteCLoraLayer(LoraLayer):
    """
    Extends LoraLayer to support Monte Carlo Low Rank Adaptation (MonteCLoRA).
    """

    def update_layer(
        self, 
        adapter_name, 
        r, 
        lora_alpha, 
        lora_dropout, 
        init_lora_weights,
        use_rslora,  
        monteclora_config=None,
        use_dora: bool = False
    ):
        super().update_layer(
            adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora
        )

        self.monteclora_A = False
        self.monteclora_B = False
        self.use_monteclora = False

        if monteclora_config is not None and monteclora_config.use_monteclora:
            self.use_monteclora = True
            
            # CRITICAL FIX 1: Register new parameter dict in adapter_layer_names 
            # so set_adapter() can toggle requires_grad
            self.adapter_layer_names = self.adapter_layer_names + ("lora_mc_sampler_A",)

            if "lora_A" in monteclora_config.monteclora_at:
                # CRITICAL FIX 2: Rename to include "lora_" prefix.
                # BaseTuner freezes ANY parameter that doesn't contain the prefix "lora_".
                if not hasattr(self, "lora_mc_sampler_A"):
                    self.lora_mc_sampler_A = nn.ModuleDict({})
                
                self.lora_mc_sampler_A[adapter_name] = MonteCLoRASampler(
                    in_features=self.in_features,
                    out_features=r, 
                    monteclora_n=monteclora_config.monteclora_n, 
                    use_entropy=monteclora_config.use_entropy,
                    dirichlet_prior=monteclora_config.dirichlet_prior, 
                    sample_scaler=monteclora_config.sample_scaler,
                    kl_loss_weight=monteclora_config.kl_loss_weight, 
                    mc_training=monteclora_config.mc_training,
                    buffer_size=monteclora_config.buffer_size
                )
                self.monteclora_A = True


class MonteCLoraLinear(nn.Module, MonteCLoraLayer):
    """
    MonteCLoRA implemented in a dense layer.
    """
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights=True,
        use_rslora: bool = False,
        use_dora: bool = False,
        monteclora_config = None,
        **kwargs,
    ) -> None:
        super().__init__()
        MonteCLoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            monteclora_config=monteclora_config
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                
                # --- MonteCLoRA Sampling Logic ---
                current_weight_A = lora_A.weight

                if self.use_monteclora and self.monteclora_A:
                    # Access via the renamed attribute
                    lora_A_vars, lora_A_wts = self.lora_mc_sampler_A[active_adapter]()
                    
                    if not isinstance(lora_A_vars, int):
                        if torch.isnan(lora_A_vars).any() or torch.isnan(lora_A_wts).any():
                            warnings.warn("MonteCLoRA sampling produced NaNs.")
                        else:
                            noise = torch.nan_to_num(
                                self.lora_mc_sampler_A[active_adapter].sample_scaler * lora_A_vars, 
                                nan=0.0
                            )
                            base_w = lora_A.weight.T 
                            perturbed_w = base_w + noise 
                            averaged_w = torch.einsum('n,nij->ij', lora_A_wts, perturbed_w)
                            current_weight_A = averaged_w.T

                # --- End Sampling Logic ---

                x = x.to(lora_A.weight.dtype)
                x_dropped = dropout(x)
                out_A = F.linear(x_dropped, current_weight_A)
                
                if not self.use_dora[active_adapter]:
                    result = result + lora_B(out_A) * scaling
                else:
                    pass

            result = result.to(torch_result_dtype)

        return result