from typing import Any, Union, Dict
from dataclasses import dataclass
from math import isclose
import numpy as np
import torch as to
from loguru import logger
from peft.tuners.lora import LoraModel, LoraLayer, Linear
from peft.tuners.lora.config import LoraConfig
from torch.nn.modules import Module
from transformers.modeling_outputs import CausalLMOutput, CausalLMOutputWithPast


class PCLoRALayer(Linear):
    def __init__(self, base_layer: Module, adapter_name: str,  **kwargs) -> None:
        super().__init__(base_layer, adapter_name, **kwargs)
        self._teacher_activations = None
        self._student_activations = None
        self._alpha = 1.0
        
    def forward(self, x: to.Tensor, *args: Any, **kwargs: Any) -> to.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        
        _deactivate_base_layer = isclose(self._alpha, 0)
        
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
            # SET TEACHER ACTIVATIONS
            self._teacher_activations = result
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            # DEACTIVATE BASE LAYER IF ALPHA IS 0
            if not _deactivate_base_layer:
                result = self.base_layer(x, *args, **kwargs)
            else:
                result = to.zeros_like(x)
                
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    # THIS IS FOR PCLoRA: SCALING WITH ALPHA SCHEDULE
                    result = self._alpha * result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    # THIS IS FOR DORA, NOT RELEVANT FOR PCLoRA
                    x = dropout(x)
                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                    )

            result = result.to(torch_result_dtype)
            # SET STUDENT ACTIVATIONS
            self._student_activations = result
        return result
    
    def update(self, alpha: float, **kwargs):
        self._alpha = alpha
            
    @property
    def teacher_activations(self) -> to.Tensor:
        return self._teacher_activations

    @property
    def student_activations(self) -> to.Tensor:
        return self._student_activations
    