from typing import Dict, Union
from dataclasses import dataclass
from loguru import logger as my_logger
import numpy as np
import torch as to
from peft.tuners.lora import LoraModel, LoraLayer
from peft.tuners.pclora.layer import PCLoRALayer
from peft.tuners.lora.config import LoraConfig
from transformers.modeling_outputs import CausalLMOutput, CausalLMOutputWithPast

@dataclass
class PCLoRACausalLLMOutput(CausalLMOutputWithPast):
    feature_distillation_loss: to.FloatTensor = None
    
class PCLoraModel(LoraModel):
    def __init__(self, model, lora_config: Union[LoraConfig, Dict]  , adapter_name: str) -> None:        
        super().__init__(model, lora_config, adapter_name)
        
        try:
            self._decay_schedule = getattr(self, f"_{lora_config[adapter_name].decay_schedule}") 
        except AttributeError:
            raise AttributeError(f"Invalid decay schedule: {lora_config[adapter_name].decay_schedule}")
        
        self._task_loss_alpha = lora_config[adapter_name].task_loss_alpha
        self._q = lora_config[adapter_name].q
        
    def _linear(self, step: int, q: int) -> float:
        return 1 - step / q if step < q else 0
    
    def _sine(self, step: int, q: int) -> float:
        return np.sin(np.pi/2 * (1 + step / q)) if step < q else 0
    
    def _identiy(self, step: int, q: int) -> float:
        return 1 if step < q else 0
    
    def _linear_cutoff(self, step: int, q: int) -> float:
        return max(1 - step / q, 0.8) 
    
    def update_lora(self, step: int, **kwargs) -> None:
        lambda_ft_distill = self._decay_schedule(step, self._q)
        
        my_logger.debug(f"Decay Schedule: \n Lambda: {lambda_ft_distill} \n Step: {step} \n Q: {self._q}")
        
        for name, module in self._get_lora_modules():
            module.update(lambda_ft_distill, **kwargs)
            
    def forward(self, *args, **kwargs):
        with to.no_grad():
            self.disable_adapter_layers()
            teacher_out: CausalLMOutput = self.model.forward(*args, **kwargs)
            
        self.enable_adapter_layers()
        student_out: CausalLMOutput = self.model.forward(*args, **kwargs)
        
        ft_dist_loss = 0
        for name, module in self._get_lora_modules():
            teacher_activations: to.Tensor = module.teacher_activations
            student_activations: to.Tensor = module.student_activations
            
            if teacher_activations.requires_grad:
                my_logger.warning(f"Teacher activations for {name} require grad. Disabling grad for teacher activations.")
                teacher_activations.requires_grad = False
                
            ft_dist_loss += to.nn.functional.mse_loss(student_activations, teacher_activations)
        task_loss = student_out.loss
        total_loss = self._task_loss_alpha * task_loss + (1- self._task_loss_alpha) * ft_dist_loss
        
        student_out.loss = total_loss
        student_out = PCLoRACausalLLMOutput(**student_out, feature_distillation_loss=ft_dist_loss.detach())
        my_logger.debug(f"Student Out Loss: {student_out.loss}")
        my_logger.debug(f"Student Out Feature Distillation Loss: {ft_dist_loss}")
        return student_out 
        
    def _get_lora_modules(self):
        for name, module in self.model.named_modules():
            if isinstance(module, LoraLayer) or isinstance(module, PCLoRALayer):
                yield name, module
                
    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        new_module = PCLoRALayer(target, adapter_name, **kwargs)
        return new_module
