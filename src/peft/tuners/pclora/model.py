from typing import Dict, Union
import numpy as np
import torch as to
from loguru import logger
from peft.tuners.lora import LoraModel, LoraLayer
from peft.tuners.pclora.layer import PCLoRALayer
from peft.tuners.lora.config import LoraConfig
from peft.tuners.lora.layer import BaseTunerLayer
from transformers.modeling_outputs import CausalLMOutput, CausalLMOutputWithPast
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class PCLoRACausalLLMOutput(CausalLMOutputWithPast):
    feature_distillation_loss: to.FloatTensor = None
    
class PCLoraModel(LoraModel):
    def __init__(self, model, lora_config: Union[LoraConfig, Dict]  , adapter_name: str) -> None:        
        super().__init__(model, lora_config, adapter_name)
        
        try:
            self._decay_schedule = getattr(self, f"_{lora_config[adapter_name].decay_schedule}") 
        except AttributeError:
            raise AttributeError(f"Invalid decay schedule: {lora_config[adapter_name].decay_schedule}")
        self._total_steps = lora_config[adapter_name].total_steps
        self._distillation_loss_lambda = lora_config[adapter_name].distillation_loss_lambda
        
    def _linear(self, step: int, total_steps: int) -> float:
        return 1 - step / total_steps
    
    def _sine(self, step: int, total_steps: int) -> float:
        return np.sin(np.pi/2 * (1 + step / total_steps))
    
    def _identiy(self, step: int, total_steps: int) -> float:
        return 1
        
    def update_lora(self, step: int, **kwargs) -> None:
        alpha = self._decay_schedule(step, self._total_steps)
        logger.info(f"Updating LoRA with alpha: {alpha}. Step: {step}. Total steps: {self._total_steps}")
        
        for name, module in self._get_lora_modules():
            module.update(alpha, **kwargs)
            
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
                logger.warning(f"Teacher activations for {name} require grad. Disabling grad for teacher activations.")
                teacher_activations.requires_grad = False
                
            ft_dist_loss += to.nn.functional.mse_loss(student_activations, teacher_activations)
        
        student_out.loss += self._distillation_loss_lambda * ft_dist_loss
        
        student_out = PCLoRACausalLLMOutput(**student_out, feature_distillation_loss=ft_dist_loss.detach())
        logger.info(f"Student Out Loss: {student_out.loss}")
        logger.info(f"Student Out Feature Distillation Loss: {ft_dist_loss}")
        return student_out 
        
    def _get_lora_modules(self):
        for name, module in self.model.named_modules():
            if isinstance(module, LoraLayer) or isinstance(module, PCLoRALayer):
                yield name, module
                
    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        logger.info(f"Creating new PCLoRALayer with config: {lora_config}")
        new_module = PCLoRALayer(target, adapter_name, **kwargs)
        return new_module
