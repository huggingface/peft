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
    task_loss: to.FloatTensor = None
    
class PCLoraModel(LoraModel):
    def __init__(self, model, lora_config: Union[LoraConfig, Dict]  , adapter_name: str) -> None:        
        super().__init__(model, lora_config, adapter_name)
        try:
            self._decay_schedule = getattr(self, f"_{self.peft_config[adapter_name].decay_schedule}") 
        except AttributeError:
            raise AttributeError(f"Invalid decay schedule: {self.peft_config[adapter_name].decay_schedule}")
        
        self._task_loss_alpha = self.peft_config[adapter_name].task_loss_alpha
        self._q = self.peft_config[adapter_name].q
        self._k = self.peft_config[adapter_name].k
        
    def _linear(self, step: int, q: int) -> float:
        return 1 - step / q if step < q else 0
    
    def _cosine(self, step: int, q: int) -> float:
        return np.sin(np.pi/2 * (1 + step / q)) if step < q else 0
    
    def _sine(self, step: int, q: int) -> float:
        return 1 - np.sin(np.pi/2 * (step / q)) if step < q else 0
    
    def _identity(self, step: int, q: int) -> float:
        return 1 if step < q else 0
    
    def _linear_cutoff(self, step: int, q: int) -> float:
        return max(1 - step / q, 0.9) 
    
    def update_lora(self, step: int, **kwargs) -> None:
        lambda_ft_distill = self._decay_schedule(step, self._q)
        for name, module in self._get_lora_modules():
            
            if step % self._k == 0 or step == self._q:
                module.update(lambda_ft_distill, **kwargs)
            
    def forward(self, *args, **kwargs):
        kwargs["output_hidden_states"] = False
        
        with to.no_grad():
            self.disable_adapter_layers()
            teacher_out: CausalLMOutput = self.model.forward(*args, **kwargs)
            
        self.enable_adapter_layers()
        student_out: CausalLMOutput = self.model.forward(*args, **kwargs)
        
        if not self.training:
            student_out: CausalLMOutput = self.model.forward(*args, **kwargs)
            return student_out
        else:

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
            student_out = PCLoRACausalLLMOutput(**student_out,
                                                feature_distillation_loss=ft_dist_loss.detach(),
                                                task_loss=task_loss.detach()
                                                )
            student_out.loss = total_loss
            return student_out 
        
    def _get_lora_modules(self):
        for name, module in self.model.named_modules():
            if isinstance(module, LoraLayer) or isinstance(module, PCLoRALayer):
                yield name, module
                
    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        new_module = PCLoRALayer(target, adapter_name, **kwargs)
        return new_module
    
    def schedule_parameters(self, step: int, **kwargs):
        return {"q": self._q, "step": step, "lambda_ft_distill": self._decay_schedule(step, self._q)}