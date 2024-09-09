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
    # kld_loss: to.FloatTensor = None
    task_loss: to.FloatTensor = None
    # all_disitllation_losses: Dict[str, to.FloatTensor] = None

def kld_loss(teacher_logits: to.Tensor, student_logits: to.Tensor) -> to.Tensor:
    """ Compute the Kullback-Leibler divergence between two distributions => Knowledge distillation loss"""
    teacher_probs = to.nn.functional.log_softmax(teacher_logits, dim=-1)
    student_probs = to.nn.functional.log_softmax(student_logits, dim=-1)
    return to.nn.functional.kl_div(student_probs, teacher_probs, reduction="batchmean", log_target=True)

class DecaySchedule:
    def __init__(self, decay_schedule: str,
                 q: int,
                 keep_constant_for_k_steps: int = 10) -> None:
        self._decay_schedule = getattr(self, f"_{decay_schedule}")  # Get the function by name
        self._keep_constant_for_k_steps = keep_constant_for_k_steps # Keep the value constant for k steps
        self._last_value = 0 # Last value of the decay schedule used as a cache
        self._q = q # End of the decay schedule
        
    def _linear(self, step: int) -> float:
        return 1 - step / self._q if step <= self._q else 0
    
    def _cosine(self, step: int) -> float:
        return np.sin(np.pi/2 * (1 + step / self._q)) if step <= self._q else 0
    
    def _sine(self, step: int) -> float:
        return 1 - np.sin(np.pi/2 * (step / self._q)) if step <= self._q else 0
    
    def _identity(self, step: int) -> float:
        return 1 if step < self._q else 0
    
    def _linear_cutoff(self, step: int) -> float:
        return max(1 - step / self._q, 0.9) 
    
        
    def step(self, step: int) -> float:
        """ Compute the decay schedule value for the current step """
        if self._keep_constant_for_k_steps > 0 and step % self._keep_constant_for_k_steps == 0 or step == self._q:
            # Update the last value
            self._last_value = self._decay_schedule(step)
            return self._last_value
        else:
            # Return the last value, i.e. keep the value constant
            return self._last_value

class PCLoraModel(LoraModel):
    def __init__(self, model, lora_config: Union[LoraConfig, Dict], adapter_name: str) -> None:        
        super().__init__(model, lora_config, adapter_name)
        try:
            self._decay_schedule = DecaySchedule(decay_schedule=self.peft_config[adapter_name].decay_schedule,
                                                 q=self.peft_config[adapter_name].q,
                                                 keep_constant_for_k_steps=self.peft_config[adapter_name].keep_constant_for_k_steps)
        except AttributeError:
            raise AttributeError(f"Invalid decay schedule: {self.peft_config[adapter_name].decay_schedule}")
        
        self._task_loss_alpha = self.peft_config[adapter_name].task_loss_alpha
        self._q = self.peft_config[adapter_name].q
        self._set_inference_mode(self.peft_config[adapter_name].inference_mode)
                    
    def forward(self, *args, **kwargs):
        kwargs["output_hidden_states"] = False        
        out: CausalLMOutput = self.model.forward(*args, **kwargs)
        # kld_loss_v = kld_loss(teacher_out.logits, student_out.logits)
        
        ft_dist_losses = {}
        for name, module in self._get_lora_modules():
            teacher_activations: to.Tensor = module.teacher_activations
            student_activations: to.Tensor = module.student_activations
            ft_dist_losses[name] = to.nn.functional.mse_loss(student_activations, teacher_activations)    
            
        ft_dist_loss_list = list(ft_dist_losses.values())
        my_logger.info(f"ft_distil_losses: {ft_dist_losses}") 
        
        ft_dist_loss = to.mean(to.tensor(ft_dist_loss_list))    
        task_loss = out["loss"]
        
        # ft_dist_losses = {k: v.detach() for k, v in ft_dist_losses.items()}
        
        total_loss = self._task_loss_alpha * task_loss + (1- self._task_loss_alpha) * ft_dist_loss
        
        my_logger.info(f"out keys: {out.keys()}")
        
        out = PCLoRACausalLLMOutput(**out,
                                    feature_distillation_loss=ft_dist_loss.detach(),
                                    task_loss=task_loss.detach(),
                                    # all_disitllation_losses=ft_dist_losses
                                    )
        out.loss = total_loss
        return out
        
    def _get_lora_modules(self):
        for name, module in self.model.named_modules():
            if isinstance(module, LoraLayer) or isinstance(module, PCLoRALayer):
                yield name, module
                
    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        new_module = PCLoRALayer(target, adapter_name, **kwargs)
        return new_module
    
    def schedule_parameters(self, step: int, **kwargs):
        return {"q": self._q, "step": step, "lambda_ft_distill": self._decay_schedule.step(step)}
    
    def update_lora(self, step: int, **kwargs) -> None:
        lambda_ft_distill = self._decay_schedule.step(step)
        for name, module in self._get_lora_modules():    
            module.update(lambda_ft_distill, **kwargs)
            
    def _set_inference_mode(self, mode: bool):
        """ Set the inference mode for all the LoRA layers. In inference mode the base layer is inactive """
        for name, module in self._get_lora_modules():
            module._inference_mode = mode