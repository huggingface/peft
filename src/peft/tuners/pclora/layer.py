from typing import Any
import torch as to
from peft.tuners.lora import Linear
from torch.nn.modules import Module


class PCLoRALayer(Linear):
    def __init__(self, base_layer: Module, adapter_name: str,  **kwargs) -> None:
        super().__init__(base_layer, adapter_name, **kwargs)
        self._teacher_activations = None
        self._student_activations = None
        self._lambda = 1.0
        self._inference_mode = False    
        
    def forward(self, x: to.Tensor, *args: Any, **kwargs: Any) -> to.Tensor:
        """
        REQ:
        1. Compute teacher and student activations in on forward pass
        2. In training the base_layer is always active
            2.1 Base layer is used for storing teacher activations. Base layer should be detached from the graph
            2.2 Base layer is also used for computing the intermediate basis for the student activations.
                If lambda is 0, the base layer is not used for computing the intermediate basis.
        3. In inference the base_layer is always inactive

        Args:
            x (to.Tensor): _description_

        Returns:
            to.Tensor: _description_
        """
        self._check_forward_args(x, *args, **kwargs) 
        
        # In inference the base_layer is always inactive
        base_result = self.base_layer(x, *args, **kwargs) if self._inference_mode is False else 0.0
        
        if base_result.requires_grad:
            # Never propagate gradients through the base layer
            base_result = base_result.detach()
            
        torch_result_dtype = base_result.dtype
        self.teacher_activations = base_result
        
        #torch_result_dtype = result.dtype
        for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    # THIS IS FOR PCLoRA: SCALING WITH LAMBDA SCHEDULE
                    result = self._lambda * base_result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    raise NotImplementedError("DORA not implemented yet")
        result = result.to(torch_result_dtype)
        self.student_activations = result
        return result
    
    def update(self, lambda_ft_distil: float, **kwargs):
        self._lambda = lambda_ft_distil
            
    @property
    def teacher_activations(self) -> to.Tensor:
        """ Store the teacher activations during forward pass. This is used for distillation loss computation."""
        return self._teacher_activations
    
    @teacher_activations.setter
    def teacher_activations(self, value: to.Tensor) -> None:
        """ Teacher activations are set during training. During inference, this should be None."""
        # Additional safeguard to ensure that teacher activations are not used during training
        if self._inference_mode is False:
            self._teacher_activations = value.detach() if value.requires_grad else value
        else:
            self._teacher_activations = None
        
    @property
    def student_activations(self) -> to.Tensor:
        """ Store the student activations during forward pass. This is used for distillation loss computation"""
        return self._student_activations
    
    @student_activations.setter
    def student_activations(self, value: to.Tensor) -> None:
        """ Student activations are set during training. During inference, this should be None."""
        if self._inference_mode is False:
            self._student_activations = value
        else:
            self._student_activations = None
    