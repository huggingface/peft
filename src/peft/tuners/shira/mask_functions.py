import torch
import torch.nn as nn

"""
This module is intended to store mask functions for use inside SHiRA construction.
The mask functions are required to have a specific signature as shown in the type-hint in ShiraConfig.

Required positional arguments:
    base_layer - This is the linear layer where the shira adapter will be attached.
    r          - Adapter rank.  Shira doesn't use the rank concept exactly, but this is used to determine the
                 number of parameters in the shira adapter in a way that is consistent with LoRA sizing.
Keyword arguments can be provided as needed by the particular mask function implementation.  

Return:
    mask - this is a torch.tensor of the same shape as base_layer.weight that contains 0s and 1s with the same
           dtype and device as base_layer.weight

If you would like to attach Shira adapters to a model using PEFT methods (such as get_peft_model()), using more arguments
than the provided positional arguments, you can create the mask function reference like the following:

    def create_mask_function_reference(**my_kwargs):
        def mask_fn(base_layer, r):
            ... your implementation here that might use my_kwargs ...
            return mask
        return mask_fn
"""

def random_mask(base_layer: nn.Module, r: int, random_seed: int = 42, **kwargs) -> torch.tensor:
    _ = kwargs
    shape = base_layer.weight.shape
    num_shira_weights = r * (shape[0] + shape[1])
    random_generator = torch.Generator()
    random_generator.manual_seed(random_seed)

    idx = (torch.randperm(base_layer.weight.numel(), generator=random_generator)[:num_shira_weights]).to(base_layer.weight.device)
    val = torch.ones_like(idx.type(base_layer.weight.dtype))
    mask = torch.zeros_like(base_layer.weight.view(1,-1))
    mask = mask.scatter_(1, idx.unsqueeze(0), val.unsqueeze(0)).view(shape)

    return mask

