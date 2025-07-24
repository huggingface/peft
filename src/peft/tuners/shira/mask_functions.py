# Copyright 2025-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module is intended to store mask functions for use inside SHiRA construction. The mask functions are required to
have a specific signature as shown below.

Required positional arguments:
    base_layer - This is the linear layer where the shira adapter will be attached. r - This parameter is used to
    determine the number of parameters in the
                 shira adapter in a way that is consistent with LoRA sizing. SHiRA is a high rank adapter. Setting this
                 parameter does not restrict the adapter rank.
Keyword arguments can be provided as needed by the particular mask function implementation.

Return:
    mask - this is a torch.tensor of the same shape as base_layer.weight that contains 0s and 1s with the same
           dtype and device as base_layer.weight

If you would like to attach SHiRA adapters to a model using PEFT methods (such as get_peft_model()), using more
arguments than the provided positional arguments, you can create the mask function reference like the following:

```
    def create_mask_function_reference(**my_kwargs):
        def mask_fn(base_layer, r):
            ... your implementation here that might use my_kwargs ...
            return mask
        return mask_fn
```
Then, you can create your peft model with custom SHiRA mask as follows:
```
    model = ...
    my_kwargs = ...
    mask_fn = create_mask_function_reference(**my_kwargs)
    peft_config = ShiraConfig(r=4, mask_type='my_custom_mask')
    peft_config.mask_fn = mask_fn
    peft_model = get_peft_model(model, peft_config)
```

Complete training examples are provided in the examples/shira/ directory.
"""

from typing import Optional

import torch
import torch.nn as nn


def random_mask(base_layer: nn.Module, r: int, random_seed: Optional[int] = None, **kwargs) -> torch.tensor:
    shape = base_layer.weight.shape
    num_shira_weights = r * (shape[0] + shape[1])
    random_generator = torch.Generator()
    if random_seed is not None:
        random_generator.manual_seed(random_seed)
    idx = (torch.randperm(base_layer.weight.numel(), generator=random_generator)[:num_shira_weights]).to(
        base_layer.weight.device
    )
    val = torch.ones_like(idx.type(base_layer.weight.dtype))
    mask = torch.zeros_like(base_layer.weight.view(1, -1))
    mask = mask.scatter_(1, idx.unsqueeze(0), val.unsqueeze(0)).view(shape)

    return mask
