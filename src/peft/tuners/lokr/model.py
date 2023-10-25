# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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

from typing import Dict, Type

import torch

from ..lycoris_utils import LycorisTuner
from .layer import Conv2d, Linear, LoKrLayer


class LoKrModel(LycorisTuner):
    """
    Creates Low-Rank Kronecker Product model from a pretrained model. The original method is partially described in
    https://arxiv.org/abs/2108.06098 and in https://arxiv.org/abs/2309.14859 Current implementation heavily borrows
    from
    https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/lokr.py

    Args:
        model (`torch.nn.Module`): The model to which the adapter tuner layers will be attached.
        config ([`LoKrConfig`]): The configuration of the LoKr model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The LoKr model.

    Example:
        ```py
        >>> from diffusers import StableDiffusionPipeline
        >>> from peft import LoKrModel, LoKrConfig

        >>> config_te = LoKrConfig(
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
        ...     rank_dropout=0.0,
        ...     module_dropout=0.0,
        ...     init_weights=True,
        ... )
        >>> config_unet = LoKrConfig(
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_modules=[
        ...         "proj_in",
        ...         "proj_out",
        ...         "to_k",
        ...         "to_q",
        ...         "to_v",
        ...         "to_out.0",
        ...         "ff.net.0.proj",
        ...         "ff.net.2",
        ...     ],
        ...     rank_dropout=0.0,
        ...     module_dropout=0.0,
        ...     init_weights=True,
        ...     use_effective_conv2d=True,
        ... )

        >>> model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> model.text_encoder = LoKrModel(model.text_encoder, config_te, "default")
        >>> model.unet = LoKrModel(model.unet, config_unet, "default")
        ```

    **Attributes**:
        - **model** ([`~torch.nn.Module`]) -- The model to be adapted.
        - **peft_config** ([`LoKrConfig`]): The configuration of the LoKr model.
    """

    prefix: str = "lokr_"
    layers_mapping: Dict[Type[torch.nn.Module], Type[LoKrLayer]] = {
        torch.nn.Conv2d: Conv2d,
        torch.nn.Linear: Linear,
    }
