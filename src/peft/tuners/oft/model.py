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

import re
from typing import Dict, Type, Union

import torch
from torch import nn

from peft.tuners.lycoris_utils import LycorisConfig, LycorisTuner

from .layer import Conv2d, Linear, OFTLayer


class OFTModel(LycorisTuner):
    """
    Creates Orthogonal Finetuning model from a pretrained model. The method is described in
    https://arxiv.org/abs/2306.07280

    Args:
        model (`torch.nn.Module`): The model to which the adapter tuner layers will be attached.
        config ([`OFTConfig`]): The configuration of the OFT model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The OFT model.

    Example:
        ```py
        >>> from diffusers import StableDiffusionPipeline
        >>> from peft import OFTModel, OFTConfig

        >>> config_te = OFTConfig(
        ...     r=8,
        ...     target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
        ...     module_dropout=0.0,
        ...     init_weights=True,
        ... )
        >>> config_unet = OFTConfig(
        ...     r=8,
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
        ...     module_dropout=0.0,
        ...     init_weights=True,
        ... )

        >>> model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> model.text_encoder = OFTModel(model.text_encoder, config_te, "default")
        >>> model.unet = OFTModel(model.unet, config_unet, "default")
        ```

    **Attributes**:
        - **model** ([`~torch.nn.Module`]) -- The model to be adapted.
        - **peft_config** ([`OFTConfig`]): The configuration of the OFT model.
    """

    prefix: str = "oft_"
    layers_mapping: Dict[Type[torch.nn.Module], Type[OFTLayer]] = {
        torch.nn.Conv2d: Conv2d,
        torch.nn.Linear: Linear,
    }

    def _create_and_replace(
        self,
        config: LycorisConfig,
        adapter_name: str,
        target: Union[OFTLayer, nn.Module],
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        """
        A private method to create and replace the target module with the adapter module.
        """

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(config.rank_pattern.keys())
        target_name_key = next(filter(lambda key: re.match(rf"(.*\.)?{key}$", current_key), pattern_keys), target_name)

        kwargs = config.to_dict()
        kwargs["r"] = config.rank_pattern.get(target_name_key, config.r)

        if isinstance(target, OFTLayer):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = self._create_new_module(config, adapter_name, target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)
