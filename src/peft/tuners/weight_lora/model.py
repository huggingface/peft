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
from itertools import chain
from typing import Dict, Type, Union

import torch
from torch import nn

from peft.tuners.lycoris_utils import LycorisConfig, LycorisTuner

from .layer import Linear, WeightLoraLayer


class WeightLoraModel(LycorisTuner):
    """
    Creates Weighted Low-Rank model from a pretrained model. The original method could be roughtly described as `W_new = W_old + w_i * AB`, where w_i is a weight (scalar) of the layer `i`.
    Current implementation heavily borrows
    from
    https://github.com/huggingface/peft/tree/main/src/peft/tuners/lokr

    Args:
        model (`torch.nn.Module`): The model to which the adapter tuner layers will be attached.
        config ([`WeightLoRAConfig`]): The configuration of the WeightLoRA model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The WeightLoRA model.

    Example:
        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import WeightLoraConfig, get_peft_model

        >>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        >>> config = WeightLoraConfig(
        ...     task_type="SEQ_CLS",
        ...     r=4,
        ...     target_modules=["fc1", "fc2", "k_proj", "out_proj", "q_proj", "v_proj"],
        ...     lora_alpha=32,
        ...     lora_dropout=0.05,
        ... )
        >>> model = get_peft_model(base_model, config)
        ```

    **Attributes**:
        - **model** ([`~torch.nn.Module`]) -- The model to be adapted.
        - **peft_config** ([`WeightLoraConfig`]): The configuration of the WeightLoRA model.
    """

    prefix: str = "weight_lora_"
    layers_mapping: Dict[Type[torch.nn.Module], Type[WeightLoraLayer]] = {
        torch.nn.Linear: Linear,
    }

    def _create_and_replace(
        self,
        config: LycorisConfig,
        adapter_name: str,
        target: Union[WeightLoraLayer, nn.Module],
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        """
        A private method to create and replace the target module with the adapter module.
        """

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(config.rank_pattern.keys(), config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf"(.*\.)?{key}$", current_key), pattern_keys), target_name)

        kwargs = config.to_dict()
        kwargs["r"] = config.rank_pattern.get(target_name_key, config.r)
        kwargs["lora_alpha"] = config.alpha_pattern.get(target_name_key, config.lora_alpha)

        if isinstance(target, WeightLoraLayer):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = self._create_new_module(config, adapter_name, target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)