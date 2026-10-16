# Copyright 2026-present the HuggingFace Inc. team.
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
import zlib
from typing import Optional

import torch

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_PACA_TARGET_MODULES_MAPPING,
    get_quantization_kwargs,
    resolve_quantization_backend,
)

from .layer import Linear, PacaLayer


def _get_layer_seed(random_seed: Optional[int], key: str) -> Optional[int]:
    # Every layer gets its own selection of partial connections. Derive a per-layer seed from the global seed and the
    # module name so that the selection is reproducible but not identical across layers of the same shape.
    if random_seed is None:
        return None
    return (random_seed + zlib.crc32(key.encode("utf-8"))) % (2**63)


class PacaModel(BaseTuner):
    """
    Creates a Partial Connection Adaptation (PaCA) model from a pretrained model. The method is described in
    https://huggingface.co/papers/2503.01905.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`PacaConfig`]): The configuration of the PaCA model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The PaCA model.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PacaConfig, get_peft_model

        >>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        >>> config = PacaConfig(r=16, paca_alpha=16, target_modules="all-linear")
        >>> model = get_peft_model(base_model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`PacaConfig`]): The configuration of the PaCA model.
    """

    prefix: str = "paca_"
    tuner_layer_cls = PacaLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_PACA_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        paca_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        kwargs = {
            "r": paca_config.r,
            "paca_alpha": paca_config.paca_alpha,
            "init_weights": paca_config.init_weights,
            "random_seed": _get_layer_seed(paca_config.random_seed, current_key),
        }

        if isinstance(target, PacaLayer):
            target.update_layer(adapter_name, **kwargs)
        else:
            kwargs.update(get_quantization_kwargs(self))
            new_module = self._create_new_module(paca_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(paca_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        is_supported = isinstance(target_base_layer, torch.nn.Linear)
        if not is_supported and (quant_backend := resolve_quantization_backend(target_base_layer)) is not None:
            is_supported = quant_backend.layer_type == "linear"
        if not is_supported:
            raise TypeError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`."
            )

        return Linear(target, adapter_name, **kwargs)
