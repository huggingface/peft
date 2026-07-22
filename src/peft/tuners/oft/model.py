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


import warnings

import torch

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_OFT_TARGET_MODULES_MAPPING,
    get_quantization_kwargs,
    resolve_quantization_backend,
)

from .layer import Conv2d, Embedding, Linear, OFTLayer


def _get_tuner_layer_class(target_base_layer: torch.nn.Module) -> type[OFTLayer] | None:
    layer_cls: type[OFTLayer] | None = None
    if isinstance(target_base_layer, torch.nn.Linear):
        layer_cls = Linear
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        layer_cls = Conv2d
    elif isinstance(target_base_layer, torch.nn.Embedding):
        layer_cls = Embedding
    elif (quant_backend := resolve_quantization_backend(target_base_layer)) is not None:
        layer_cls = {"linear": Linear, "conv2d": Conv2d}.get(quant_backend.layer_type)

    return layer_cls


class OFTModel(BaseTuner):
    """
    Creates Orthogonal Finetuning model from a pretrained model. The method is described in
    https://huggingface.co/papers/2306.07280

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
    tuner_layer_cls = OFTLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_OFT_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        oft_config,
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
            "r": oft_config.r,
            "fan_in_fan_out": oft_config.fan_in_fan_out,
        }
        kwargs.update(get_quantization_kwargs(self))

        # If it is not a OFTLayer, create a new module, else update it with new adapters
        if not isinstance(target, OFTLayer):
            new_module = self._create_new_module(oft_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        else:
            target.update_layer(
                adapter_name,
                r=oft_config.r,
                config=oft_config,
            )

    @staticmethod
    def _create_new_module(oft_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        layer_cls = _get_tuner_layer_class(target_base_layer)
        if layer_cls is None:
            raise TypeError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear`, `torch.nn.Conv2d`, "
                "`torch.nn.Embedding` (optionally quantized) are supported."
            )

        if (layer_cls == Linear) and kwargs.get("fan_in_fan_out", False):
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            oft_config.fan_in_fan_out = False
            kwargs["fan_in_fan_out"] = False

        if layer_cls == Embedding:
            kwargs.pop("fan_in_fan_out", None)

        new_module = layer_cls(target, adapter_name, config=oft_config, **kwargs)
        return new_module

    def _check_merge_allowed(self):
        """Verify that the configuration supports merging.

        Currently gptq quantization and replicated layers do not support merging.
        """
        super()._check_merge_allowed()
        if getattr(self.model, "quantization_method", None) == "gptq":
            raise ValueError("Cannot merge OFT layers when the model is gptq quantized")
        if self.peft_config.get("layer_replication"):
            raise ValueError("Cannot merge OFT layers when base model layers are replicated")
