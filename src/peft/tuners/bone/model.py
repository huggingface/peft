# Copyright 2024-present the HuggingFace Inc. team.
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


import torch

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_BONE_TARGET_MODULES_MAPPING

from .layer import BoneLayer, BoneLinear


class BoneModel(BaseTuner):
    """
    Creates Householder reflection adaptation (Bone) model from a pretrained model. The method is described in
    https://huggingface.co/papers/2409.15371

    Args:
        model (`torch.nn.Module`): The model to which the adapter tuner layers will be attached.
        config ([`BoneConfig`]): The configuration of the Bone model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The Bone model.

    Example:
        ```py
        >>> from diffusers import StableDiffusionPipeline
        >>> from peft import BoneModel, BoneConfig

        >>> config_te = BoneConfig(
        ...     r=8,
        ...     target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
        ...     init_weights=True,
        ... )
        >>> config_unet = BoneConfig(
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
        ...     init_weights=True,
        ... )

        >>> model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> model.text_encoder = BoneModel(model.text_encoder, config_te, "default")
        >>> model.unet = BoneModel(model.unet, config_unet, "default")
        ```

    **Attributes**:
        - **model** ([`~torch.nn.Module`]) -- The model to be adapted.
        - **peft_config** ([`BoneConfig`]): The configuration of the Bone model.
    """

    prefix: str = "bone_"
    base_layer_cls = BoneLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_BONE_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        bone_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": bone_config.r,
            "init_weights": bone_config.init_weights,
        }
        kwargs["bias"] = bias

        # If it is not a BoneLayer, create a new module, else update it with new adapters
        if not isinstance(target, BoneLayer):
            new_module = self._create_new_module(bone_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        else:
            target.update_layer(
                adapter_name,
                r=bone_config.r,
                init_weights=bone_config.init_weights,
            )

    @staticmethod
    def _create_new_module(bone_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = BoneLinear(target, adapter_name, **kwargs)
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear` is supported."
            )

        return new_module
