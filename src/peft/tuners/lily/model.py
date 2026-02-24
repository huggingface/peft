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

from __future__ import annotations

import torch
from torch import nn

from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
)
from peft.utils import TRANSFORMERS_MODELS_TO_LILY_TARGET_MODULES_MAPPING

from .config import LilyConfig
from .layer import LilyLayer, Linear


class LilyModel(BaseTuner):
    """
    Creates a Low-Rank Interconnected Adaptation Across Layers (Lily) model from a pretrained transformers model.

    The method is described in detail in https://arxiv.org/abs/2407.09946.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`LilyConfig`]): The configuration of the Lily model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The Lily PEFT model.

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LilyConfig`]): The configuration of the Lily model.
    """

    prefix: str = "lily_"
    tuner_layer_cls = LilyLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LILY_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        lily_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        """
        Create a new Lily layer with independent A and B for each layer.
        Sharing of A/B across layers is deferred to _post_injection_hook.
        """
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        if isinstance(target, LilyLayer):
            target.update_layer(
                adapter_name,
                lily_config.r,
                scaling=lily_config.scaling,
                stride_A=lily_config.stride_A,
                num_B=lily_config.num_B,
                init_weights=lily_config.init_weights,
            )
        else:
            new_module = self._create_new_module(lily_config, adapter_name, target)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(lily_config, adapter_name, target):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            return Linear(
                target,
                adapter_name,
                r=lily_config.r,
                scaling=lily_config.scaling,
                stride_A=lily_config.stride_A,
                num_B=lily_config.num_B,
                init_weights=lily_config.init_weights,
            )

        raise NotImplementedError(
            f"Lily does not support target modules of type {type(target_base_layer)} yet."
        )

    def _post_injection_hook(self, model: nn.Module, config: LilyConfig, adapter_name: str) -> None:
        """
        After all layers have been independently initialized, apply A/B sharing across layers.

        A sharing: for each (target_module_suffix, weight_shape) group, consecutive blocks of
        `stride_A` layers share the same A. The first layer in each block keeps its own A;
        subsequent layers in the block have their lily_A replaced by the group leader's.

        B sharing: all layers in the same (target_module_suffix, weight_shape) group share
        the B from the very first layer in that group.

        Both lily_A and lily_B are nn.Linear modules, so they move correctly with
        model.to(device) via standard nn.Module parameter propagation.

        Note: (target_module_suffix, weight_shape) is used as the group key rather than
        target_module_suffix alone, to correctly handle architectures like UNet where the
        same target key can appear with different shapes across layers.
        """
        stride_A = config.stride_A

        # Collect all adapted LilyLayer modules in traversal order, grouped by
        # (target_module_suffix, weight_shape).
        # Maps (target_suffix, weight_shape) -> list of LilyLayer in traversal order.
        target_to_layers: dict[tuple[str, torch.Size], list[LilyLayer]] = {}

        for key, module in model.named_modules():
            if not isinstance(module, LilyLayer):
                continue
            if adapter_name not in module.lily_A:
                continue

            base = module.get_base_layer()
            shape = base.weight.shape  # (out_features, in_features)

            # Find the longest matching target suffix for this key
            matched_suffix = None
            if isinstance(config.target_modules, str):
                matched_suffix = config.target_modules
            else:
                for suffix in config.target_modules:
                    if key.endswith(suffix):
                        matched_suffix = suffix

            if matched_suffix is None:
                # Should not happen since inject_adapter already matched this layer
                continue

            group_key = (matched_suffix, shape)
            if group_key not in target_to_layers:
                target_to_layers[group_key] = []
            target_to_layers[group_key].append(module)

        # Apply A and B sharing for each (target_suffix, shape) group.
        for (target_suffix, shape), layers in target_to_layers.items():
            # B sharing: all layers share the first layer's B
            shared_B = layers[0].lily_B[adapter_name]
            for i, layer in enumerate(layers):
                if i != 0:
                    layer.lily_B[adapter_name] = shared_B

                # A sharing: layers within the same stride_A block share the group leader's A.
                # Group leader is the first layer in each block of stride_A consecutive layers.
                group_idx = (i // stride_A) * stride_A
                if i != group_idx:
                    layer.lily_A[adapter_name] = layers[group_idx].lily_A[adapter_name]
