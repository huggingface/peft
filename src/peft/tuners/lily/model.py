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

import logging
import re
from typing import Optional

import torch
from torch import nn

from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
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

    @staticmethod
    def _check_target_module_exists(lily_config, key):
        return check_target_module_exists(lily_config, key)

    def _pre_injection_hook(self, model: nn.Module, config: LilyConfig, adapter_name: str) -> None:
        """
        Pre-compute and cache all shared A and B matrices before adapter injection begins.

        A matrices are shared across consecutive groups of layers:
          stride = total_layers_per_shape / num_A
        B matrices are shared globally across all layers (one per target-module/shape pair).

        All caches are keyed by adapter_name first, so multiple adapters can coexist
        without overwriting each other's matrices or counters.
        """
        model_config = self.get_model_config(model)
        peft_config = self._prepare_adapter_config(config, model_config)

        # Initialize top-level dicts on first call; keyed as [adapter_name][target][shape]
        if not hasattr(self, "_lily_As"):
            self._lily_As: dict[str, dict[str, dict[torch.Size, list[nn.Linear]]]] = {}
        if not hasattr(self, "_lily_Bs"):
            self._lily_Bs: dict[str, dict[str, dict[torch.Size, nn.Linear]]] = {}
        if not hasattr(self, "_lily_counters"):
            self._lily_counters: dict[str, dict[str, dict[torch.Size, int]]] = {}
        if not hasattr(self, "_lily_strides"):
            self._lily_strides: dict[str, dict[str, dict[torch.Size, int]]] = {}

        # Reset all sub-dicts for the current adapter to avoid stale state
        self._lily_As[adapter_name] = {}
        self._lily_Bs[adapter_name] = {}
        self._lily_counters[adapter_name] = {}
        self._lily_strides[adapter_name] = {}

        # Count how many layers match each (target_module, weight_shape) pair
        num_layers: dict[str, dict[torch.Size, int]] = {}
        if isinstance(peft_config.target_modules, str):
            target_modules_iter = [peft_config.target_modules]
        else:
            target_modules_iter = list(peft_config.target_modules)

        for target_key in target_modules_iter:
            num_layers[target_key] = {}

        for key, module in model.named_modules():
            matched_target = self._match_target(key, peft_config.target_modules)
            if matched_target is None:
                continue
            base = module.get_base_layer() if isinstance(module, BaseTunerLayer) else module
            if not isinstance(base, torch.nn.Linear):
                continue
            shape = base.weight.shape
            num_layers[matched_target][shape] = num_layers[matched_target].get(shape, 0) + 1

        # Compute strides and validate divisibility, stored under adapter_name
        for target, shapes in num_layers.items():
            self._lily_strides[adapter_name][target] = {}
            for shape, count in shapes.items():
                if count % config.num_A != 0:
                    raise ValueError(
                        f"Number of layers ({count}) for target '{target}' with shape {shape} "
                        f"is not divisible by num_A={config.num_A}."
                    )
                self._lily_strides[adapter_name][target][shape] = count // config.num_A

        # Pre-create shared B matrices: one nn.Linear per (target, shape), stored under adapter_name
        for target, shapes in num_layers.items():
            self._lily_Bs[adapter_name][target] = {}
            for shape in shapes:
                out_features, in_features = shape
                self._lily_Bs[adapter_name][target][shape] = nn.Linear(
                    out_features, config.num_B * config.r, bias=False
                )

        # Pre-create shared A matrices: num_A groups per (target, shape), stored under adapter_name
        for target, shapes in num_layers.items():
            self._lily_As[adapter_name][target] = {}
            for shape in shapes:
                out_features, in_features = shape
                self._lily_As[adapter_name][target][shape] = [
                    nn.Linear(in_features, config.r, bias=False) for _ in range(config.num_A)
                ]

        # Initialize counters for the current adapter; reset to 0 for every (target, shape)
        for target, shapes in num_layers.items():
            self._lily_counters[adapter_name][target] = {shape: 0 for shape in shapes}

        logging.info("=" * 50)
        logging.info("Lily adapter injecting, configuration as follows:")
        logging.info(f"  adapter_name:            {adapter_name}")
        logging.info(f"  num_A (shared A groups): {config.num_A}")
        logging.info(f"  num_B (B experts):       {config.num_B}")
        logging.info(f"  stride per A group:      {self._lily_strides[adapter_name]}")
        for target, shapes in num_layers.items():
            for shape, count in shapes.items():
                logging.info(f"  layers for '{target}' shape={shape}: {count}")
        logging.info("=" * 50)

    @staticmethod
    def _match_target(key: str, target_modules) -> Optional[str]:
        """Return the matched target-module name for *key*, or None if no match."""
        if isinstance(target_modules, str):
            return target_modules if re.fullmatch(target_modules, key) else None
        for target_key in target_modules:
            if key.endswith(target_key):
                return target_key
        return None

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
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
    
        if not hasattr(self, "_lily_As") or adapter_name not in self._lily_As:
            self._pre_injection_hook(self.model, lily_config, adapter_name)

        matched_target = self._match_target(current_key, lily_config.target_modules)
        if matched_target is None:
            raise ValueError(f"Could not match key '{current_key}' to any target module.")

        base_layer = target.get_base_layer() if isinstance(target, LilyLayer) else target

        if not isinstance(base_layer, torch.nn.Linear):
            raise ValueError(
                f"Target module '{current_key}' matched to '{matched_target}' is not a Linear layer, which is currently the only supported target for Lily. Found type {type(base_layer)} instead."
            )
        
        shape = base_layer.weight.shape

        # Look up stride, counter and matrices using adapter_name as the top-level key
        stride = self._lily_strides[adapter_name][matched_target][shape]
        counter = self._lily_counters[adapter_name][matched_target][shape]
        group_idx = counter // stride
        lily_A = self._lily_As[adapter_name][matched_target][shape][group_idx]
        lily_B = self._lily_Bs[adapter_name][matched_target][shape]

        # Advance counter so the next layer in this (adapter, target, shape) group
        # gets the correct A matrix
        self._lily_counters[adapter_name][matched_target][shape] += 1

        logging.debug(
            f"Assigning A group {group_idx} to '{current_key}' "
            f"(adapter={adapter_name}, counter={counter}, stride={stride})"
        )

        out_features, in_features = shape
        kwargs = {"in_features": in_features, "out_features": out_features}

        if isinstance(target, LilyLayer):
            target.update_layer(
                adapter_name,
                lily_config.r,
                scaling=lily_config.scaling,
                lily_A=lily_A,
                lily_B=lily_B,
                num_A=lily_config.num_A,
                num_B=lily_config.num_B,
                init_weights=lily_config.init_weights,
            )
        else:
            new_module = self._create_new_module(lily_config, adapter_name, target, lily_A, lily_B, **kwargs)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)

        # Unpack the original module wrapped by the child layer
        if hasattr(child, "base_layer"):
            child = child.base_layer

        meta = torch.device("meta")
        for name, module in new_module.named_modules():
            if self.prefix in name:
                if hasattr(child, "qweight"):
                    weight = child.qweight
                elif hasattr(child, "W_q"):
                    weight = child.W_q
                elif hasattr(child, "weight"):
                    weight = child.weight
                elif getattr(child, "in_proj_weight", None) is not None:
                    weight = child.in_proj_weight
                else:
                    weight = next(child.parameters())
                if not any(p.device == meta for p in module.parameters()):
                    module.to(weight.device)

    @staticmethod
    def _create_new_module(lily_config, adapter_name, target, lily_A, lily_B, **kwargs):
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
                num_A=lily_config.num_A,
                num_B=lily_config.num_B,
                lily_A=lily_A,
                lily_B=lily_B,
                init_weights=lily_config.init_weights,
                **kwargs,
            )

        raise NotImplementedError(
            f"Lily does not support target modules of type {type(target_base_layer)} yet."
        )