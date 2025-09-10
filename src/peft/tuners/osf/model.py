from __future__ import annotations

import re
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTuner, check_target_module_exists
from peft.utils.constants import TRANSFORMERS_MODELS_TO_OSF_TARGET_MODULES_MAPPING

from .layer import OSFLayer, Linear, dispatch_default


class OSFModel(BaseTuner):
    """A minimal tuner implementing Orthogonal Subspace Fine-tuning."""

    prefix: str = "osf_"

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False):
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    def _prepare_adapter_config(self, peft_config, model_config):
        # Infer default target modules from mapping if not provided
        if getattr(peft_config, "target_modules", None) is None:
            model_type = model_config.get("model_type")
            if model_type not in TRANSFORMERS_MODELS_TO_OSF_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_OSF_TARGET_MODULES_MAPPING[model_type]
            )
        return peft_config

    def inject_adapter(
        self,
        model: nn.Module,
        adapter_name: str,
        autocast_adapter_dtype: bool = True,
        low_cpu_mem_usage: bool = False,
    ) -> None:
        # Delegate to BaseTuner to perform standard target discovery and replacement
        return super().inject_adapter(
            model,
            adapter_name,
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

    def _create_and_replace(
        self,
        osf_config,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ):
        # OSF only works on 2D weight matrices
        if not hasattr(target, 'weight') or len(target.weight.shape) != 2:
            return None
            
        # Determine effective rank for this target
        effective_rank = osf_config.effective_rank
        if effective_rank is None:
            # Default to 50% of min dimension
            effective_rank = min(target.weight.shape) // 2
        
        # Check for per-module rank overrides
        if hasattr(osf_config, 'rank_pattern') and osf_config.rank_pattern:
            for pattern, rank in osf_config.rank_pattern.items():
                if re.search(pattern, current_key):
                    effective_rank = rank
                    break

        kwargs = {
            "effective_rank": effective_rank,
        }

        # Create a new or update an existing OSF layer in place
        if isinstance(target, OSFLayer):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = dispatch_default(target, adapter_name, osf_config, **kwargs)
            if new_module is None:
                return None
            # If adding an additional adapter, keep it frozen initially
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _check_target_module_exists(osf_config, key):
        return check_target_module_exists(osf_config, key)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if (
                self.prefix not in n
                and "svd_params" not in n
                and not n.endswith(("_U_low", "_S_low", "_V_low"))
            ):
                p.requires_grad = False

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        pass

    def enable_adapter_layers(self) -> None:
        self._set_adapter_layers(True)

    def disable_adapter_layers(self) -> None:
        self._set_adapter_layers(False)

    def set_adapter(self, adapter_name):
        self.active_adapter = adapter_name

    def unload(self):
        raise NotImplementedError("OSF models cannot be unloaded yet")

    def merge_adapter(self, *args, **kwargs):
        raise NotImplementedError("OSF models do not support merging")

    def unmerge_adapter(self, *args, **kwargs):
        raise NotImplementedError("OSF models do not support merging")

    def merge_and_unload(self, *args, **kwargs):
        raise NotImplementedError("OSF models do not support merging")

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # child layer may wrap the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        # If new module is a simple wrapper, ensure weight/bias/state stay aligned
        if not hasattr(new_module, "base_layer") and hasattr(child, "weight"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)
