from __future__ import annotations

import re
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTuner
from peft.utils.constants import TRANSFORMERS_MODELS_TO_OSF_TARGET_MODULES_MAPPING

from .layer import OSFLayer, Linear, dispatch_default
from .utils import (
    attach_gradient_hooks,
    auto_generate_target_osf_config,
    create_osf_model_class,
)


class OSFModel(BaseTuner):
    """A minimal tuner implementing Orthogonal Subspace Fine-tuning."""

    prefix: str = "osf_"

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False):
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    def _prepare_adapter_config(self, peft_config, model_config):
        return peft_config

    def inject_adapter(self, model: nn.Module, adapter_name: str, autocast_adapter_dtype: bool = True, low_cpu_mem_usage: bool = False) -> None:
        # For now, keep using the legacy approach
        # TODO: Refactor to use _create_and_replace pattern
        svd_cfg = auto_generate_target_osf_config(model)
            
        OSFCls = create_osf_model_class(model.__class__)
        base_cfg = getattr(model, "config", None)
        osf_model = OSFCls(base_cfg, svd_config=svd_cfg, initialize_svd=False)
        osf_model.load_state_dict(model.state_dict())
        osf_model.reinitialize_svd()
        attach_gradient_hooks(osf_model)
        self.model = osf_model

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

        # Create new OSF layer
        new_module = dispatch_default(target, adapter_name, osf_config, **kwargs)
        
        return new_module

    def _check_target_module_exists(self, *args, **kwargs) -> bool:
        return True

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if "svd_params" not in n and not n.endswith(("_U_low", "_S_low", "_V_low")):
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