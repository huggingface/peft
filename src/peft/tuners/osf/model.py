from __future__ import annotations

import re
import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTuner
from peft.utils.constants import TRANSFORMERS_MODELS_TO_OSF_TARGET_MODULES_MAPPING

from .layer import OSFLayer, Linear, dispatch_default


class OSFModel(BaseTuner):
    """A minimal tuner implementing Orthogonal Subspace Fine-tuning."""

    prefix: str = "osf_"
    tuner_layer_cls = OSFLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_OSF_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False):
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped base model.

        This mirrors the behavior of other tuners (e.g., LoRA), ensuring attributes
        like `device` resolve to the underlying transformers model.
        """
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":  # avoid infinite recursion during init
                raise
            return getattr(self.model, name)

    def _create_and_replace(
        self,
        osf_config,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        *,
        parameter_name: str | None = None,
    ) -> None:
        # OSF only works on 2D weight matrices
        if not hasattr(target, 'weight') or len(target.weight.shape) != 2:
            return None
            
        # Determine effective rank for this target
        effective_rank = osf_config.effective_rank
        
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

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if (
                self.prefix not in n
                and "svd_params" not in n
                and not n.endswith(("_U_low", "_S_low", "_V_low"))
            ):
                p.requires_grad = False

    def _cast_adapter_dtype(self, adapter_name: str, autocast_adapter_dtype: bool = True) -> None:
        """
        Ensure all OSF adapter components have consistent dtype with the base model.

        Instead of forcing float32, we match the base model's actual dtype for consistency.
        """
        if not autocast_adapter_dtype:
            return

        for module in self.model.modules():
            if not hasattr(module, 'osf_svd_params'):
                continue

            # Get target dtype from base layer weight
            base_layer = getattr(module, 'base_layer', None)
            if base_layer is None or not hasattr(base_layer, 'weight'):
                continue

            target_dtype = base_layer.weight.dtype

            # Cast trainable low-rank parameters to match base model dtype
            if adapter_name in module.osf_svd_params:
                svd_params = module.osf_svd_params[adapter_name]
                for param_name, param in svd_params.items():
                    if param.dtype != target_dtype:
                        param.data = param.data.to(target_dtype)

            # Cast frozen high-rank buffers to match base model dtype
            for buffer_dict_name in OSFLayer.other_param_names:
                if hasattr(module, buffer_dict_name):
                    buffer_dict = getattr(module, buffer_dict_name)
                    if adapter_name in buffer_dict:
                        buffer = buffer_dict[adapter_name]
                        if buffer.dtype != target_dtype:
                            buffer_dict[adapter_name] = buffer.to(target_dtype)

    # Use BaseTuner's merge and merge_and_unload implementations.
    # Explicitly disallow unmerging at the model level for OSF.
    def unmerge_adapter(self, *args, **kwargs):
        raise NotImplementedError("OSF models do not support unmerging")
