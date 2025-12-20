from __future__ import annotations

import re

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTuner
from peft.utils.constants import TRANSFORMERS_MODELS_TO_OSF_TARGET_MODULES_MAPPING

from .layer import OSFLayer, dispatch_default


class OSFModel(BaseTuner):
    """A minimal tuner implementing Orthogonal Subspace Fine-tuning."""

    prefix: str = "osf_"
    tuner_layer_cls = OSFLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_OSF_TARGET_MODULES_MAPPING

    def __init__(
        self,
        model,
        config,
        adapter_name,
        low_cpu_mem_usage: bool = False,
        state_dict: dict[str, torch.Tensor] | None = None,
    ):
        # Pass state_dict through for compatibility with BaseTuner
        super().__init__(
            model,
            config,
            adapter_name,
            low_cpu_mem_usage=low_cpu_mem_usage,
            state_dict=state_dict,
        )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped base model.

        This mirrors the behavior of other tuners (e.g., LoRA), ensuring attributes like `device` resolve to the
        underlying transformers model.
        """
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":  # avoid infinite recursion during init
                raise
            return getattr(self.model, name)

    def _prepare_adapter_config(self, peft_config, model_config):
        # If target_modules is unspecified, try mapping; else fall back to all linear layers for custom models
        if getattr(peft_config, "target_modules", None) is None:
            model_type = model_config.get("model_type")
            if model_type in self.target_module_mapping:
                peft_config.target_modules = set(self.target_module_mapping[model_type])
            else:
                from peft.utils.constants import INCLUDE_LINEAR_LAYERS_SHORTHAND

                peft_config.target_modules = INCLUDE_LINEAR_LAYERS_SHORTHAND
        return peft_config

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
        if not hasattr(target, "weight") or len(target.weight.shape) != 2:
            return None

        # Determine effective rank for this target (supports int or fractional in (0,1])
        def _resolve_rank(value, min_dim: int) -> int:
            if value is None:
                return max(min_dim // 2, 0)
            # floats in (0,1] => fraction of min_dim
            if isinstance(value, float) and 0 < value <= 1:
                r = int(min_dim * value)
            else:
                r = int(value)
            return max(min(min_dim, r), 0)

        min_dim = min(target.weight.shape)
        effective_rank = _resolve_rank(getattr(osf_config, "effective_rank", None), min_dim)

        # Check for per-module rank overrides (allow int or fractional)
        if hasattr(osf_config, "rank_pattern") and osf_config.rank_pattern:
            for pattern, rank in osf_config.rank_pattern.items():
                if re.search(pattern, current_key):
                    effective_rank = _resolve_rank(rank, min_dim)
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
            # Only OSF adapter parameters (in osf_svd_params) should be trainable
            if "osf_svd_params" not in n:
                p.requires_grad = False

    def _cast_adapter_dtype(self, adapter_name: str, autocast_adapter_dtype: bool = True) -> None:
        """
        Ensure all OSF adapter components have consistent dtype with the base model.

        Instead of forcing float32, we match the base model's actual dtype for consistency.
        """
        if not autocast_adapter_dtype:
            return

        for module in self.model.modules():
            if not hasattr(module, "osf_svd_params"):
                continue

            # Get target dtype from base layer weight
            base_layer = getattr(module, "base_layer", None)
            if base_layer is None or not hasattr(base_layer, "weight"):
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
