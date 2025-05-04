from __future__ import annotations

"""Row-based PEFT tuner

This mirrors HuggingFace's Vera implementation, but instead of using shared
projections, it uses a per-layer weight matrix where only the first row is
trainable. This provides a parameter-efficient way to adapt the model while
maintaining the same injection / (un)merging / multi-adapter plumbing as other
PEFT methods.
"""

from dataclasses import asdict
from enum import Enum
from typing import Any, Optional, Union
import warnings

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTuner, check_target_module_exists
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _get_submodules,
)
from peft.import_utils import is_bnb_available, is_bnb_4bit_available

from .config import DiagConfig
from .layer import DiagLayer, Linear

# Bits‑and‑Bytes wrappers are imported lazily so that the file can be parsed
# without having bitsandbytes installed.
if is_bnb_available():
    from .bnb import Linear8bitLt  # type: ignore
else:
    Linear8bitLt = nn.Linear  # dummy alias to satisfy isinstance checks

if is_bnb_4bit_available():
    from .bnb import Linear4bit  # type: ignore
else:
    Linear4bit = nn.Linear


class DiagModel(BaseTuner):
    """Parameter‑efficient row-based adapter where only the first row of each weight matrix is trainable.

    The public API (prepare, merge_and_unload, unload, etc.) matches
    HuggingFace PEFT tuners.
    """

    prefix: str = "diag_"  # used for freezing logic

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------
    def __init__(self, model, config: DiagConfig, adapter_name: str, *, low_cpu_mem_usage: bool = False):
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    # ------------------------------------------------------------------
    # Adapter‑config validation
    # ------------------------------------------------------------------
    def _check_new_adapter_config(self, config: DiagConfig) -> None:  # noqa: D401
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. "
                "When using multiple adapters, set bias='none' for all adapters."
            )

    # ------------------------------------------------------------------
    # Target‑layer detection helpers (mirrors Vera)
    # ------------------------------------------------------------------
    @staticmethod
    def _check_target_module_exists(diag_config: DiagConfig, key: str) -> bool:  # noqa: D401
        return check_target_module_exists(diag_config, key)

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------
    def _create_and_replace(
        self,
        diag_config: DiagConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        **optional_kwargs,
    ) -> None:
        """Wrap *one* module with our row-based `Linear` if eligible."""

        # 1. Untangle PEFT wrappers if any ---------------------------------------------------
        base = target.get_base_layer() if hasattr(target, "get_base_layer") else target

        # 2. Skip non‑Linear/Conv1D layers ---------------------------------------------------
        valid_linear_types = (nn.Linear, Linear4bit, Linear8bitLt)
        if not isinstance(base, valid_linear_types + (torch.nn.Conv1d,)):
            return

        # 3. Respect user‑supplied `target_modules` ------------------------------------------
        if target_name not in diag_config.target_modules:
            return

        # 4. Already wrapped? then *update* --------------------------------------------------
        if isinstance(target, DiagLayer):
            target.update_layer(
                adapter_name,
                diag_config.diag_alpha,
                diag_config.diag_dropout,
                diag_config.init_diag_weights,
                diag_config.bias,
            )
            return

        # 5. Fresh wrap ----------------------------------------------------------------------
        new_module = self._create_new_module(diag_config, adapter_name, target, **optional_kwargs)
        if adapter_name not in self.active_adapter:
            new_module.disable_adapters = True  # freeze if not active
        self._replace_module(parent, target_name, new_module, target)

    # Factory for wrapper -------------------------------------------------------
    @staticmethod
    def _create_new_module(
        diag_config: DiagConfig,
        adapter_name: str,
        target: nn.Module,
        **kwargs,
    ) -> DiagLayer:
        args = dict(
            diag_alpha=diag_config.diag_alpha,
            diag_dropout=diag_config.diag_dropout,
            fan_in_fan_out=diag_config.fan_in_fan_out,
            init_diag_weights=diag_config.init_diag_weights,
            bias=diag_config.bias,
            **kwargs,
        )
        base = target  # keep exact wrapper type (8‑bit etc.)
        is_conv1d = isinstance(base, torch.nn.Conv1d)
        return Linear(base, adapter_name, is_target_conv_1d_layer=is_conv1d, **args)

    # ------------------------------------------------------------------
    # Replacement helper (device‑aware, like Vera)
    # ------------------------------------------------------------------
    @staticmethod
    def _replace_module(parent: nn.Module, child_name: str, new_module: nn.Module, child: nn.Module) -> None:
        setattr(parent, child_name, new_module)
        new_module.to(child.weight.device)

    # ------------------------------------------------------------------
    # Trainable‑param masking (mirrors Vera)
    # ------------------------------------------------------------------
    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:  # noqa: D401
        for n, p in model.named_parameters():
            p.requires_grad = n.startswith(self.prefix)

        # handle bias training option ----------------------------
        for active in self.active_adapters:
            bias_mode = self.peft_config[active].bias
            if bias_mode == "none":
                continue
            if bias_mode == "all":
                for name, param in model.named_parameters():
                    if "bias" in name:
                        param.requires_grad = True
            elif bias_mode == "row_only":
                for m in model.modules():
                    if isinstance(m, DiagLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(bias_mode)

    # ------------------------------------------------------------------
    # (Un)merging helpers
    # ------------------------------------------------------------------
    def merge_adapter(self) -> None:  # noqa: D401
        for m in self.model.modules():
            if isinstance(m, DiagLayer):
                m.merge()

    def unmerge_adapter(self) -> None:  # noqa: D401
        for m in self.model.modules():
            if isinstance(m, DiagLayer):
                m.unmerge()

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------
    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for m in self.model.modules():
            if isinstance(m, DiagLayer):
                m.disable_adapters = not enabled

    def enable_adapter_layers(self):
        self._set_adapter_layers(True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(False)

    # ------------------------------------------------------------------
    # Multi‑adapter management
    # ------------------------------------------------------------------
    def set_adapter(self, adapter_name: str):
        for m in self.model.modules():
            if isinstance(m, DiagLayer):
                m.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
    def _prepare_adapter_config(self, cfg: DiagConfig, model_cfg):  # noqa: D401
        if cfg.target_modules is None:
            if model_cfg["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`.")
            cfg.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_cfg["model_type"]]
            )
        else:
            cfg.target_modules = set(cfg.target_modules)
        return cfg

    # ------------------------------------------------------------------
    # Public helpers (dict serialization identical to Vera)
    # ------------------------------------------------------------------
    def get_peft_config_as_dict(self, *, inference: bool = False):  # noqa: D401
        out = {}
        for key, value in self.peft_config.items():
            cfg = {k: (v.value if isinstance(v, Enum) else v) for k, v in asdict(value).items()}
            if inference:
                cfg["inference_mode"] = True
            out[key] = cfg
        return out

    # ------------------------------------------------------------------
    # Attribute forwarding -------------------------------------------------
    # ------------------------------------------------------------------
    def __getattr__(self, name: str):  # noqa: D401
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model":
                raise
            return getattr(self.model, name)
