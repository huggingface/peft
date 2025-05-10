from __future__ import annotations
from dataclasses import asdict
from enum import Enum
from typing import Any, Optional, Union
import warnings

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _get_submodules,
)
from peft.import_utils import is_bnb_available, is_bnb_4bit_available

from .config import UILinLoRAConfig
from .layer import UILinLoRALayer, Linear

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


class UILinLoRAModel(BaseTuner):
    """Parameter‑efficient row-based adapter where only the first row of each weight matrix is trainable.

    The public API (prepare, merge_and_unload, unload, etc.) matches
    HuggingFace PEFT tuners.
    """

    prefix: str = "uilinlora_"  # used for freezing logic

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------
    def __init__(self, model, config: UILinLoRAConfig, adapter_name: str, *, low_cpu_mem_usage: bool = False):
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    # ------------------------------------------------------------------
    # Adapter‑config validation
    # ------------------------------------------------------------------
    def _check_new_adapter_config(self, config: UILinLoRAConfig) -> None:  # noqa: D401
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. "
                "When using multiple adapters, set bias='none' for all adapters."
            )

    # ------------------------------------------------------------------
    # Target‑layer detection helpers (mirrors Vera)
    # ------------------------------------------------------------------
    @staticmethod
    def _check_target_module_exists(uilinlora_config: UILinLoRAConfig, key: str) -> bool:  # noqa: D401
        return check_target_module_exists(uilinlora_config, key)

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------
    def _create_and_replace(
        self,
        uilinlora_config: UILinLoRAConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        **optional_kwargs,
    ) -> None:
        """Wrap one module with our UILinLoRALayer if eligible."""

        # 1. Get base layer
        base = target.get_base_layer() if hasattr(target, "get_base_layer") else target

        # 2. Check valid types
        valid_linear_types = (nn.Linear, Linear4bit, Linear8bitLt)
        if not isinstance(base, valid_linear_types + (torch.nn.Conv1d,)):
            return

        # 3. Match target_modules with full key
        if not self._check_target_module_exists(uilinlora_config, current_key):
            return
        
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "rank": uilinlora_config.rank,
            "uilinlora_dropout": uilinlora_config.uilinlora_dropout,
            "fan_in_fan_out": uilinlora_config.fan_in_fan_out,
            "init_weights": uilinlora_config.init_uilinlora_weights,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }
        kwargs["bias"] = bias

        # 4. Already wrapped
        if isinstance(target, Linear):
            target.update_layer(
                adapter_name,
                uilinlora_config.uilinlora_alpha,
                uilinlora_config.uilinlora_dropout,
                uilinlora_config.init_uilinlora_weights,
                uilinlora_config.bias,
            )
            return

        # 5. Fresh wrap
        new_module = self._create_new_module(uilinlora_config, adapter_name, target, **optional_kwargs, **kwargs)
        if adapter_name not in self.active_adapter:
            new_module.disable_adapters = True  # freeze if inactive
        self._replace_module(parent, target_name, new_module, target)


    # Factory for wrapper -------------------------------------------------------
    @staticmethod
    def _create_new_module(
        uilinlora_config: UILinLoRAConfig,
        adapter_name: str,
        target: nn.Module,
        **kwargs,
    ) -> UILinLoRALayer:
        # ── lazy-import bitsandbytes wrappers ─────────────────────────────
        if is_bnb_available():
            import bitsandbytes as bnb
            from .bnb import Linear8bitLt            # your quant wrapper

        if is_bnb_4bit_available():
            from .bnb import Linear4bit
        
        # caller may pass these markers
        loaded_in_8bit = kwargs.get("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.get("loaded_in_4bit", False)
        bias           = kwargs.pop("bias", "none")

        # unwrap if target is already a PEFT layer
        target_base = target.get_base_layer() if isinstance(target, BaseTunerLayer) else target

        # ──────────────────────────────────────────────────────────────────
        # choose the correct UILinLoRA wrapper
        # ──────────────────────────────────────────────────────────────────
        if loaded_in_8bit and isinstance(target_base, bnb.nn.Linear8bitLt):
            eightbit_kwargs       = kwargs.copy()
            eightbit_kwargs.update(
                has_fp16_weights = target_base.state.has_fp16_weights,
                threshold        = target_base.state.threshold,
                index            = target_base.index,
            )
            cls = Linear8bitLt

        elif loaded_in_4bit and isinstance(target_base, bnb.nn.Linear4bit):
            fourbit_kwargs        = kwargs.copy()
            fourbit_kwargs.update(
                compute_dtype       = target_base.compute_dtype,
                compress_statistics = target_base.weight.compress_statistics,
                quant_type          = target_base.weight.quant_type,
            )
            cls = Linear4bit
            kwargs = fourbit_kwargs

        else:
            # plain FP32/FP16 path
            cls = Linear
            # fan_in_fan_out sanity like VeRA
            if isinstance(target_base, nn.Linear) and kwargs.get("fan_in_fan_out", False):
                warnings.warn("fan_in_fan_out=True on torch.nn.Linear – forcing to False.")
                kwargs["fan_in_fan_out"] = uilinlora_config.fan_in_fan_out = False

        # mark conv1d layers
        if hasattr(torch.nn, "Conv1d") and isinstance(target_base, torch.nn.Conv1d):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs.get("fan_in_fan_out", False):
                warnings.warn("Conv1d needs fan_in_fan_out=True – forcing it.")
                kwargs["fan_in_fan_out"] = uilinlora_config.fan_in_fan_out = True

        # ── common args forwarded to every wrapper ────────────────────────
        common = dict(
            rank           = uilinlora_config.rank,
            scaling_factor = uilinlora_config.scaling_factor,
            enforce_sv_positive = uilinlora_config.enforce_sv_positive,
            bias           = bias,
            uilinlora_alpha = uilinlora_config.uilinlora_alpha,
            uilinlora_dropout = uilinlora_config.uilinlora_dropout,
            init_uilinlora_weights = uilinlora_config.init_uilinlora_weights,
            fan_in_fan_out = uilinlora_config.fan_in_fan_out,
        )
        kwargs.update(common)

        # instantiate and return
        return cls(target, adapter_name, **kwargs)


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
    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        # for active_adapter in self.active_adapters:
        #     bias = self.peft_config[active_adapter].bias
        #     if bias == "none":
        #         continue

        #     if bias == "all":
        #         for n, p in model.named_parameters():
        #             if "bias" in n:
        #                 p.requires_grad = True
        #     elif bias == "uilinlora_only":
        #         for m in model.modules():
        #             if isinstance(m, UILinLoRALayer) and hasattr(m, "uilinlora_bias"):
        #                 if active_adapter in m.uilinlora_bias:
        #                     m.uilinlora_bias[active_adapter].requires_grad = True
        #     else:
        #         raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")


    # ------------------------------------------------------------------
    # (Un)merging helpers
    # ------------------------------------------------------------------
    def merge_adapter(self) -> None:  # noqa: D401
        for m in self.model.modules():
            if isinstance(m, UILinLoRALayer):
                m.merge()

    def unmerge_adapter(self) -> None:  # noqa: D401
        for m in self.model.modules():
            if isinstance(m, UILinLoRALayer):
                m.unmerge()

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------
    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for m in self.model.modules():
            if isinstance(m, UILinLoRALayer):
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
            if isinstance(m, UILinLoRALayer):
                m.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
    def _prepare_adapter_config(self, cfg: UILinLoRAConfig, model_cfg):  # noqa: D401
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
