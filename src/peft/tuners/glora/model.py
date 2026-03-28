from dataclasses import asdict
from enum import Enum
from typing import Any, Optional, Union

import torch.nn as nn
from tqdm import tqdm

from peft.config import PeftConfig
from peft.tuners.glora.layer import GloraLayer
from peft.tuners.tuners_utils import BaseTuner
from peft.utils import (
    TRANSFORMERS_MODELS_TO_GLORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
)
from peft.utils.peft_types import PeftType

from .config import GloraConfig
from .layer import GloraLinear


def mark_only_glora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """
    Freezes all parameters of the model except the GLORA parameters. If bias is 'glora_only', 'all', or
    'some_other_custom', it handles bias terms as well.
    """
    for n, p in model.named_parameters():
        if "glora_" not in n:
            p.requires_grad = False

    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "glora_only":
        for m in model.modules():
            if isinstance(m, GloraLinear) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True


class GloraModel(BaseTuner):
    """
    Creates Generalized Low Rank Adapter (Glora) model from a pretrained transformers model.
    """

    def __init__(self, model: nn.Module, config: GloraConfig, adapter_name: str = "default"):
        super().__init__(model, config, adapter_name)
        self.model = model
        self.forward = self.model.forward

        self.peft_config: dict[str, GloraConfig] = {}
        self.active_adapter: Union[str, list[str]] = adapter_name
        self.peft_type = PeftType.GLORA
        self.adapters_config_history: dict[str, Any] = {}

        # Accept both single config and dict of configs
        if isinstance(config, GloraConfig):
            self.peft_config[adapter_name] = config
        elif isinstance(config, dict):
            for name, cfg in config.items():
                self.peft_config[name] = cfg
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        # Add all adapters after peft_config is set
        for name, cfg in self.peft_config.items():
            self.add_adapter(name, cfg)

    def add_adapter(self, adapter_name: str, config: GloraConfig):
        # Avoid re-adding if already present
        if hasattr(self, "_added_adapters") and adapter_name in self._added_adapters:
            return
        if not hasattr(self, "_added_adapters"):
            self._added_adapters = set()

        # Prepare config (resolve target_modules if needed)
        model_config_dict = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
        current_config = self._prepare_peft_config(config, model_config_dict)
        self.peft_config[adapter_name] = current_config

        # Replace or add adapters in target modules
        self._find_and_replace(adapter_name)

        # Mark only Glora params as trainable
        mark_only_glora_as_trainable(self.model, bias=getattr(current_config, "bias", "none"))

        # Optionally freeze for inference
        if getattr(current_config, "inference_mode", False):
            _freeze_adapter(self.model, adapter_name)

        self._added_adapters.add(adapter_name)

    def _create_new_module(self, peft_config: GloraConfig, adapter_name: str, target: nn.Module) -> GloraLinear:
        bias = hasattr(target, "bias") and target.bias is not None
        if not isinstance(target, nn.Linear):
            raise ValueError(
                f"Target module {target} is not a nn.Linear layer, which is required for GLORA replacement."
            )

        in_features, out_features = target.in_features, target.out_features
        kwargs_glora = {
            "config_A_B": peft_config.config_A_B,
            "config_C": peft_config.config_C,
            "config_D_E": peft_config.config_D_E,
        }
        new_module = GloraLinear(in_features, out_features, bias=bias, **kwargs_glora)
        # Add the adapter to the new module
        new_module.add_adapter(
            adapter_name,
            peft_config.r,
            peft_config.config_A_B,
            peft_config.config_C,
            peft_config.config_D_E,
        )
        return new_module

    def _find_and_replace(self, adapter_name: str):
        peft_config = self.peft_config[adapter_name]
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]  # Cache keys

        for key in key_list:
            if not self._check_target_module_exists(peft_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)

            if isinstance(target, GloraLinear):
                # Add adapter to existing GloraLinear
                target.add_adapter(
                    adapter_name,
                    peft_config.r,
                    peft_config.config_A_B,
                    peft_config.config_C,
                    peft_config.config_D_E,
                )
            elif isinstance(target, nn.Linear):
                new_module = self._create_new_module(peft_config, adapter_name, target)
                self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent: nn.Module, child_name: str, new_module: nn.Module, child: nn.Module) -> None:
        setattr(parent, child_name, new_module)
        # Copy weights and bias
        if hasattr(child, "weight") and hasattr(new_module, "weight"):
            new_module.weight = child.weight
        if hasattr(child, "bias") and hasattr(new_module, "bias") and child.bias is not None:
            new_module.bias = child.bias
        # Copy state if present
        if getattr(child, "state", None) is not None:
            new_module.state = child.state
            new_module.to(child.weight.device)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model":
                raise
            return getattr(self.model, name)

    @staticmethod
    def _prepare_peft_config(peft_config: GloraConfig, model_config: dict) -> GloraConfig:
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_GLORA_TARGET_MODULES_MAPPING:
                raise ValueError(
                    f"Please specify `target_modules` in `GloraConfig` for model_type {model_config['model_type']}"
                )
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_GLORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        return peft_config

    def set_adapter(self, adapter_name: str | list[str], inference_mode: bool = False) -> None:
        if self.active_adapter == adapter_name:
            return

        for module in self.model.modules():
            if isinstance(module, GloraLayer):
                module.set_adapter(adapter_name, inference_mode=inference_mode)
        self.active_adapter = adapter_name

    def enable_adapter_layers(self):
        for module in self.model.modules():
            if hasattr(module, "enable_adapters"):
                module.enable_adapters()

    def disable_adapter_layers(self):
        for module in self.model.modules():
            if hasattr(module, "disable_adapters"):
                module.disable_adapters()

    def delete_adapter(self, adapter_name: str):
        if adapter_name not in self.peft_config:
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]
        for module in self.model.modules():
            if hasattr(module, "delete_adapter"):
                module.delete_adapter(adapter_name)
        # Update active_adapter if needed
        if self.active_adapter == adapter_name:
            self.active_adapter = next(iter(self.peft_config.keys()), None)

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ):
        """
        This method merges the Glora layers into the base model.
        """
        if getattr(self, "hf_device_map", None):
            raise ValueError("Merging LoRA weights is not supported when using HF device map.")

        key_list = [key for key, _ in self.model.named_modules()]
        desc = "Merging GLORA layers"

        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue

            if isinstance(target, GloraLinear):
                if not target.active_adapters:
                    continue
                # Merge all or specified adapters
                merge_adapters = adapter_names if adapter_names is not None else target.active_adapters
                target.merge(safe_merge=safe_merge, adapter_names=merge_adapters)
                new_module = nn.Linear(target.in_features, target.out_features, bias=(target.bias is not None))
                new_module.weight.data = target.weight.data.clone()  # Get merged weight
                if target.bias is not None:
                    new_module.bias.data = target.bias.data.clone()  # Get merged bias
                self._replace_module(parent, target_name, new_module.to(target.weight.device), target)

            if isinstance(target, ModulesToSaveWrapper):
                pass
        return self.model

    def set_adapter_eval_config(self, adapter_name: str, eval_config: dict[str, str]):
        """
        Sets the evaluation configuration for all GloraLinear layers associated with a given adapter. The eval_config
        dictionary should specify the path choices for A, B, C, D, E. Example: {'A':'lora_8', 'B':'none', 'C':'vector',
        'D':'constant', 'E':'none'}
        """
        if adapter_name not in self.peft_config:
            raise ValueError(f"Adapter {adapter_name} not found.")

        for module in self.model.modules():
            if isinstance(module, GloraLinear):
                if adapter_name in module.eval_config:
                    module.eval_config[adapter_name] = eval_config
                    self.adapters_config_history[adapter_name] = eval_config

    def get_peft_config_as_dict(self, inference: bool = False) -> dict[str, Any]:
        config_dict = {}
        for adapter_name, peft_config_obj in self.peft_config.items():
            config = asdict(peft_config_obj)
            if inference:
                config["inference_mode"] = True
            for k, v in config.items():
                if isinstance(v, Enum):
                    config[k] = v.value
            config_dict[adapter_name] = config
        return config_dict

    def _create_and_replace(
        self,
        peft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        parameter_name: Optional[str] = None,
    ):
        new_module = self._create_new_module(peft_config, adapter_name, target)
        self._replace_module(parent, target_name, new_module, target)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        mark_only_glora_as_trainable(model)

    def _prepare_adapter_config(self, peft_config: PeftConfig, model_config: dict) -> PeftConfig:
        assert isinstance(peft_config, GloraConfig) # ty linting
        return GloraModel._prepare_peft_config(peft_config, model_config)
