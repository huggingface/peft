from dataclasses import asdict
from enum import Enum
from typing import Any, Optional

from torch import nn

from peft.config import PeftConfig
from peft.tuners.glora.layer import GloraLayer
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_GLORA_TARGET_MODULES_MAPPING

from .config import GloraConfig
from .layer import GloraLinear


class GloraModel(BaseTuner):
    """
    Creates Generalized Low Rank Adapter (Glora) model from a pretrained transformers model.
    """

    prefix: str = "glora_"
    tuner_layer_cls = GloraLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_GLORA_TARGET_MODULES_MAPPING

    def __init__(
        self,
        model: nn.Module,
        config: GloraConfig,
        adapter_name: str = "default",
        low_cpu_mem_usage: bool = False,
        state_dict: Optional[dict[str, Any]] = None,
    ):
        self.adapters_config_history: dict[str, Any] = {}
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage, state_dict=state_dict)

    def _create_and_replace(
        self,
        peft_config: GloraConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        parameter_name: Optional[str] = None,
    ) -> None:
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        if isinstance(target, GloraLinear):
            target.add_adapter(
                adapter_name,
                peft_config.r,
                peft_config.config_A_B,
                peft_config.config_C,
                peft_config.config_D_E,
            )
            if adapter_name not in self.active_adapters:
                target.requires_grad_(False)
        else:
            new_module = self._create_new_module(peft_config, adapter_name, target, current_key)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(
        peft_config: GloraConfig,
        adapter_name: str,
        target: nn.Module,
        current_key: str | None = None,
        **optional_kwargs: Any,
    ) -> GloraLinear:
        del optional_kwargs  # unused; accepted for signature compatibility with the tuner injection path
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target
        if type(target_base_layer) is not nn.Linear:
            raise ValueError(
                f"Target module {target} is not a plain torch.nn.Linear (after unwrapping); GLORA does not support this layer type."
            )

        kwargs_glora = {
            "config_A_B": peft_config.config_A_B,
            "config_C": peft_config.config_C,
            "config_D_E": peft_config.config_D_E,
        }
        new_module = GloraLinear(target, **kwargs_glora)
        new_module.add_adapter(
            adapter_name,
            peft_config.r,
            peft_config.config_A_B,
            peft_config.config_C,
            peft_config.config_D_E,
        )
        return new_module

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

    def set_adapter_eval_config(self, adapter_name: str, eval_config: dict[str, str]):
        """
        Sets the evaluation configuration for all GloraLinear layers associated with a given adapter. The eval_config
        dictionary should specify the path choices for A, B, C, D, E. Example: {'A':'lora_8', 'B':'none', 'C':'vector',
        'D':'constant', 'E':'none'}
        """
        if adapter_name not in self.peft_config:
            raise ValueError(f"Adapter {adapter_name} not found.")

        for module in self.model.modules():
            if isinstance(module, GloraLinear) and adapter_name in module.eval_config:
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

    def _prepare_adapter_config(self, peft_config: PeftConfig, model_config: dict) -> PeftConfig:
        assert isinstance(peft_config, GloraConfig)  # ty linting
        return GloraModel._prepare_peft_config(peft_config, model_config)
