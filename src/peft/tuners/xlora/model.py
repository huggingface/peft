from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from peft.tuners.lora.model import LoraModel
from peft.utils.peft_types import PeftType

from .. import lora
from .classifier import InhibitorFlagPayload, Number, XLoraClassifier
from .config import XLoraConfig
from .insertion import BaseTunerWrapper, PeftModelWrapper, XLoraConv2dLayer, XLoraEmbeddingLayer, XLoraLinearLayer


def convert_layers_to_xlora(
    base: nn.Module,  # PeftModel
    config: XLoraConfig,
) -> int:
    """
    Returns the number of swapped layers.
    """
    total_swapped = 0

    scaling_keys = None
    for module in base.modules():
        if isinstance(module, lora.LoraLayer):
            if not scaling_keys:
                scaling_keys = list(module.scaling.keys())  # NOTE(EricLBuehler): Python 3.7: dicts are ordered!

        if isinstance(module, lora.Linear):
            new_layer: Union[XLoraLinearLayer, XLoraEmbeddingLayer, XLoraConv2dLayer] = XLoraLinearLayer(
                model=base,
                target=module,
                target_forward=module.forward,
                layer_number=total_swapped,
                config=config,
            )
            module.forward = new_layer.forward  # type: ignore[method-assign]
            total_swapped += 1
        elif isinstance(module, lora.Embedding):
            new_layer = XLoraEmbeddingLayer(
                model=base,
                target=module,
                target_forward=module.forward,
                layer_number=total_swapped,
                config=config,
            )
            module.forward = new_layer.forward  # type: ignore[method-assign]
            total_swapped += 1
        elif isinstance(module, lora.Conv2d):
            new_layer = XLoraConv2dLayer(
                model=base,
                target=module,
                target_forward=module.forward,
                layer_number=total_swapped,
                config=config,
            )
            module.forward = new_layer.forward  # type: ignore[method-assign]
            total_swapped += 1

    return total_swapped


class XLoraModel(LoraModel):
    """
    Creates an X-LoRA (Mixture of LoRA experts), model from a pretrained transformers model.

    The method is described in detail in https://arxiv.org/abs/2402.07148.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`XLoraConfig`]): The configuration of the Lora model.
        adapter_name (`str`): The name of the adapter, does not affect the LoRA adapter names.
        model_peft (`PeftModel`): Base peft model.

    Returns:
        `torch.nn.Module`: The X-LoRA model.

    Example:
        ```py
        >>> from transformers import AutoModelForCausalLM, AutoConfig
        >>> from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training

        >>> model_config = AutoConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        >>> config = XLoraConfig(
        ...     task_type="CAUSAL_LM",
        ...     hidden_size=model_config.hidden_size,
        ...     xlora_depth=4,
        ...     adapters={
        ...         "adapter_1": "./path/to/the/checkpoint/",
        ...         "adapter_2": "./path/to/the/checkpoint/",
        ...         "adapter_n": "./path/to/the/checkpoint/",
        ...     },
        ... )

        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "mistralai/Mistral-7B-Instruct-v0.1",
        ...     trust_remote_code=True,
        ...     use_flash_attention_2=False,
        ...     device_map="cuda:0",
        ...     torch_dtype=torch.bfloat16,
        ... )
        >>> model = prepare_model_for_int8_training(model)
        >>> xlora_model = get_peft_model(model, config)
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        config: Union[dict[str, XLoraConfig], XLoraConfig],
        adapter_name: str,
        model_peft: nn.Module,
    ) -> None:
        # model_peft: PeftModel
        if not isinstance(model, PreTrainedModel):
            raise TypeError(f"Expected model type to be 'PreTrainedModel', got '{type(model)}' instead.")
        if isinstance(config, dict):
            if len(config) != 1:
                raise TypeError(f"Expected one config.")
            peft_config = config[adapter_name]
        else:
            peft_config = config
        if not isinstance(peft_config, XLoraConfig):
            raise TypeError(f"Expected config type to be 'XLoraConfig', got '{type(model)}' instead.")

        super().__init__(model, config, adapter_name, model_peft, _disable_inject=True)

        if hasattr(model.config, "use_cache"):
            assert not model.config.use_cache, "`use_cache` must be False"

        use_trainable_adapters = peft_config.use_trainable_adapters
        adapters_items = iter(peft_config.adapters.items())

        # Because we call load_adapter, which requires base_model to be defined
        model_peft.base_model = self
        # For load_adapter to think we are a LoraModel
        model_peft.peft_type = PeftType.LORA

        for adapter_name, model_id in adapters_items:
            model_peft.load_adapter(model_id, adapter_name, is_trainable=use_trainable_adapters)

        self.set_adapter(list(peft_config.adapters.keys()))
        model_peft.peft_type = PeftType.XLORA

        def hook(module, *args, **kwargs) -> None:
            args_real = args[0]
            kwargs_real: dict = args[1]
            kwargs_real.update(kwargs)

            xlora_classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore

            if "_xlora_classifier_inhibitor_flag" in kwargs_real:
                payload: InhibitorFlagPayload = kwargs_real["_xlora_classifier_inhibitor_flag"]

                del kwargs_real["_xlora_classifier_inhibitor_flag"]

                self.internal_xlora_scalings = torch.full(  # type: ignore
                    (payload.batch_size, payload.seq_len, xlora_classifier.n_layers, xlora_classifier.n_classes),
                    payload.override_scaling_pass_value,
                )

                return

            xlora_scalings = xlora_classifier.forward(
                *args_real,
                **kwargs_real,
            )
            # Set the scalings
            self.internal_xlora_scalings = xlora_scalings

        model.register_forward_pre_hook(hook, with_kwargs=True, prepend=True)

        self.eval()
        if not use_trainable_adapters:
            total_frozen = 0
            for name, param in self.named_parameters():
                if "lora_" in name:
                    param.requires_grad = False
                    total_frozen += 1

        total_swapped = convert_layers_to_xlora(
            model_peft,
            peft_config,
        )

        n_classes = len(peft_config.adapters)
        xlora_classifier = XLoraClassifier(model_peft, peft_config, n_classes, total_swapped)

        peft_model_wrapper = PeftModelWrapper(
            model_peft,
            model_peft.save_pretrained,
            peft_config,
            model_peft.get_nb_trainable_parameters,
            model_peft.generate,
        )
        model_peft.save_pretrained = peft_model_wrapper.save_pretrained  # type: ignore
        model_peft.generate = peft_model_wrapper.generate  # type: ignore

        # Setup the model internal state
        self.internal_xlora_classifier = xlora_classifier
        self.internal_xlora_scalings = None  # type: ignore
        self.xlora_config = peft_config

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)  # Important to *call* the model

    def set_topk_lora(self, value: Optional[int]):
        """
        Sparsely select the specified top_k LoRA experts instead of the default dense method. Set to None to use dense. This is reflected in the config.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        classifier.config.top_k_lora = value

    def get_topk_lora(self) -> Optional[int]:
        """
        Get the current top_k LoRA experts value.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        return classifier.config.top_k_lora

    def set_global_scaling_weight(self, weight: float):
        """
        Set the global LoRA weight, a scalar to multiply the output of each LoRA adapter by. This is by default 1. This is reflected in the config.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        classifier.config.global_scaling_weight = weight

    def get_global_scaling_weight(self) -> float:
        """
        Get the global LoRA weight.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        return classifier.config.global_scaling_weight

    def get_latest_scalings(self) -> Optional[torch.Tensor]:
        """
        Returns the latest scalings prediction, or None if no scalings have been predicted. The tensor is of shape (batch_size, seq_len, n_layers, n_classes).
        """
        return self.internal_xlora_scalings

    def get_scalings_log(self) -> List[torch.Tensor]:
        """
        Returns a shallow (only copying the list itself not the tensors) copy of the list containing the scalings log. Editing the list does not change the underlying log.
        The tensors are of shape (batch_size, seq_len, n_layers, n_classes). The seq_len dim may vary with input dimension.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        return classifier.log_scalings.copy()

    def set_scaling_pass_value(self, value: Union[Number, None]):
        """
        Manually set the scalings to a specific value during the scaling pass, forever. Call this function with None to enable the default
        scalings.

        This is reflected in the config.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        classifier.set_override_scaling_pass_value(value)

    def print_scalings_predictions(self, n_predictions_lifetime: int):
        """
        Print the scaling states for the next n classifier predictions (i.e. forward, generate passes)
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        classifier.n_predictions_lifetime = n_predictions_lifetime

    def enable_scalings_logging(self):
        """
        Enable scalings logging.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        classifier.scalings_logging = True

    def disable_scalings_logging(self):
        """
        Disable scalings logging, clearing the log.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        classifier.scalings_logging = False
        classifier.log_scalings = []

    def flush_log_scalings(self, path: str):
        """
        Write the scalings log (a tensor of shape (num_logged, batch_size, seq_len, n_layers, n_classes)) to the specified path.
        If the tensor cannot be constructed, multiple files are written containing tensors of shape
        (num_logged, batch_size, seq_len, n_layers, n_classes) such that each file contains one sequence length. Additionally a JSON
        file is outputted containing the mapping from each sequence log file to the index of the contained tensor so that one may reconstruct
        the log order.

        The file specified should not contain an extension.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        classifier.flush_log_scalings(path)

    def get_nb_trainable_parameters(self) -> Tuple[int, int]:
        """
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        model_trainable_params, model_all_param = self.base_model_get_nb_trainable_parameters()

        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        # Ignoring xlora_trainable_params as it is already included in model_trainable_params
        _xlora_trainable_params, xlora_all_param = classifier.get_nb_trainable_parameters()

        trainable_params, all_param = (
            model_trainable_params,
            (model_all_param + xlora_all_param),
        )

        return trainable_params, all_param

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model, including of the XLora classifier.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()

        print(
            f"trainable params: {trainable_params:,d} || "
            f"all params: {all_param:,d} || "
            f"trainable%: {100 * trainable_params / all_param:.4f}"
        )

    def set_use_trainable_adapters(self, use_trainable_adapters: bool):
        """
        Set the adapters to trainable or not trainable.

        This is reflected in the config.
        """
        for name, param in self.named_parameters():
            if "lora_" in name:
                param.requires_grad = use_trainable_adapters

        self.xlora_config.use_trainable_adapters = use_trainable_adapters

    def get_use_trainable_adapters(self) -> bool:
        """
        Get the trainable or not trainable state of the adapters.
        """
        return self.xlora_config.use_trainable_adapters
