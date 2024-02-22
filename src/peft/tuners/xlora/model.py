from typing import Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from peft.tuners.lora.model import LoraModel
from peft.utils.peft_types import PeftType

from .. import lora
from .classifier import InhibitorFlagPayload, xLoRAClassifier
from .config import xLoRAConfig
from .insertion import BaseTunerWrapper, PeftModelWrapper, xLoRAConv2dLayer, xLoRAEmbeddingLayer, xLoRALinearLayer


def convert_layers_to_xlora(
    base: nn.Module,  # PeftModel
    config: xLoRAConfig,
) -> int:
    """
    Returns the number of swapped layers.
    """
    assert isinstance(base.base_model, lora.LoraModel)
    total_swapped = 0

    scaling_keys = None
    for module in base.modules():
        if isinstance(module, lora.LoraLayer):
            if not scaling_keys:
                scaling_keys = list(module.scaling.keys())  # NOTE(EricLBuehler): Python 3.7: dicts are ordered!

        if isinstance(module, lora.Linear):
            assert scaling_keys is not None
            new_layer: Union[xLoRALinearLayer, xLoRAEmbeddingLayer, xLoRAConv2dLayer] = xLoRALinearLayer(
                model=base,
                target=module,
                target_forward=module.forward,
                layer_number=total_swapped,
                config=config,
            )
            module.forward = new_layer.forward  # type: ignore[method-assign]
            total_swapped += 1
        elif isinstance(module, lora.Embedding):
            assert scaling_keys is not None
            new_layer = xLoRAEmbeddingLayer(
                model=base,
                target=module,
                target_forward=module.forward,
                layer_number=total_swapped,
                config=config,
            )
            module.forward = new_layer.forward  # type: ignore[method-assign]
            total_swapped += 1
        elif isinstance(module, lora.Conv2d):
            assert scaling_keys is not None
            new_layer = xLoRAConv2dLayer(
                model=base,
                target=module,
                target_forward=module.forward,
                layer_number=total_swapped,
                config=config,
            )
            module.forward = new_layer.forward  # type: ignore[method-assign]
            total_swapped += 1

    return total_swapped


class xLoRAModel(LoraModel):
    """
    Creates an X-LoRA (Mixture of LoRA experts), model from a pretrained transformers model.

    The method is described in detail in https://arxiv.org/abs/2402.07148.

    The X-LoRA model modifies the methods of `model_peft` to inject its own API.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`xLoRAConfig`]): The configuration of the Lora model.
        adapter_name (`str`): The name of the adapter, does not affect the LoRA adapter names.
        model_peft (`PeftModel`): Base peft model.

    Returns:
        `torch.nn.Module`: The X-LoRA model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM, AutoConfig
        >>> from peft import LoraModel, LoraConfig

        >>> model_config = AutoConfig.from_pretrained("t5-base")
        >>> config = xLoRAConfig(
        ...     task_type="SEQ_2_SEQ_LM",
        ...     hidden_size=model_config.hidden_size,
        ...     xlora_depth=3,
        ...     adapters={
        ...         "adapter_1": "./path/to/the/checkpoint/",
        ...         "adapter_2": "./path/to/the/checkpoint/",
        ...         "adapter_n": "./path/to/the/checkpoint/",
        ...     },
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> xlora_model = xLoRAModel(model, config, "xlora")
        ```

        ```py
        >>> from transformers import AutoModelForCausalLM, AutoConfig
        >>> from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training

        >>> model_config = AutoConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        >>> config = xLoRAConfig(
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
        config: Union[dict[str, xLoRAConfig], xLoRAConfig],
        adapter_name: str,
        model_peft: nn.Module,
    ) -> None:
        # model_peft: PeftModel
        assert isinstance(model, PreTrainedModel)
        if isinstance(config, dict):
            assert len(config) == 1
            peft_config = config[adapter_name]
        else:
            peft_config = config
        assert isinstance(peft_config, xLoRAConfig)

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

            xlora_classifier: xLoRAClassifier = model_peft.internal_xlora_classifier  # type: ignore

            if "_xlora_classifier_inhibitor_flag" in kwargs_real:
                payload: InhibitorFlagPayload = kwargs_real["_xlora_classifier_inhibitor_flag"]

                del kwargs_real["_xlora_classifier_inhibitor_flag"]

                model_peft.internal_xlora_scalings = torch.full(  # type: ignore
                    (payload.batch_size, payload.seq_len, xlora_classifier.n_layers, xlora_classifier.n_classes),
                    payload.override_scaling_pass_value,
                )

                return

            xlora_scalings = xlora_classifier.forward(
                *args_real,
                **kwargs_real,
            )
            # Set the scalings
            model_peft.internal_xlora_scalings = xlora_scalings

        model.register_forward_pre_hook(hook, with_kwargs=True, prepend=True)

        self.eval()
        if not use_trainable_adapters:
            total_frozen = 0
            for name, param in self.named_parameters():
                if "lora_" in name:
                    param.requires_grad = False
                    total_frozen += 1

        assert isinstance(self, LoraModel)

        total_swapped = convert_layers_to_xlora(
            model_peft,
            peft_config,
        )

        n_classes = len(peft_config.adapters)
        xlora_classifier = xLoRAClassifier(model_peft, peft_config, n_classes, total_swapped)

        # Setup the internal state
        base_model_wrapper = BaseTunerWrapper(self, xlora_classifier)
        self.forward = base_model_wrapper.forward  # type: ignore[method-assign]

        peft_model_wrapper = PeftModelWrapper(
            model_peft,
            model_peft.save_pretrained,
            peft_config,
            model_peft.get_nb_trainable_parameters,
            model_peft.generate,
        )
        model_peft.save_pretrained = peft_model_wrapper.save_pretrained  # type: ignore
        model_peft.generate = peft_model_wrapper.generate  # type: ignore

        assert not hasattr(model_peft, "set_use_trainable_adapters")
        model_peft.set_use_trainable_adapters = peft_model_wrapper.set_use_trainable_adapters  # type: ignore

        assert not hasattr(model_peft, "print_scalings_predictions")
        model_peft.print_scalings_predictions = peft_model_wrapper.print_scalings_predictions  # type: ignore

        assert not hasattr(model_peft, "enable_scalings_logging")
        model_peft.enable_scalings_logging = peft_model_wrapper.enable_scalings_logging  # type: ignore

        assert not hasattr(model_peft, "disable_scalings_logging")
        model_peft.disable_scalings_logging = peft_model_wrapper.disable_scalings_logging  # type: ignore

        assert not hasattr(model_peft, "flush_log_scalings")
        model_peft.flush_log_scalings = peft_model_wrapper.flush_log_scalings  # type: ignore

        assert not hasattr(model_peft, "get_scalings_log")
        model_peft.get_scalings_log = peft_model_wrapper.get_scalings_log  # type: ignore

        assert not hasattr(model_peft, "set_scaling_pass_value")
        model_peft.set_scaling_pass_value = peft_model_wrapper.set_scaling_pass_value  # type: ignore

        assert not hasattr(model_peft, "set_global_scaling_weight")
        model_peft.set_global_scaling_weight = peft_model_wrapper.set_global_scaling_weight  # type: ignore

        assert not hasattr(model_peft, "get_global_scaling_weight")
        model_peft.get_global_scaling_weight = peft_model_wrapper.get_global_scaling_weight  # type: ignore

        assert not hasattr(model_peft, "set_topk_lora")
        model_peft.set_topk_lora = peft_model_wrapper.set_topk_lora  # type: ignore

        assert not hasattr(model_peft, "get_topk_lora")
        model_peft.get_topk_lora = peft_model_wrapper.get_topk_lora  # type: ignore

        model_peft.get_nb_trainable_parameters = peft_model_wrapper.get_nb_trainable_parameters  # type: ignore

        model_peft.print_trainable_parameters = peft_model_wrapper.print_trainable_parameters  # type: ignore

        # Setup the model internal state
        assert not hasattr(model_peft, "internal_xlora_classifier")
        model_peft.internal_xlora_classifier = xlora_classifier

        assert not hasattr(model_peft, "internal_xlora_scalings")
        model_peft.internal_xlora_scalings = None  # type: ignore
