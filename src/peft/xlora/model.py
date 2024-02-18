import os
from typing import Optional, Union

import torch
import torch.nn as nn
from huggingface_hub import file_exists, hf_hub_download  # type: ignore
from huggingface_hub.utils import EntryNotFoundError  # type: ignore
from safetensors.torch import load_file as safe_load_file
from transformers import PreTrainedModel  # type: ignore

from ..peft_model import PeftConfig, PeftModel
from ..tuners import lora
from ..utils.other import (
    infer_device,
)
from .classifier import InhibitorFlagPayload, xLoRAClassifier
from .config import xLoRAConfig
from .insertion import BaseTunerWrapper, PeftModelWrapper, xLoRAConv2dLayer, xLoRAEmbeddingLayer, xLoRALinearLayer


def convert_layers_to_xlora(
    base: PeftModel,
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


class xLoRAModel(PeftModel, PeftModelWrapper):
    def __init__(self, model: nn.Module, peft_config: PeftConfig) -> None:
        assert isinstance(model, PreTrainedModel)
        assert isinstance(peft_config, xLoRAConfig)

        if hasattr(model.config, "use_cache"):
            assert not model.config.use_cache, "`use_cache` must be False"

        use_trainable_adapters = peft_config.use_trainable_adapters
        adapters_items = iter(peft_config.adapters.items())
        first_item = next(adapters_items)
        model_peft = PeftModel.from_pretrained(
            model, first_item[1], first_item[0], is_trainable=use_trainable_adapters
        )

        for adapter_name, model_id in adapters_items:
            model_peft.load_adapter(model_id, adapter_name, is_trainable=use_trainable_adapters)

        model_peft.base_model.set_adapter(list(peft_config.adapters.keys()))

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
                    payload.override_scaling_pass_value,  # requires_grad=True
                )  # TODO(EricLBuehler): is the requires_grad=True necessary?

                return

            xlora_scalings = xlora_classifier.forward(
                *args_real,
                **kwargs_real,
            )
            # Set the scalings
            model_peft.internal_xlora_scalings = xlora_scalings

        model.register_forward_pre_hook(hook, with_kwargs=True, prepend=True)

        model_peft.base_model.eval()
        if not use_trainable_adapters:
            total_frozen = 0
            for name, param in model_peft.base_model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = False
                    total_frozen += 1

        assert isinstance(model_peft.base_model, lora.LoraModel)

        total_swapped = convert_layers_to_xlora(
            model_peft,
            peft_config,
        )

        n_classes = len(peft_config.adapters)
        xlora_classifier = xLoRAClassifier(model_peft, peft_config, n_classes, total_swapped)

        # Setup the internal state
        base_model_wrapper = BaseTunerWrapper(model_peft.base_model, xlora_classifier)
        model_peft.base_model.forward = base_model_wrapper.forward  # type: ignore[method-assign]

        peft_model_wrapper = PeftModelWrapper(
            model_peft,
            model_peft.save_pretrained,
            peft_config,
            model_peft.get_nb_trainable_parameters,
            model_peft.generate,
        )
        model_peft.save_pretrained = peft_model_wrapper.save_pretrained  # type: ignore[method-assign]
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


def _load_classifier_weights(model_id: str, device: Optional[str] = None, **hf_hub_download_kwargs) -> dict:
    r"""
    A helper method to load the classifier weights from the HuggingFace Hub or locally. Copied from load_peft_weights

    Args:
        model_id (`str`):
            The local path to the adapter weights or the name of the adapter to load from the HuggingFace Hub.
        device (`str`):
            The device to load the weights onto.
        hf_hub_download_kwargs (`dict`):
            Additional arguments to pass to the `hf_hub_download` method when loading from the HuggingFace Hub.
    """
    path = (
        os.path.join(model_id, hf_hub_download_kwargs["subfolder"])
        if hf_hub_download_kwargs.get("subfolder", None) is not None
        else model_id
    )

    SAFETENSORS_WEIGHTS_NAME = "xlora_classifier.safetensors"
    WEIGHTS_NAME = "xlora_classifier.pt"

    if device is None:
        device = infer_device()

    if os.path.exists(os.path.join(path, SAFETENSORS_WEIGHTS_NAME)):
        filename = os.path.join(path, SAFETENSORS_WEIGHTS_NAME)
        use_safetensors = True
    elif os.path.exists(os.path.join(path, WEIGHTS_NAME)):
        filename = os.path.join(path, WEIGHTS_NAME)
        use_safetensors = False
    else:
        token = hf_hub_download_kwargs.get("token", None)
        if token is None:
            token = hf_hub_download_kwargs.get("use_auth_token", None)

        hub_filename = (
            os.path.join(hf_hub_download_kwargs["subfolder"], SAFETENSORS_WEIGHTS_NAME)
            if hf_hub_download_kwargs.get("subfolder", None) is not None
            else SAFETENSORS_WEIGHTS_NAME
        )
        has_remote_safetensors_file = file_exists(
            repo_id=model_id,
            filename=hub_filename,
            revision=hf_hub_download_kwargs.get("revision", None),
            repo_type=hf_hub_download_kwargs.get("repo_type", None),
            token=token,
        )
        use_safetensors = has_remote_safetensors_file

        if has_remote_safetensors_file:
            # Priority 1: load safetensors weights
            filename = hf_hub_download(
                model_id,
                SAFETENSORS_WEIGHTS_NAME,
                **hf_hub_download_kwargs,
            )
        else:
            try:
                filename = hf_hub_download(model_id, WEIGHTS_NAME, **hf_hub_download_kwargs)
            except EntryNotFoundError:
                raise ValueError(
                    f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {WEIGHTS_NAME} or {SAFETENSORS_WEIGHTS_NAME} is present at {model_id}."
                )

    if use_safetensors:
        if hasattr(torch.backends, "mps") and (device == torch.device("mps")):
            adapters_weights = safe_load_file(filename, device="cpu")
        else:
            adapters_weights = safe_load_file(filename, device=device)
    else:
        adapters_weights = torch.load(filename, map_location=torch.device(device))

    return adapters_weights


def _get_file_path_dir(load_directory: Union[str, os.PathLike], name: str, dir: str) -> str:
    if os.path.exists(os.path.join(load_directory, dir, name)):
        return os.path.join(load_directory, dir, name)
    return hf_hub_download(load_directory, filename=name, subfolder=dir)
