import json
import os
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from safetensors.torch import save_model  # type: ignore
from torch import Tensor, nn

from peft.tuners import lora
from peft.tuners.tuners_utils import BaseTuner  # type: ignore

from .classifier import Number, xLoRAClassifier
from .config import xLoRAConfig


class xLoRALayer:
    """
    A xLoRALayer wraps any LoraLayer and performs the xLoRA operation on the LoRA adaptors specified.
    Its primary API is the forward method, which uses the scalings to execute the
    xLoRA algorithm.
    """

    __slots__ = {"model", "target_forward", "target", "layer_number", "config"}

    def __init__(
        self,
        model: nn.Module,  # PeftModel
        target: lora.LoraLayer,
        target_forward: Callable[..., Any],
        layer_number: int,
        config: xLoRAConfig,
    ) -> None:
        self.model = model
        self.target_forward = target_forward
        self.target = target
        self.layer_number = layer_number
        self.config = config

    @staticmethod
    def apply_scalings_to_x(x: torch.Tensor, scalings_layer: torch.Tensor, adapter: int) -> torch.Tensor:
        # scalings_layer = [batch_size, seq_len, n_classes]
        scalings = scalings_layer[:, :, adapter].unsqueeze(-1)
        # scalings_layer = [batch_size, seq_len, 1]
        return x * scalings

    def get_maybe_topk_scalings(self) -> torch.Tensor:
        # xlora_scalings = [batch_size, seq_len, n_classes]
        xlora_scalings: Tensor = self.model.internal_xlora_scalings[:, :, self.layer_number, :]  # type: ignore

        if self.config.top_k_lora is not None:
            _, topk_indices = torch.topk(xlora_scalings, k=self.config.top_k_lora, dim=1)

            # Mask the topk to True, the rest to False
            mask = torch.zeros_like(xlora_scalings, dtype=torch.bool)
            mask.scatter_(1, topk_indices, True)

            xlora_scalings = xlora_scalings * mask.to(xlora_scalings.dtype)

        classifier: xLoRAClassifier = self.model.internal_xlora_classifier  # type: ignore
        if classifier.config.enable_softmax_topk:
            nonzero_mask = xlora_scalings != 0
            softmax_res_nonzero = torch.softmax(xlora_scalings[nonzero_mask], dim=-1)
            xlora_scalings[nonzero_mask] = softmax_res_nonzero

        return xlora_scalings


class xLoRALinearLayer(xLoRALayer):
    def __init__(
        self,
        model: nn.Module,  # PeftModel
        target: lora.Linear,
        target_forward: Callable[..., Any],
        layer_number: int,
        config: xLoRAConfig,
    ) -> None:
        super().__init__(model, target, target_forward, layer_number, config)

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        This method is designed to be a drop-in-replacement for the peft LoRA layers' .forward method.
        To use it, a bound method must be created (bound to an instance of the xLoRALayer class).
        """

        previous_dtype = x.dtype
        xlora_scalings = self.get_maybe_topk_scalings()

        if self.target.disable_adapters:
            if self.target.merged:
                self.target.unmerge()
            result = self.target.base_layer(x, *args, **kwargs)
        elif self.target.merged:
            result = self.target.base_layer(x, *args, **kwargs)
        else:
            result = self.target.base_layer(x, *args, **kwargs)

            for adapter_n, active_adapter in enumerate(self.target.active_adapters):
                if active_adapter not in self.target.lora_A.keys():
                    continue
                lora_A = self.target.lora_A[active_adapter]
                lora_B = self.target.lora_B[active_adapter]
                dropout = self.target.lora_dropout[active_adapter]
                scaling = self.target.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)  # type: ignore
                x_mod = self.apply_scalings_to_x(x, xlora_scalings, adapter_n)
                result += lora_B(lora_A(dropout(x_mod))) * scaling * self.config.global_scaling_weight

        result = result.to(previous_dtype)
        return result


class xLoRAEmbeddingLayer(xLoRALayer):
    def __init__(
        self,
        model: nn.Module,  # PeftModel
        target: lora.Embedding,
        target_forward: Callable[..., Any],
        layer_number: int,
        config: xLoRAConfig,
    ) -> None:
        super().__init__(model, target, target_forward, layer_number, config)

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        This method is designed to be a drop-in-replacement for the peft LoRA layers' .forward method.
        To use it, a bound method must be created (bound to an instance of the xLoRALayer class).
        """

        xlora_scalings = self.get_maybe_topk_scalings()

        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        if self.target.disable_adapters:
            if self.target.merged:
                self.target.unmerge()
            result = self.target.base_layer(x, *args, **kwargs)
        elif self.target.merged:
            result = self.target.base_layer(x, *args, **kwargs)
        else:
            result = self.target.base_layer(x, *args, **kwargs)
            for adapter_n, active_adapter in enumerate(self.target.active_adapters):
                if active_adapter not in self.target.lora_embedding_A:
                    continue
                embedding_A = self.target.lora_embedding_A[active_adapter].T
                embedding_B = self.target.lora_embedding_B[active_adapter].T
                scaling = self.target.scaling[active_adapter]
                x_mod = self.apply_scalings_to_x(x, xlora_scalings, adapter_n)
                after_A = self.target._embed(x_mod, embedding_A)  # type: ignore
                result += (after_A @ embedding_B) * scaling * self.config.global_scaling_weight

        return result


class xLoRAConv2dLayer(xLoRALayer):
    def __init__(
        self,
        model: nn.Module,  # PeftModel
        target: lora.Conv2d,
        target_forward: Callable[..., Any],
        layer_number: int,
        config: xLoRAConfig,
    ) -> None:
        super().__init__(model, target, target_forward, layer_number, config)

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        This method is designed to be a drop-in-replacement for the peft LoRA layers' .forward method.
        To use it, a bound method must be created (bound to an instance of the xLoRALayer class).
        """

        previous_dtype = x.dtype
        xlora_scalings = self.get_maybe_topk_scalings()

        if self.target.disable_adapters:
            if self.target.merged:
                self.target.unmerge()
            result = self.target.base_layer(x, *args, **kwargs)
        elif self.target.merged:
            result = self.target.base_layer(x, *args, **kwargs)
        else:
            result = self.target.base_layer(x, *args, **kwargs)
            for adapter_n, active_adapter in enumerate(self.target.active_adapters):
                if active_adapter not in self.target.lora_A.keys():
                    continue
                lora_A = self.target.lora_A[active_adapter]
                lora_B = self.target.lora_B[active_adapter]
                dropout = self.target.lora_dropout[active_adapter]
                scaling = self.target.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)  # type: ignore
                x_mod = self.apply_scalings_to_x(x, xlora_scalings, adapter_n)
                result += lora_B(lora_A(dropout(x_mod))) * scaling * self.config.global_scaling_weight

        result = result.to(previous_dtype)
        return result


class BaseTunerWrapper:
    def __init__(self, base_model: BaseTuner, classifier: xLoRAClassifier):
        self.model = base_model.model
        self.classifier = classifier

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)  # Important to *call* the model


class PeftModelWrapper:
    def __init__(
        self,
        base_model: nn.Module,  # PeftModel
        base_model_save: Callable[..., None],
        config: xLoRAConfig,
        base_model_get_nb_trainable_parameters: Callable[..., Tuple[int, int]],
        base_model_generate: Callable[..., Any],
    ):
        self.model = base_model
        self.base_model_save = base_model_save
        self.config = config
        self.base_model_get_nb_trainable_parameters = base_model_get_nb_trainable_parameters
        self.base_model_generate = base_model_generate

    def generate(self, *args, **kwargs):
        res = self.base_model_generate(*args, **kwargs)  # type: ignore
        # TODO(EricLBuehler): Evaluate effectiveness and performance degradation
        self.model.base_model.eval()
        if not self.config.use_trainable_adapters:
            for name, param in self.model.base_model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = False
        return res

    def set_topk_lora(self, value: Optional[int]):
        """
        Sparsely select the specified top_k LoRA experts instead of the default dense method. Set to None to use dense. This is reflected in the config.
        """
        classifier: xLoRAClassifier = self.model.internal_xlora_classifier  # type: ignore
        classifier.config.top_k_lora = value

    def get_topk_lora(self) -> Optional[int]:
        """
        Get the current top_k LoRA experts value.
        """
        classifier: xLoRAClassifier = self.model.internal_xlora_classifier  # type: ignore
        return classifier.config.top_k_lora

    def set_global_scaling_weight(self, weight: float):
        """
        Set the global LoRA weight, a scalar to multiply the output of each LoRA adapter by. This is by default 1. This is reflected in the config.
        """
        classifier: xLoRAClassifier = self.model.internal_xlora_classifier  # type: ignore
        classifier.config.global_scaling_weight = weight

    def get_global_scaling_weight(self) -> float:
        """
        Get the global LoRA weight.
        """
        classifier: xLoRAClassifier = self.model.internal_xlora_classifier  # type: ignore
        return classifier.config.global_scaling_weight

    def get_latest_scalings(self) -> Optional[Tensor]:
        """
        Returns the latest scalings prediction, or None if no scalings have been predicted. The tensor is of shape (batch_size, seq_len, n_layers, n_classes).
        """
        return self.model.internal_xlora_scalings

    def get_scalings_log(self) -> List[Tensor]:
        """
        Returns a shallow (only copying the list itself not the tensors) copy of the list containing the scalings log. Editing the list does not change the underlying log.
        The tensors are of shape (batch_size, seq_len, n_layers, n_classes). The seq_len dim may vary with input dimension.
        """
        classifier: xLoRAClassifier = self.model.internal_xlora_classifier  # type: ignore
        return classifier.log_scalings.copy()

    def set_scaling_pass_value(self, value: Union[Number, None]):
        """
        Manually set the scalings to a specific value during the scaling pass, forever. Call this function with None to enable the default
        scalings.

        This is reflected in the config.
        """
        classifier: xLoRAClassifier = self.model.internal_xlora_classifier  # type: ignore
        classifier.set_override_scaling_pass_value(value)

    def print_scalings_predictions(self, n_predictions_lifetime: int):
        """
        Print the scaling states for the next n classifier predictions (i.e. forward, generate passes)
        """
        classifier: xLoRAClassifier = self.model.internal_xlora_classifier  # type: ignore
        classifier.n_predictions_lifetime = n_predictions_lifetime

    def enable_scalings_logging(self):
        """
        Enable scalings logging.
        """
        classifier: xLoRAClassifier = self.model.internal_xlora_classifier  # type: ignore
        classifier.scalings_logging = True

    def disable_scalings_logging(self):
        """
        Disable scalings logging, clearing the log.
        """
        classifier: xLoRAClassifier = self.model.internal_xlora_classifier  # type: ignore
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
        classifier: xLoRAClassifier = self.model.internal_xlora_classifier  # type: ignore
        classifier.flush_log_scalings(path)

    def get_nb_trainable_parameters(self) -> Tuple[int, int]:
        """
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        model_trainable_params, model_all_param = self.base_model_get_nb_trainable_parameters()

        classifier: xLoRAClassifier = self.model.internal_xlora_classifier  # type: ignore
        xlora_trainable_params, xlora_all_param = classifier.get_nb_trainable_parameters()

        trainable_params, all_param = (
            (model_trainable_params + xlora_trainable_params),
            (model_all_param + xlora_all_param),
        )

        return trainable_params, all_param

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model, including of the xLoRA classifier.
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
        for name, param in self.model.base_model.named_parameters():
            if "lora_" in name:
                param.requires_grad = use_trainable_adapters

        self.config.use_trainable_adapters = use_trainable_adapters

    def get_use_trainable_adapters(self) -> bool:
        """
        Get the trainable or not trainable state of the adapters.
        """
        return self.config.use_trainable_adapters

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        selected_adapters: Optional[list[str]] = None,
        save_embedding_layers: Union[str, bool] = "auto",
        is_main_process: bool = True,
        **kwargs: Any,
    ) -> None:
        r"""
        This function saves the classifier weights to a directory. It is the counerpart to `from_pretrained`.

        Args:
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            safe_serialization (`bool`, *optional*):
                Whether to save the adapter files in safetensors format, defaults to `True`.
            is_main_process (`bool`, *optional*):
                Whether the process calling this is the main process or not. Will default to `True`. Will not save the
                checkpoint if not on the main process, which is important for multi device setups (e.g. DDP).
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        if is_main_process:
            os.makedirs(save_directory, exist_ok=True)

        classifier: xLoRAClassifier = self.model.internal_xlora_classifier  # type: ignore

        conf = classifier.config.__dict__.copy()
        del conf["device"]

        if is_main_process:
            os.makedirs(os.path.join(save_directory, "adapters"), exist_ok=True)
        self.base_model_save(
            save_directory=os.path.join(save_directory, "adapters"),
            safe_serialization=safe_serialization,
            is_main_process=is_main_process,
            selected_adapters=selected_adapters,
            save_embedding_layers=save_embedding_layers,
            **kwargs,
        )

        with open(os.path.join(save_directory, "xlora_config.json"), "w") as f:
            json.dump(conf, f)

        if safe_serialization:
            # https://github.com/huggingface/peft/blob/main/src/peft/peft_model.py#L223
            if is_main_process and safe_serialization:
                save_model(classifier, os.path.join(save_directory, "xlora_classifier.safetensors"))
        elif is_main_process:
            state_dict = classifier.state_dict()
            torch.save(state_dict, os.path.join(save_directory, "xlora_classifier.pt"))
