import json
import os
from typing import Any, Callable, Optional, Tuple, Union

import torch
from safetensors.torch import save_model  # type: ignore
from torch import Tensor, nn

from peft.tuners import lora
from peft.tuners.tuners_utils import BaseTuner
from peft.tuners.xlora.model import xLoRAModel  # type: ignore

from .classifier import xLoRAClassifier
from .config import xLoRAConfig


class xLoRALayer:
    """
    A xLoRALayer wraps any LoraLayer and performs the xLoRA operation on the LoRA adaptors specified.
    Its primary API is the forward method, which uses the scalings to execute the
    xLoRA algorithm.
    """

    def __init__(
        self,
        model: xLoRAModel,
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
    def __init__(self, base_model: BaseTuner):
        self.model = base_model.model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)  # Important to *call* the model


class PeftModelWrapper:
    def __init__(
        self,
        peft_model: nn.Module,  # PeftModel
        base_model_save: Callable[..., None],
        config: xLoRAConfig,
        base_model_get_nb_trainable_parameters: Callable[..., Tuple[int, int]],
        base_model_generate: Callable[..., Any],
    ):
        self.peft_model = peft_model
        self.base_model_save = base_model_save
        self.config = config
        self.base_model_get_nb_trainable_parameters = base_model_get_nb_trainable_parameters
        self.base_model_generate = base_model_generate

    def generate(self, *args, **kwargs):
        res = self.base_model_generate(*args, **kwargs)  # type: ignore
        # TODO(EricLBuehler): Evaluate effectiveness and performance degradation
        self.peft_model.base_model.eval()
        if not self.config.use_trainable_adapters:
            for name, param in self.peft_model.base_model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = False
        return res

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

        classifier: xLoRAClassifier = self.peft_model.base_model.internal_xlora_classifier  # type: ignore

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
