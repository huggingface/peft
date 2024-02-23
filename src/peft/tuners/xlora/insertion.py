import json
import os
from typing import Any, Callable, Optional, Union

import torch
from safetensors.torch import save_model  # type: ignore
from torch import Tensor, nn

from peft.tuners import lora

from .classifier import XLoraClassifier
from .config import XLoraConfig


class XLoraLayer:
    """
    A XLoraLayer wraps any LoraLayer and performs the XLora operation on the LoRA adaptors specified.
    Its primary API is the forward method, which uses the scalings to execute the
    XLora algorithm.
    """

    def __init__(
        self,
        model: nn.Module,  # PeftModel
        target: lora.LoraLayer,
        target_forward: Callable[..., Any],
        layer_number: int,
        config: XLoraConfig,
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

        classifier: XLoraClassifier = self.model.base_model.internal_xlora_classifier  # type: ignore
        if classifier.config.enable_softmax_topk:
            nonzero_mask = xlora_scalings != 0
            softmax_res_nonzero = torch.softmax(xlora_scalings[nonzero_mask], dim=-1)
            xlora_scalings[nonzero_mask] = softmax_res_nonzero

        return xlora_scalings

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        return self.target.forward(x, *args, **kwargs)


class PeftModelWrapper:
    def __init__(
        self,
        peft_model: nn.Module,  # PeftModel
        base_model_save: Callable[..., None],
    ):
        self.peft_model = peft_model
        self.base_model_save = base_model_save

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

        classifier: XLoraClassifier = self.peft_model.base_model.internal_xlora_classifier  # type: ignore

        conf = classifier.config.__dict__.copy()
        del conf["device"]

        self.base_model_save(
            save_directory=save_directory,
            safe_serialization=safe_serialization,
            is_main_process=is_main_process,
            selected_adapters=selected_adapters,
            save_embedding_layers=save_embedding_layers,
            **kwargs,
        )

        conf["adapters"] = {
            name: os.path.join(save_directory, name) if name != "default" else save_directory
            for name in conf["adapters"]
        }
        with open(os.path.join(save_directory, "xlora_config.json"), "w") as f:
            json.dump(conf, f)

        if safe_serialization:
            # https://github.com/huggingface/peft/blob/main/src/peft/peft_model.py#L223
            if is_main_process and safe_serialization:
                save_model(classifier, os.path.join(save_directory, "xlora_classifier.safetensors"))
        elif is_main_process:
            state_dict = classifier.state_dict()
            torch.save(state_dict, os.path.join(save_directory, "xlora_classifier.pt"))
