# todo
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers.pytorch_utils import Conv1D

import loralib as lora
from loralib import mark_only_lora_as_trainable

from ..utils import PETConfig


@dataclass
class LoRAConfig(PETConfig):
    """
    This is the configuration class to store the configuration of a :class:`~pet.LoRA`.

    Args:
        r: (:obj:`init`): LoRA attention dimension
        target_modules (:obj: list of :obj: str): The names of the modules to apply LoRA to.
        lora_alpha (:obj: float): The alpha parameter for LoRA scaling.
        lora_dropout (:obj: float): The dropout probability for LoRA layers.
        merge_weights (:obj: bool):
            Whether to merge the weights of the LoRA layers with the base transformer model in `eval` mode.
        fan_in_fan_out (:obj: bool): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora (:obj: list of :obj: bool): Used with `lora.MergedLinear`.
        bias (:obj: str): Bias type for LoRA. Can be 'none', 'all' or 'lora_only'
    """

    r: int = field(default=8, metadata={"help": "LoRA attention dimension"})
    target_modules: Optional[list] = field(default=None, metadata={"help": "List of modules to replace with LoRA"})
    lora_alpha: int = field(default=None, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "LoRA dropout"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the LoRA model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_lora: Optional[list[bool]] = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})
    bias: str = field(default="none", metadata={"help": "Bias type for LoRA. Can be 'none', 'all' or 'lora_only'"})


class LoRAModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (LoRA) model from a pretrained transformers model.

    Args:
        model (:obj:`transformers.PreTrainedModel`): The model to be adapted.
        config (:obj:`LoRAConfig`): The configuration of the LoRA model.

    Returns:
        :obj:`torch.nn.Module`: The LoRA model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoRAConfig
        >>> from pet import LoRAModel, LoRAConfig
        >>> config = LoRAConfig(
            pet_type="LORA",
            task_type="SEQ_2_SEQ_LM",
            r=8,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.01,
            )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> lora_model = LoRAModel(config, model)

    Attributes:
        model (:obj:`transformers.PreTrainedModel`): The model to be adapted.
        config (:obj:`LoRAConfig`): The configuration of the LoRA model.
    """

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.config.bias)

    def _find_and_replace(self):
        kwargs = {
            "r": self.config.r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "fan_in_fan_out": self.config.fan_in_fan_out,
            "merge_weights": self.config.merge_weights,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if any(key.endswith(target_key) for target_key in self.config.target_modules):
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                if isinstance(target, torch.nn.Linear):
                    new_module = lora.Linear(target.in_features, target.out_features, bias=bias, **kwargs)
                elif isinstance(target, Conv1D):
                    kwargs.update({"enable_lora": self.config.enable_lora})
                    in_features, out_features = target.weight.shape
                    new_module = lora.MergedLinear(in_features, out_features, bias=bias, **kwargs)
                self._replace_module(parent, target_name, new_module, target)

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
