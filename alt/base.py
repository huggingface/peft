from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from torch import nn


@dataclass
class AdapterConfig:
    target_modules: str | list[str] | None = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )

    modules_to_save: list[str] = field(
        default_factory=list,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )


class AdapterLayer(nn.Module):
    def __init__(self, base_module: nn.Module) -> None:
        super().__init__()
        self.base_module = base_module  # TODO rename to base_layer

        self.active: bool = True
        self.merged: bool = False
        self.reset_params()
        self.reset_device()

    @classmethod
    def from_config(cls, config: AdapterConfig, base_module: nn.Module) -> AdapterLayer:
        raise NotImplementedError

    def set_active(self, active: bool) -> None:
        self.active = active

    def reset_params(self) -> None:
        raise NotImplementedError

    def reset_device(self) -> None:
        raise NotImplementedError

    def reset_requires_grad(self) -> None:
        raise NotImplementedError

    def merge(self) -> None:
        raise NotImplementedError

    def unmerge(self) -> None:
        raise NotImplementedError

    def _pre_forward(self, *args, **kwargs):
        return args, kwargs

    def forward(self, *args, **kwargs) -> Any:
        args, kwargs = self._pre_forward(*args, **kwargs)
        output = self.base_module(*args, **kwargs)
        return self._post_forward(output, *args, **kwargs)

    def _post_forward(self, output, *args, **kwargs):
        return output


class ModulesToSaveWrapper(AdapterLayer):
    def reset_params(self) -> None:
        self.new_module = copy.deepcopy(self.base_module)

    def reset_device(self) -> None:
        pass

    def reset_requires_grad(self) -> None:
        self.base_module.requires_grad_(False)
        self.new_module.requires_grad_(True)

    def merge(self) -> None:
        if self.merged:
            return

        self.base_module, self.new_module = self.new_module, self.base_module
        self.merged = True

    def unmerge(self) -> None:
        if not self.merged:
            return

        self.base_module, self.new_module = self.new_module, self.base_module
        self.merged = False

    def forward(self, *args, **kwargs) -> Any:
        args, kwargs = self._pre_forward(*args, **kwargs)

        if self.active is self.merged:
            output = self.base_module(*args, **kwargs)
        else:
            output = self.new_module(*args, **kwargs)

        return self._post_forward(output, *args, **kwargs)
