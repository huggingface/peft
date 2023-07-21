from __future__ import annotations

from dataclasses import dataclass

import torch
from base import AdapterConfig, AdapterLayer
from torch import nn


@dataclass
class IA3Config(AdapterConfig):
    pass


class LinearIA3Layer(AdapterLayer):
    @classmethod
    def from_config(cls, config: AdapterConfig, base_module: nn.Module) -> LinearIA3Layer:
        return cls(base_module)

    def reset_device(self) -> None:
        self.to(self.base_module.weight.device)  # type: ignore

    def reset_params(self) -> None:
        if not isinstance(self.base_module, nn.Linear):
            raise ValueError(f"{self.__class__.__name__} must be applied to an nn.Linear layer")
        self.ia3_weight = nn.Parameter(torch.ones_like(self.base_module.weight[:1]))

    def reset_requires_grad(self) -> None:
        self.base_module.requires_grad_(False)
        self.ia3_weight.requires_grad_(True)

    def _pre_forward(self, X, *args, **kwargs):
        if self.merged or not self.active:
            return (X,) + args, kwargs

        return (X * self.ia3_weight,) + args, kwargs

    def merge(self) -> None:
        if self.merged:
            return

        self.base_module.weight.data *= self.ia3_weight
        #self.base_module.weight.data = torch.mul(self.base_module.weight.data, self.ia3_weight.data)
        self.merged = True

    def unmerge(self) -> None:
        if not self.merged:
            return

        self.base_module.weight.data /= self.ia3_weight
        #self.base_module.weight.data = torch.div(self.base_module.weight.data, self.ia3_weight.data + 1e-8)
        self.merged = False
