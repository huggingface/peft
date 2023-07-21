from __future__ import annotations

from dataclasses import dataclass, field

import torch
from base import AdapterConfig, AdapterLayer
from torch import nn
from torch.nn import functional as F


@dataclass
class LoraConfig(AdapterConfig):
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})

    def __post_init__(self) -> None:
        if isinstance(self.target_modules, list):
            assert len(self.target_modules) >= 1
        assert self.r > 0


class LoraLayer(AdapterLayer):
    def __init__(self, base_module: nn.Module, r: int = 8) -> None:
        if r <= 0:
            raise ValueError("TODO")

        self.r = r
        super().__init__(base_module)

    @classmethod
    def from_config(cls, config: AdapterConfig, base_module: nn.Module) -> LoraLayer:
        r = getattr(config, "r", None)
        assert isinstance(r, int)
        return cls(base_module, r=r)

    def reset_device(self) -> None:
        self.to(self.base_module.weight.device)  # type: ignore

    def reset_requires_grad(self) -> None:
        self.base_module.requires_grad_(False)
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)

    def _post_forward(self, output, x):
        if self.merged or not self.active:
            return output

        lora_output = self.lora_B(self.lora_A(x))
        return output + lora_output

    def merge(self) -> None:
        if self.merged:
            return

        delta_weight = self.get_delta_weight()
        self.base_module.weight.data += delta_weight
        self.merged = True

    def unmerge(self) -> None:
        if not self.merged:
            return

        delta_weight = self.get_delta_weight()
        self.base_module.weight.data -= delta_weight
        self.merged = False

    def get_delta_weight(self) -> torch.Tensor:
        raise NotImplementedError


class LinearLoraLayer(LoraLayer):
    def reset_params(self) -> None:
        if not isinstance(self.base_module, nn.Linear):
            raise ValueError(f"{self.__class__.__name__} must be applied to an nn.Linear layer")

        self.lora_A = nn.Linear(self.base_module.in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, self.base_module.out_features, bias=False)
        nn.init.zeros_(self.lora_B.weight)

    def get_delta_weight(self) -> torch.Tensor:
        return self.lora_B.weight @ self.lora_A.weight  # type: ignore


class EmbeddingLoraLayer(LoraLayer):
    def reset_params(self) -> None:
        if not isinstance(self.base_module, nn.Embedding):
            raise ValueError(f"{self.__class__.__name__} must be applied to an nn.Embedding layer")

        self.lora_A = nn.Embedding(
            self.base_module.num_embeddings,
            self.r,
            padding_idx=self.base_module.padding_idx,
            max_norm=self.base_module.max_norm,
            norm_type=self.base_module.norm_type,
            scale_grad_by_freq=self.base_module.scale_grad_by_freq,
            sparse=self.base_module.sparse,
            device=self.base_module.weight.device,
            dtype=self.base_module.weight.dtype,
        )
        self.lora_B = nn.Linear(self.r, self.base_module.embedding_dim, bias=False)
        nn.init.zeros_(self.lora_B.weight)

    def get_delta_weight(self) -> torch.Tensor:
        return self.lora_A.weight @ self.lora_B.weight.T
