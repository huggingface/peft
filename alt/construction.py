"""Function and classes that construct an adapter from a config"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterator

from torch import nn

from base import AdapterConfig, AdapterLayer
from ia3 import IA3Config, LinearIA3Layer
from lora import EmbeddingLoraLayer, LinearLoraLayer, LoraConfig


def _get_selection_strategy(config: AdapterConfig, base_model: nn.Module) -> Any:  # TODO
    if isinstance(config.target_modules, str):
        return _regex_selection_strategy(config, base_model)
    if isinstance(config.target_modules, list):
        return _list_match_selection_strategy(config, base_model)
    raise ValueError("TODO")


def _regex_selection_strategy(config: AdapterConfig, base_model: nn.Module) -> Iterator[tuple[str, Any]]:
    assert isinstance(config.target_modules, str)
    for name, _ in base_model.named_modules():
        if re.fullmatch(config.target_modules, name):
            yield name, None


def _list_match_selection_strategy(config: AdapterConfig, base_model: nn.Module) -> Iterator[tuple[str, Any]]:
    assert isinstance(config.target_modules, list)
    assert isinstance(config.target_modules[0], str)
    for name, _ in base_model.named_modules():
        if any(name.endswith(key) for key in config.target_modules):
            yield name, None


def _get_adaptation_strategy(
    config: AdapterConfig, layer_specific_args: Any | None
) -> Callable[[nn.Module, str], AdapterLayer]:
    # TODO: more complex strategies, e.g. allowing to provide user defined layers
    # as replacement layers, or even mixing things up, like:
    # - one is LoRA layer with r=8, another with r=16
    # - one is LoRA layer, another is IA³ layer
    if layer_specific_args is None:
        return _OneToOneMappingStrategy(config)
    raise ValueError("TODO")


class _OneToOneMappingStrategy:
    # TODO could be partial-ed function
    def __init__(self, config: AdapterConfig) -> None:
        self.config = config

    def __call__(self, base_model: nn.Module, name: str) -> AdapterLayer:
        layer = getattr(base_model, name)

        if isinstance(layer, nn.Linear):
            if isinstance(self.config, LoraConfig):
                return LinearLoraLayer.from_config(self.config, layer)
            if isinstance(self.config, IA3Config):
                return LinearIA3Layer.from_config(self.config, layer)

        if isinstance(layer, nn.Embedding):
            if isinstance(self.config, LoraConfig):
                return EmbeddingLoraLayer.from_config(self.config, layer)

        raise TypeError(f"Could not find a suitable adapter layer for {type(layer)} and config {type(self.config)}")
