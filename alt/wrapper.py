from __future__ import annotations

from typing import Iterator, Type

import torch
from base import AdapterConfig, AdapterLayer, ModulesToSaveWrapper
from construction import _get_adaptation_strategy, _get_selection_strategy
from torch import nn


class Adapter:
    def __init__(self, model: nn.Module, name: str):
        self.model = model
        self.name = name
        self._adapter_layers: dict[str, AdapterLayer] = {}

    def _apply(self) -> None:
        for name, new_layer in self._adapter_layers.items():
            self._replace_layer(self.model, name, new_layer)
        self._reset_requires_grad()

    def _unapply(self) -> None:
        for name, layer in self._adapter_layers.items():
            self._replace_layer(self.model, name, layer.base_module)

    def _replace_layer(self, base_model: nn.Module, name: str, new_layer: nn.Module) -> None:
        # TODO: deal with ModuleList, ModuleDict
        old_layer = getattr(base_model, name, None)
        if old_layer is None:
            raise AttributeError("TODO")

        if old_layer is new_layer:
            return

        if isinstance(old_layer, AdapterLayer) and isinstance(new_layer, AdapterLayer):
            raise ValueError("Trying to replace an adapter layer with another adapter layer")

        if getattr(base_model, name) is not new_layer:
            setattr(base_model, name, new_layer)

    def _reset_requires_grad(self) -> None:
        for layer in self._adapter_layers.values():
            layer.reset_requires_grad()

    def _merge(self) -> None:
        for layer in self._adapter_layers.values():
            layer.merge()

    def _unmerge(self) -> None:
        for layer in self._adapter_layers.values():
            layer.unmerge()

    def _add_layer(self, name: str, layer: AdapterLayer | Type[AdapterLayer]) -> None:
        if not isinstance(layer, nn.Module):
            # layer is not (fully) iniitalized, initialize it here
            if not hasattr(self.model, name):
                raise AttributeError("TODO")
            old_layer = getattr(self.model, name)
            layer = layer(old_layer)

        self._adapter_layers[name] = layer
        self._replace_layer(self.model, name, layer)

    def _delete_layer(self, name: str) -> None:
        existing_names = self._adapter_layers.keys()
        if name not in existing_names:
            raise KeyError(f"Adapter layer {name} does not exist, existing layers: {existing_names}")

        adapter_layer = self._adapter_layers.pop(name)
        assert isinstance(adapter_layer, AdapterLayer)
        self._replace_layer(self.model, name, adapter_layer.base_module)

    def named_adapter_parameters(self) -> Iterator[tuple[str, torch.Tensor]]:
        for layer in self._adapter_layers.values():
            yield from layer.named_parameters()

    def adapter_parameters(self) -> Iterator[torch.Tensor]:
        for _, param in self.named_adapter_parameters():
            yield param


class AdapterWrapper(nn.Module):
    ##################
    # INITIALIZATION #
    ##################

    def __init__(self, model: nn.Module, config: AdapterConfig | None = None) -> None:
        super().__init__()
        self.model = model
        self.config = config

        self._active_adapter_name: str | None = None
        self._adapter_registry: dict[str, Adapter] = {}

    @property
    def active_adapter(self) -> Adapter | None:
        # this can return None if no adapter is active yet
        if self._active_adapter_name is None:
            return None
        return self._adapter_registry.get(self._active_adapter_name)

    def reset_base_model(self) -> None:
        self.model.requires_grad_(False)

        if self.active_adapter is not None:
            self.unmerge_adapter()
            self.active_adapter._unapply()

    #####################
    # HANDLING ADAPTERS #
    #####################

    def add_adapter(self, adapter_name: str = "default") -> None:
        if adapter_name in self._adapter_registry:
            raise KeyError("TODO")

        self.reset_base_model()
        self._active_adapter_name = adapter_name
        adapter = Adapter(self.model, name=adapter_name)
        self._adapter_registry[adapter_name] = adapter
        adapter._apply()

    def set_adapter(self, adapter_name: str) -> None:
        if adapter_name == self._active_adapter_name:
            return

        if adapter_name not in self._adapter_registry:
            raise KeyError("TODO")

        self.reset_base_model()
        self._active_adapter_name = adapter_name
        if self.active_adapter is not None:
            # note: should never be None at this point but mypy doesn't understand that
            self.active_adapter._apply()

    def delete_adapter(self, adapter_name: str) -> None:
        if adapter_name not in self._adapter_registry:
            raise KeyError("TODO")

        if len(self._adapter_registry) == 1:
            # TODO: probably not needed, should work without adapter
            raise ValueError("Cannot delete last adapter")

        is_active_adapter = adapter_name == self._active_adapter_name
        if is_active_adapter:
            self.unmerge_adapter()
            # we know there must be a key != currently active adapter because of the check above
            guess_next_adapter = next(k for k in self._adapter_registry if k != adapter_name)
            self.set_adapter(guess_next_adapter)

        del self._adapter_registry[adapter_name]

    def add_adapter_layer(self, name: str, layer: AdapterLayer | Type[AdapterLayer]) -> None:
        """Add an adapter layer to the active adapter."""
        if self.active_adapter is None:
            raise ValueError("There is no active adapter, create it first by calling .add_adapter")

        self.active_adapter._add_layer(name, layer)
        self.active_adapter._apply()

    def delete_adapter_layer(self, name: str) -> None:
        """Delete an adapter layer from the active adapter."""
        if self.active_adapter is None:
            raise ValueError("There is no active adapter")

        self.active_adapter._delete_layer(name)

    def _unload_and_optionally_merge(self, merge: bool) -> nn.Module:
        adapter = self.active_adapter
        if adapter is None:
            return self.model

        if merge:
            adapter._merge()
        else:
            adapter._unmerge()

        # remove the adapter layers
        adapter._unapply()
        return self.model

    def merge_adapter(self) -> None:
        if self.active_adapter is not None:
            self.active_adapter._merge()

    def unmerge_adapter(self) -> None:
        if self.active_adapter is not None:
            self.active_adapter._unmerge()

    def unload(self) -> nn.Module:
        return self._unload_and_optionally_merge(merge=False)

    def merge_and_unload(self) -> nn.Module:
        return self._unload_and_optionally_merge(merge=True)

    def activate_adapter(self) -> None:
        # TODO
        pass

    def deactivate_adapter(self) -> None:
        # TODO
        pass

    ########################
    # CREATION FROM CONFIG #
    ########################

    def add_adapter_from_config(self, config: AdapterConfig, adapter_name: str = "default") -> AdapterWrapper:
        self.add_adapter(adapter_name)

        selection_strategy = _get_selection_strategy(config, self.model)
        any_match = False
        for name, layer_specific_args in selection_strategy:
            any_match = True
            adaptation_strategy = _get_adaptation_strategy(config, layer_specific_args)
            new_layer = adaptation_strategy(self.model, name)
            self.add_adapter_layer(name, new_layer)

        if not any_match:
            raise ValueError("Could not find any matching layers for the given config")

        modules_to_save = config.modules_to_save or []
        for name in modules_to_save:
            self.add_adapter_layer(name, ModulesToSaveWrapper)

        return self

    @classmethod
    def from_config(cls, model: nn.Module, config: AdapterConfig, adapter_name: str = "default") -> AdapterWrapper:
        wrapper = cls(model, config=config)
        wrapper.add_adapter_from_config(config, adapter_name)
        return wrapper

    ##################################
    # EXPOSING METHODS OF BASE MODEL #
    ##################################

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()
