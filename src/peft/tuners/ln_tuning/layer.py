# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from copy import deepcopy
from typing import List, Optional

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


class SelectLayer(nn.Module, BaseTunerLayer):
    """
    Selects a layer from the model.
    """

    adapter_layer_names = ("select_new_layers", )

    def __init__(self, base_layer: nn.Module, adapter_name: str):
        super().__init__()
        self.base_layer = base_layer
        self.select_new_layers = nn.ModuleDict({})
        self.update_layer(self.base_layer, adapter_name)
        # self.adapter_names = [adapter_name]
        self._active_adapter = adapter_name
        self.merged_adapters = []

    def update_layer(self, layer: nn.Module, adapter_name: str):
        self.select_new_layers[adapter_name] = deepcopy(layer)

    def enable_adapters(self, enabled: bool) -> None:
        """Toggle the enabling and disabling of adapters

        Takes care of setting the requires_grad flag for the adapter weights.

        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
        if enabled:
            self.set_adapter(self.active_adapters)
            self._disable_adapters = False
        else:
            if self.merged:
                self.unmerge()
            # disable grads on all adapter layers
            for layer_name in self.adapter_layer_names:
                layer = getattr(self, layer_name)
                layer.requires_grad_(False)
            self._disable_adapters = True

    def merge(self, adapter_names: Optional[List[str]] = None):
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        if len(adapter_names) > 1:
            raise ValueError(
                "You can only use one adapter for SelectLayer Adapter. "
                "Because SelectLayer selects one set of layers from the original arch."
            )
        merged_adapters = set(self.merged_adapters)
        if merged_adapters:
            warnings.warn("Already merged with {}. Unmerging first.".format(merged_adapters))
            self.unmerge()

        self.base_layer, self.select_new_layers[adapter_names[0]] = (
            self.select_new_layers[adapter_names[0]],
            self.base_layer,
        )
        self.merged_adapters.append(adapter_names[0])

    def unmerge(self):
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        merged_name = self.merged_adapters.pop(0)
        self.base_layer, self.select_new_layers[merged_name] = (
            self.select_new_layers[merged_name],
            self.base_layer,
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            if len(self.active_adapters) != 1:
                raise ValueError(
                    "You can only use one adapter for SelectLayer Adapter."
                    "Because SelectLayer selects one set of layers from the original arch."
                )
            active_adapter = self.active_adapters[0]
            result = self.select_new_layers[active_adapter](x, *args, **kwargs)

        return result
