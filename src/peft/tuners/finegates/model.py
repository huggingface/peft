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

from __future__ import annotations

import operator
from typing import Optional

from torch import nn

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTuner, get_device_map
from peft.utils import TRANSFORMERS_MODELS_TO_FINEGATES_TARGET_MODULES_MAPPING

from .config import FineGatesConfig
from .layer import FineGatesLayer, dispatch_default


class FineGatesModel(BaseTuner):
    """
    FineGates tuner implementation.

    FineGates learns row and column gates for frozen linear layers. It can be merged into the base model, which
    materializes structured zeros in the affected weights and biases.
    """

    prefix: str = "finegates_"
    tuner_layer_cls = FineGatesLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_FINEGATES_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        finegates_config: FineGatesConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key,
    ) -> None:
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        kwargs = {
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }
        try:
            kwargs["get_apply_tensor_subclass"] = operator.attrgetter(
                "hf_quantizer.quantization_config.get_apply_tensor_subclass"
            )(self.model)
        except AttributeError:
            pass

        if isinstance(target, FineGatesLayer):
            target.update_layer(adapter_name, config=finegates_config)
        else:
            device_map = get_device_map(self.model)
            new_module = self._create_new_module(
                finegates_config,
                adapter_name,
                target,
                device_map=device_map,
                **kwargs,
            )
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(finegates_config: FineGatesConfig, adapter_name: str, target: nn.Module, **kwargs):
        if isinstance(target, FineGatesLayer):
            return target

        dispatchers = []
        if is_bnb_available():
            # FineGates merge/unmerge relies on direct access to dense weights. Keep quantized support explicit for now.
            pass
        if is_bnb_4bit_available():
            pass
        dispatchers.append(dispatch_default)

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, finegates_config=finegates_config, **kwargs)
            if new_module is not None:
                break

        if new_module is None:
            target_type = type(target.get_base_layer()) if isinstance(target, FineGatesLayer) else type(target)
            raise ValueError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear` and `Conv1D` are "
                f"supported, got {target_type}."
            )

        return new_module

    def _unload_and_optionally_merge(self, *args, **kwargs):
        if getattr(self.model, "is_loaded_in_8bit", False):
            raise ValueError("Cannot merge FineGates layers when the model is loaded in 8-bit mode")

        if getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge FineGates layers when the model is loaded in 4-bit mode")

        return super()._unload_and_optionally_merge(*args, **kwargs)

    def _get_finegates_loss(self, adapter_names: Optional[list[str] | str] = None):
        if adapter_names is None:
            adapter_names = self.active_adapters
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        total_loss = 0.0
        num_layers = 0
        for module in self.model.modules():
            if not isinstance(module, FineGatesLayer):
                continue
            for adapter_name in adapter_names:
                if adapter_name not in module._available_adapters:
                    continue
                total_loss = total_loss + module.get_sparsity_loss(adapter_name)
                num_layers += 1

        if num_layers == 0:
            return 0.0
        return total_loss / num_layers

    def get_finegates_compression_stats(self, adapter_names: Optional[list[str] | str] = None) -> dict[str, object]:
        if adapter_names is None:
            adapter_names = self.active_adapters
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        layer_stats = {}
        totals = {"active_params": 0, "pruned_params": 0, "total_params": 0}
        for module_name, module in self.model.named_modules():
            if not isinstance(module, FineGatesLayer):
                continue
            for adapter_name in adapter_names:
                if adapter_name not in module._available_adapters:
                    continue
                stats = module.get_compression_statistics(adapter_name)
                layer_stats[f"{module_name}:{adapter_name}"] = stats
                totals["active_params"] += stats["active_params"]
                totals["pruned_params"] += stats["pruned_params"]
                totals["total_params"] += stats["active_params"] + stats["pruned_params"]

        if totals["total_params"] > 0:
            totals["param_sparsity"] = totals["pruned_params"] / totals["total_params"]
        else:
            totals["param_sparsity"] = 0.0

        return {"total": totals, "layers": layer_stats}
