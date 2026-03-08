# Copyright 2026-present the HuggingFace Inc. team.
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

from typing import Optional

from torch import nn

from peft.tuners.tuners_utils import BaseTuner, get_device_map
from peft.utils import TRANSFORMERS_MODELS_TO_PSOFT_TARGET_MODULES_MAPPING

from .config import PsoftConfig
from .layer import PsoftLayer, dispatch_default


class PsoftModel(BaseTuner):
    """
    PSOFT (Efficient Orthogonal Fine-Tuning with Principal Subspace Adaptation) model.

    Inserts an r*r orthogonal (or scaled) transformation R between low-rank A and B: ΔW = B @ (R-I) @ A. Use
    ab_svd_init="psoft_init" to initialize A/B from SVD and freeze them, training only R (and optional magnitude
    vectors).

    Args:
        model: The model to adapt.
        config: PsoftConfig.
        adapter_name: Adapter name, default "default".
        low_cpu_mem_usage: Create empty adapter weights on meta device.
    """

    prefix: str = "psoft_"
    tuner_layer_cls = PsoftLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_PSOFT_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        peft_config: PsoftConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        *,
        parameter_name: Optional[str] = None,
    ) -> None:
        if current_key is None:
            raise ValueError("Current key must not be None.")

        kwargs = {
            "target_name": current_key,
            "parameter_name": parameter_name,
        }

        if isinstance(target, PsoftLayer):
            target.update_layer(adapter_name, config=peft_config, **kwargs)
            return

        device_map = get_device_map(self.model)
        new_module = self._create_new_module(peft_config, adapter_name, target, device_map=device_map, **kwargs)

        if adapter_name not in self.active_adapters:
            new_module.requires_grad_(False)
        self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(psoft_config: PsoftConfig, adapter_name: str, target: nn.Module, **kwargs) -> nn.Module:
        new_module = dispatch_default(target, adapter_name, config=psoft_config, **kwargs)
        if new_module is None:
            raise ValueError(
                f"Target module {target} is not supported by minimal PSOFT. Only torch.nn.Linear is supported."
            )
        return new_module
