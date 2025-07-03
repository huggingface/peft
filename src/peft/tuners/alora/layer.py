from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn

from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.lora.layer import dispatch_default as lora_dispatch_default
from peft.tuners.lora.variants import ALoraLinearVariant

from .config import aLoraConfig


class aLoraLayer(LoraLayer):
    def resolve_lora_variant(self, *, use_dora: bool, **kwargs) -> Optional[object]:
        return ALoraLinearVariant()

    def _mixed_batch_forward(
        self,
        x: torch.Tensor,
        *args,
        adapter_names: list[str],
        alora_offsets: list[int],
        **kwargs,
    ) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        ks = alora_offsets
        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
            if len(ks) > 1:
                ks_batch = ks[sub_batch_indices_list[i]]
                for j in range(len(ks_batch)):
                    k = min(ks_batch[j], result.shape[1])
                    lora_output = lora_B(lora_A(dropout(sub_batch[j, -k:, :]))) * scaling
                    result[sub_batch_indices_list[i][j], -k:, :] += lora_output.to(torch_result_dtype)
            else:
                ks_batch = ks
                k = min(result.shape[1], ks_batch[0])
                lora_output = lora_B(lora_A(dropout(sub_batch[:, -k:, :]))) * scaling
                result[sub_batch_indices_list[i], -k:, :] += lora_output.to(torch_result_dtype)

        return result


class Linear(LoraLinear, aLoraLayer):
    pass


def dispatch_default(target: nn.Module, adapter_name: str, lora_config: aLoraConfig, **kwargs) -> Optional[nn.Module]:
    if isinstance(target, LoraLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, nn.Linear):
        if kwargs.get("fan_in_fan_out", False):
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        return Linear(target, adapter_name, **kwargs)

    return lora_dispatch_default(target, adapter_name, lora_config, **kwargs)
