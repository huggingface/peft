# Copyright 2025-present the HuggingFace Inc. team.
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

import warnings
from typing import Any, Optional

import bitsandbytes as bnb
import torch

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_bnb_weight

from .config import RoadVariant
from .layer import RoadLayer, _apply_road, _get_delta_weight


if is_bnb_available():

    class Linear8bitLt(torch.nn.Module, RoadLayer):
        # Road implemented in a dense layer
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            variant: RoadVariant = "road_1",
            group_size: int = 64,
            init_weights: bool = True,
            **kwargs,
        ) -> None:
            super().__init__()
            RoadLayer.__init__(self, base_layer)

            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name,
                variant=variant,
                group_size=group_size,
                init_weights=init_weights,
            )

        def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
            """
            Merge the active adapter weights into the base weights

            Args:
                safe_merge (`bool`, *optional*):
                    If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                    before merging the weights. This is useful if you want to check if the merge operation will produce
                    NaNs. Defaults to `False`.
                adapter_names (`list[str]`, *optional*):
                    The list of adapter names that should be merged. If None, all active adapters will be merged.
                    Defaults to `None`.
            """
            adapter_names = check_adapters_to_merge(self, adapter_names)
            if not adapter_names:
                # no adapter to merge
                return

            for active_adapter in adapter_names:
                if active_adapter in self._available_adapters:
                    warnings.warn(
                        "Merge road module to 8-bit linear may get different generations due to rounding errors."
                    )

                    weight = self.get_base_layer().weight
                    state = self.get_base_layer().state
                    if state.SCB is None:
                        state.SCB = weight.SCB

                    # Dequantize the result of identity matrix and int8 weight because bitsandbytes does not support int8
                    # dequantization directly
                    output = dequantize_bnb_weight(weight, state=state)
                    road_R = _get_delta_weight(
                        self.variant[active_adapter],
                        self.group_size[active_adapter],
                        self.road_theta[active_adapter].data,
                        self.road_alpha[active_adapter].data,
                    )

                    w_data = torch.matmul(road_R, output.to(road_R.dtype))
                    w_data = w_data.to(road_R.dtype).to(road_R.device).contiguous()

                    if safe_merge and not torch.isfinite(w_data).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    self.get_base_layer().weight = bnb.nn.Int8Params(
                        w_data.to("cpu"), requires_grad=False, has_fp16_weights=weight.has_fp16_weights
                    ).to(weight.device)

                    if self.get_base_layer().bias is not None:
                        bias = self.get_base_layer().bias
                        orig_dtype = bias.dtype
                        bias_data = bias.data
                        new_bias = torch.matmul(road_R, bias_data.to(road_R.dtype))
                        bias.data = new_bias.to(orig_dtype)

                    state.reset_grads()
                    self.merged_adapters.append(active_adapter)

        def unmerge(self) -> None:
            """
            This method unmerges all merged adapter layers from the base weights.
            """
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return

            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter in self._available_adapters:
                    warnings.warn(
                        "Unmerge road module to 8-bit linear may get different generations due to rounding errors."
                    )

                    weight = self.get_base_layer().weight
                    state = self.get_base_layer().state
                    if state.SCB is None:
                        state.SCB = weight.SCB
                    output = dequantize_bnb_weight(weight, state=state)

                    road_R = _get_delta_weight(
                        self.variant[active_adapter],
                        self.group_size[active_adapter],
                        self.road_theta[active_adapter].data,
                        self.road_alpha[active_adapter].data,
                    )
                    inv_road_R = torch.linalg.inv(road_R.to(torch.float32)).to(road_R.dtype)

                    w_data = torch.matmul(inv_road_R, output.to(road_R.dtype))
                    w_data = w_data.to(road_R.dtype).to(road_R.device).contiguous()

                    self.get_base_layer().weight = bnb.nn.Int8Params(
                        w_data.to("cpu"), requires_grad=False, has_fp16_weights=weight.has_fp16_weights
                    ).to(weight.device)

                    if self.get_base_layer().bias is not None:
                        bias = self.get_base_layer().bias
                        orig_dtype = bias.dtype
                        bias_data = bias.data
                        new_bias = torch.matmul(inv_road_R, bias_data)
                        bias.data = new_bias.to(orig_dtype)

                    state.reset_grads()

        def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs)
            else:
                result = self.base_layer(x, *args, **kwargs)

                for active_adapter in self.active_adapters:
                    if active_adapter not in self._available_adapters:
                        continue

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        result = self._cast_input_dtype(result, self.road_theta[active_adapter].dtype)

                    result = _apply_road(
                        self.variant[active_adapter],
                        self.group_size[active_adapter],
                        self.road_theta[active_adapter],
                        self.road_alpha[active_adapter],
                        result,
                    )

                    if requires_conversion:
                        x = x.to(expected_dtype)

            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "road." + rep

    def dispatch_bnb_8bit(target: torch.nn.Module, adapter_name: str, **kwargs):
        new_module = None

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        loaded_in_8bit = kwargs.get("loaded_in_8bit", False)
        if loaded_in_8bit and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(target, adapter_name, **eightbit_kwargs)

        return new_module


if is_bnb_4bit_available():

    class Linear4bit(torch.nn.Module, RoadLayer):
        # OFT implemented in a dense layer
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            variant: RoadVariant = "road_1",
            group_size: int = 64,
            init_weights: bool = True,
            **kwargs,
        ) -> None:
            super().__init__()
            RoadLayer.__init__(self, base_layer)

            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name,
                variant=variant,
                group_size=group_size,
                init_weights=init_weights,
            )

        def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
            """
            Merge the active adapter weights into the base weights

            Args:
                safe_merge (`bool`, *optional*):
                    If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                    before merging the weights. This is useful if you want to check if the merge operation will produce
                    NaNs. Defaults to `False`.
                adapter_names (`list[str]`, *optional*):
                    The list of adapter names that should be merged. If None, all active adapters will be merged.
                    Defaults to `None`.
            """
            adapter_names = check_adapters_to_merge(self, adapter_names)
            if not adapter_names:
                # no adapter to merge
                return

            for active_adapter in adapter_names:
                if active_adapter in self._available_adapters:
                    warnings.warn(
                        "Merge oft module to 4-bit linear may get different generations due to rounding errors."
                    )
                    # Refer to https://gist.github.com/ChrisHayduk/1a53463331f52dca205e55982baf9930
                    weight = self.get_base_layer().weight
                    kwargs = weight.__dict__

                    output = dequantize_bnb_weight(weight, state=weight.quant_state)

                    road_R = _get_delta_weight(
                        self.variant[active_adapter],
                        self.group_size[active_adapter],
                        self.road_theta[active_adapter].data,
                        self.road_alpha[active_adapter].data,
                    )
                    w_data = torch.matmul(road_R, output.to(road_R.dtype))
                    w_data = w_data.to(road_R.dtype).to(road_R.device)

                    if safe_merge and not torch.isfinite(w_data).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    if "bnb_quantized" in kwargs:
                        kwargs["bnb_quantized"] = False
                    kwargs["requires_grad"] = False
                    kwargs.pop("data", None)
                    # torch.compile can introduce attributes preceded by '_', remove them
                    kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
                    self.get_base_layer().weight = bnb.nn.Params4bit(w_data.to("cpu"), **kwargs).to(weight.device)

                    if self.get_base_layer().bias is not None:
                        bias = self.get_base_layer().bias
                        orig_dtype = bias.dtype
                        bias_data = bias.data
                        new_bias = torch.matmul(road_R, bias_data.to(road_R.dtype))
                        bias.data = new_bias.to(orig_dtype)

                    self.merged_adapters.append(active_adapter)

        def unmerge(self) -> None:
            """
            This method unmerges all merged adapter layers from the base weights.
            """
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return

            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter in self._available_adapters:
                    warnings.warn(
                        "Unmerge oft module to 4-bit linear may get different generations due to rounding errors."
                    )

                    weight = self.get_base_layer().weight
                    kwargs = weight.__dict__
                    output = dequantize_bnb_weight(weight, state=weight.quant_state)

                    road_R = _get_delta_weight(
                        self.variant[active_adapter],
                        self.group_size[active_adapter],
                        self.road_theta[active_adapter].data,
                        self.road_alpha[active_adapter].data,
                    )
                    inv_road_R = torch.linalg.inv(road_R.to(torch.float32)).to(road_R.dtype)

                    w_data = torch.matmul(inv_road_R, output.to(road_R.dtype))
                    w_data = w_data.to(road_R.dtype).to(road_R.device)

                    if "bnb_quantized" in kwargs:
                        kwargs["bnb_quantized"] = False
                    kwargs["requires_grad"] = False
                    kwargs.pop("data", None)
                    self.get_base_layer().weight = bnb.nn.Params4bit(w_data.to("cpu"), **kwargs).to(weight.device)

                    if self.get_base_layer().bias is not None:
                        bias = self.get_base_layer().bias
                        orig_dtype = bias.dtype
                        bias_data = bias.data
                        new_bias = torch.matmul(inv_road_R, bias_data)
                        bias.data = new_bias.to(orig_dtype)

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs)
            else:
                result = self.base_layer(x, *args, **kwargs)
                # As per Tim Dettmers, for 4bit, we need to defensively clone here.
                # The reason is that in some cases, an error can occur that backprop
                # does not work on a manipulated view. This issue may be solved with
                # newer PyTorch versions but this would need extensive testing to be
                # sure.
                # result = result.clone()

                for active_adapter in self.active_adapters:
                    if active_adapter not in self._available_adapters:
                        continue

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        result = self._cast_input_dtype(result, self.road_theta[active_adapter].dtype)

                    result = _apply_road(
                        self.variant[active_adapter],
                        self.group_size[active_adapter],
                        self.road_theta[active_adapter],
                        self.road_alpha[active_adapter],
                        result,
                    )
                    if requires_conversion:
                        x = x.to(expected_dtype)

            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "oft." + rep

    def dispatch_bnb_4bit(target: torch.nn.Module, adapter_name: str, **kwargs):
        new_module = None

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        loaded_in_4bit = kwargs.get("loaded_in_4bit", False)
        if loaded_in_4bit and is_bnb_4bit_available() and isinstance(target_base_layer, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target_base_layer.compute_dtype,
                    "compress_statistics": target_base_layer.weight.compress_statistics,
                    "quant_type": target_base_layer.weight.quant_type,
                }
            )
            new_module = Linear4bit(target, adapter_name, **fourbit_kwargs)

        return new_module
