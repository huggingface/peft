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
import importlib.metadata as importlib_metadata
from typing import Any, Optional

import packaging.version
import torch

from peft.import_utils import is_auto_awq_available
from peft.tuners.oft.layer import OFTLayer
from peft.tuners.tuners_utils import BaseTunerLayer


class AwqOFTLinear(torch.nn.Module, OFTLayer):
    def __init__(
        self,
        base_layer,
        adapter_name,
        r: int = 0,
        oft_block_size: int = 32,
        module_dropout: float = 0.0,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        init_weights: bool = True,
        **kwargs,
    ):
        super().__init__()
        OFTLayer.__init__(self, base_layer)

        # self.base_layer and self.quant_linear_module are the same; we need the former for consistency and the latter
        # for backwards compatibility
        self.quant_linear_module = base_layer

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            oft_block_size=oft_block_size,
            module_dropout=module_dropout,
            coft=coft,
            eps=eps,
            block_share=block_share,
            init_weights=init_weights,
        )

    def forward(self, x: torch.Tensor):
        # oft_rotation = torch.eye(self.bs, device=x.device, dtype=x.dtype).repeat(self.rank, 1, 1)

        if self.disable_adapters:
            result = self.quant_linear_module(x)
            return result

        for active_adapter in self.active_adapters:
            if active_adapter not in self.oft_r.keys():
                continue

            oft_r = self.oft_r[active_adapter]
            oft_block_size = self.oft_block_size[active_adapter]
            # oft_s = self.oft_s[active_adapter]
            # dropout = self.oft_dropout[active_adapter]

            rank = self.r[active_adapter]
            coft = self.coft[active_adapter]
            eps = self.eps[active_adapter]

            """
            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = x.dtype
                x = self._cast_input_dtype(x, oft_r.dtype)

            if coft:
                with torch.no_grad():
                    oft_r.copy_(self._project_batch(oft_r, eps=eps))
            orth_rotate = self._cayley_batch(oft_r, oft_block_size)
            # orth_rotate = dropout(orth_rotate)

            current_oft_rot_dtype = oft_rotation.dtype
            if orth_rotate.dtype != current_oft_rot_dtype:
                orth_rotate = orth_rotate.to(current_oft_rot_dtype)
            # oft_rotation = self.oft_matmul(orth_rotate, oft_rotation.unsqueeze(0)).squeeze(0)
            oft_rotation = torch.bmm(orth_rotate, oft_rotation)
            oft_rotation = oft_rotation.to(current_oft_rot_dtype)



        batch_dims = x.shape[:-1]
        x_reshaped = x.view(*batch_dims, rank, -1)
        x_rotated_reshaped = torch.einsum('...rk,rkc->...rc', x_reshaped, oft_rotation)
        # x_rotated_reshaped = torch.einsum('rkc,...rk->...rc', oft_rotation.transpose(-1, -2), x_reshaped)
        x_rotated = x_rotated_reshaped.reshape(*batch_dims, self.in_features)
        """
        x_rotated = oft_r(x)

        result = self.quant_linear_module(x_rotated)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "oft." + rep


def dispatch_awq(
    target: torch.nn.Module,
    adapter_name: str,
    **kwargs: Any,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if is_auto_awq_available():
        from awq.modules.linear import WQLinear_GEMM

        if isinstance(target_base_layer, WQLinear_GEMM):
            # Raise the error only at the dispatch level
            AUTOAWQ_MINIMUM_VERSION = packaging.version.parse("0.2.0")
            version_autoawq = packaging.version.parse(importlib_metadata.version("autoawq"))

            if AUTOAWQ_MINIMUM_VERSION > version_autoawq:
                raise ImportError(
                    f"Found an incompatible version of auto-awq. Found version {version_autoawq}, "
                    f"but only versions above {AUTOAWQ_MINIMUM_VERSION} are supported for PEFT."
                )

            new_module = AwqOFTLinear(target, adapter_name, **kwargs)
            target.qweight = target_base_layer.qweight

    return new_module
