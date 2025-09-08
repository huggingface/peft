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

import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .config import RoadConfig, RoadVariant


class RoadLayer(BaseTunerLayer):
    """
    Road layer.

    Generally the idea of RoAD is to split the input vector into many 2D vectors and rotate each 2D vector with its own
    2D rotation matrix. For additional flexibility, each rotation matrix is multiplied by a trainable scale.

    when applied to vector R @ x each pair of elements of x is transformed like this: `y₀ = x₀ * α * cosθ - xₙ * α *
    sinθ` and `yₙ = x₀ * α * sinθ + xₙ * α * cosθ`

    The scales α and angles θ are learned for each pair of elements and, moreover, each of the 4 instances in the
    rotation matrix may actually be different (when using variant 2 or 4).

    Note that instead of using two consecutive elements x₀ x₁ we first split the whole vector into groups and pair
    elements from the first with the second half of the same group, which allows for more efficient inference
    implementation.

    The adapter needs to only store the angles θ and scales α, rather than the full matrix R and the inference
    implementation only needs to do elementwise vector multiplications.

    For merging the weights, we make use of the following formula: R @ (W @ x + b) = (R @ W) @ x + R @ b. The lhs part
    is how it is used in unmerged state (using efficient elementwise implementation instead of matrix multiplication)
    and the rhs part is how it is used in merged state where (R @ W) becomes the new weight matrix and R @ b becomes
    the new bias.

    """

    adapter_layer_names: tuple[str, ...] = ("road_theta", "road_alpha")
    other_param_names: tuple[str, ...] = ("variant", "group_size")

    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.variant = {}
        self.group_size = {}
        self.road_theta = nn.ParameterDict({})
        self.road_alpha = nn.ParameterDict({})

        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type '{type(base_layer)}' encountered, cannot apply RoAd adapter.")
        self.in_features = in_features
        self.out_features = out_features

    @property
    def _available_adapters(self) -> set[str]:
        return {*self.road_theta}

    def update_layer(
        self,
        adapter_name,
        variant,
        group_size,
        init_weights,
        inference_mode: bool = False,
    ):
        self.variant[adapter_name] = variant
        self.group_size[adapter_name] = group_size

        if self.out_features % group_size != 0:
            raise ValueError(
                f"The out_features of the base layer must be divisible by group_size ({group_size}) when using RoadLayer."
            )

        # Actual trainable parameters
        if variant == "road_1":
            size = self.out_features // 2
        elif variant == "road_2":
            size = self.out_features
        elif variant == "road_4":
            size = self.out_features * 2
        else:
            raise ValueError(
                f"Unsupported variant {variant} for RoadLayer. Supported variants are road_1, road_2, and road_4."
            )
        self.road_theta[adapter_name] = nn.Parameter(torch.empty(size))
        self.road_alpha[adapter_name] = nn.Parameter(torch.empty(size))

        self.reset_parameters(adapter_name, init_weights)
        self._move_adapter_to_device_of_base_layer(adapter_name)

        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def reset_parameters(self, adapter_name, init_weights):
        if init_weights is False:
            nn.init.normal_(self.road_theta[adapter_name].data, mean=0.0, std=0.5)
            nn.init.normal_(self.road_alpha[adapter_name].data, mean=1.0, std=0.5)
            return
        nn.init.zeros_(self.road_theta[adapter_name].data)
        nn.init.ones_(self.road_alpha[adapter_name].data)


class Linear(nn.Module, RoadLayer):
    # Road implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        variant: RoadVariant = "road_1",
        group_size: int = 64,
        init_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        RoadLayer.__init__(self, base_layer, **kwargs)

        self._active_adapter = adapter_name

        self.update_layer(
            adapter_name,
            variant,
            group_size,
            init_weights=init_weights,
        )

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self._available_adapters:
                    continue

                result = self._cast_input_dtype(result, self.road_theta[active_adapter].dtype)
                result = _apply_road(
                    self.variant[active_adapter],
                    self.group_size[active_adapter],
                    self.road_theta[active_adapter],
                    self.road_alpha[active_adapter],
                    result,
                )

            result = result.to(torch_result_dtype)

        return result

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self._available_adapters:
                continue

            dtype = self.road_theta[active_adapter].data.dtype

            # getting the sub-batch, passing it to Road layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = result[sub_batch_indices_list[i]].to(dtype)
            result[sub_batch_indices_list[i]] = _apply_road(
                self.variant[active_adapter],
                self.group_size[active_adapter],
                self.road_theta[active_adapter],
                self.road_alpha[active_adapter],
                sub_batch,
            )

        return result

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If `True`, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If `None`, all active adapters will be merged.
                Defaults to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self._available_adapters:
                base_layer = self.get_base_layer()
                orig_dtype = base_layer.weight.dtype
                road_R = _get_delta_weight(
                    self.variant[active_adapter],
                    self.group_size[active_adapter],
                    self.road_theta[active_adapter].data,
                    self.road_alpha[active_adapter].data,
                )
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weight = base_layer.weight.data.clone()
                    orig_weight = torch.matmul(road_R.to(orig_dtype), orig_weight)

                    if not torch.isfinite(orig_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weight.contiguous().to(orig_dtype)

                    if base_layer.bias is not None:
                        orig_bias = base_layer.bias.clone()
                        orig_bias = torch.matmul(road_R.to(orig_dtype), orig_bias)

                        if not torch.isfinite(orig_bias).all():
                            raise ValueError(
                                f"NaNs detected in the merged bias. The adapter {active_adapter} seems to be broken"
                            )

                        base_layer.bias.data = orig_bias.contiguous().to(orig_dtype)
                else:
                    orig_weight = base_layer.weight.data
                    orig_weight = torch.matmul(road_R.to(orig_dtype), orig_weight)
                    base_layer.weight.data = orig_weight.contiguous().to(orig_dtype)

                    if base_layer.bias is not None:
                        orig_bias = base_layer.bias.data
                        orig_bias = torch.matmul(road_R.to(orig_dtype), orig_bias)
                        base_layer.bias.data = orig_bias.contiguous().to(orig_dtype)

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            # Going in reverse order
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self._available_adapters:
                weight = self.get_base_layer().weight
                orig_dtype = weight.dtype
                road_R = _get_delta_weight(
                    self.variant[active_adapter],
                    self.group_size[active_adapter],
                    self.road_theta[active_adapter].data,
                    self.road_alpha[active_adapter].data,
                )
                # Since our matrix are not necessarily orthogonal we need inverse instead of transpose.
                # In practice we expect this to basically always work since we start from block diagonal rotation matrix.
                inv_road_R = torch.linalg.inv(road_R.to(torch.float32)).to(orig_dtype)
                orig_weight = torch.matmul(inv_road_R, weight.data)
                weight.data = orig_weight.contiguous()

                if self.get_base_layer().bias is not None:
                    orig_bias = torch.matmul(inv_road_R, self.get_base_layer().bias.data)
                    self.get_base_layer().bias.data = orig_bias.contiguous()

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "road." + rep


def _get_delta_weight(variant: RoadVariant, group_size: int, road_theta: torch.Tensor, road_alpha: torch.Tensor):
    first_col, second_col = _prepare_cols(variant, group_size, road_theta, road_alpha)

    # To help understand the logic below consider how rope embeddings work
    # here it is similar, but done in groups.
    # https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509/3

    # First column is simply put on the main diagonal
    output_tensor = torch.diag(first_col)
    # For second column we need to swap each half groups and add minus sign
    size = second_col.shape[0]
    swapped_second_col = second_col.reshape(-1, 2, group_size // 2)[:, [1, 0], :].flatten()
    rotated_diag_second_col = torch.diag(swapped_second_col).reshape(-1, 2, group_size // 2, size)[:, [1, 0], :, :]
    rotated_diag_second_col[:, 0, :, :] *= -1
    rotated_diag_second_col = rotated_diag_second_col.reshape(size, size)
    output_tensor += rotated_diag_second_col

    return output_tensor


def _prepare_cols(
    variant: RoadVariant, group_size: int, road_theta: torch.Tensor, road_alpha: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # In inference mode, this can be cached
    if variant == "road_1":
        # In each group there are only group_size // 2 parameters that are reused
        road_theta = road_theta.reshape(-1, group_size // 2).repeat_interleave(2, dim=0).flatten()
        road_alpha = road_alpha.reshape(-1, group_size // 2).repeat_interleave(2, dim=0).flatten()

        theta_cos = road_theta.cos()
        theta_sin = road_theta.sin()

        first_col = road_alpha * theta_cos
        second_col = road_alpha * theta_sin
    elif variant == "road_2":
        # Each group has exactly group_size parameters
        theta_cos = road_theta.cos()
        theta_sin = road_theta.sin()

        first_col = road_alpha * theta_cos
        second_col = road_alpha * theta_sin
    elif variant == "road_4":
        # Each group has 2*group_size parameters, first half used for first column, second half for second column
        road_theta = road_theta.reshape(-1, 2, group_size)
        theta_cos = road_theta[:, 0, :].cos().flatten()
        theta_sin = road_theta[:, 1, :].sin().flatten()
        road_alpha = road_alpha.reshape(-1, 2, group_size)
        alpha_1 = road_alpha[:, 0, :].flatten()
        alpha_2 = road_alpha[:, 1, :].flatten()

        first_col = alpha_1 * theta_cos
        second_col = alpha_2 * theta_sin
    else:
        raise ValueError(
            f"Unsupported variant {variant} for RoadLayer. Supported variants are road_1, road_2, and road_4."
        )

    return first_col, second_col


def _apply_road(
    variant: RoadVariant, group_size: int, road_theta: torch.Tensor, road_alpha: torch.Tensor, x: torch.Tensor
):
    first_col, second_col = _prepare_cols(variant, group_size, road_theta, road_alpha)

    # Split in half groups and join back
    # See equation 4 in the RoAD paper
    x_grouped = x.reshape(-1, 2, group_size // 2)
    x1 = x_grouped[:, 0, :]
    x2 = x_grouped[:, 1, :]
    rotate_half_x = torch.stack((-x2, x1), dim=1).reshape(x.shape)
    result = x * first_col + rotate_half_x * second_col
    return result


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    road_config: RoadConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Linear):
        new_module = Linear(target, adapter_name, **kwargs)

    return new_module
