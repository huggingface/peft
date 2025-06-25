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

import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .config import OFTConfig


class MultiplicativeDropoutLayer(nn.Module):
    """
    Implements the multiplicative dropout layer for OFT.
    """

    def __init__(self, p=0.0):
        """
        Initializes the multiplicative dropout layer.

        Parameters:
        p (float): The probability of dropping out a block. Defaults to 0.0.
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        """
        Applies multiplicative dropout to the input tensor.

        Parameters:
        x (Tensor): The input tensor of shape (D, H, H), where `D` represents
                    the number of OFT blocks, and `H` is the size of the square blocks along the last two dimensions,
                    the block size in OFT.
        """
        if self.training and self.p > 0:
            # Ensure the last two dimensions are the same
            if x.shape[-1] != x.shape[-2]:
                raise ValueError("The last two dimensions of input should be the same!")

            D, H, _ = x.shape

            # If block share, skip the multiplicative dropout
            if D == 1:
                return x

            num_to_replace = int(self.p * D)
            num_zeros = D - num_to_replace
            mask = torch.cat([torch.ones(num_to_replace, device=x.device), torch.zeros(num_zeros, device=x.device)])
            mask = mask[torch.randperm(D)].view(D, 1, 1)
            eye_matrix = torch.eye(H, device=x.device).repeat(D, 1, 1)
            x = (1 - mask) * x + mask * eye_matrix
        return x


class OFTRotationModule(nn.Module):
    def __init__(
        self,
        r,
        n_elements,
        block_size,
        in_features,
        coft=False,
        eps=6e-5,
        block_share=False,
        kernel_size=(0, 0),
        use_cayley_neumann=True,
        num_cayley_neumann_terms=5,
    ):
        super().__init__()
        self.r = r
        self.n_elements = n_elements
        self.block_size = block_size
        self.in_features = in_features
        self.weight = nn.Parameter(torch.empty(r, n_elements))
        self.coft = coft
        self.eps = eps
        self.block_share = block_share
        # Conv2d specific parameters
        self.kernel_size = kernel_size
        self.use_cayley_neumann = use_cayley_neumann
        self.num_cayley_neumann_terms = num_cayley_neumann_terms
        # Create indices for upper triangle (excluding diagonal)
        self.rows, self.cols = torch.triu_indices(block_size, block_size, 1)

    def _pytorch_skew_symmetric(self, vec, block_size):
        batch_size = vec.shape[0]
        matrix = torch.zeros(batch_size, block_size, block_size, device=vec.device, dtype=vec.dtype)

        matrix[:, self.rows, self.cols] = vec
        matrix = matrix - matrix.transpose(-2, -1)
        return matrix

    def _pytorch_skew_symmetric_inv(self, matrix, block_size):
        batch_size = matrix.shape[0]

        # Extract the upper triangular elements
        vec = matrix[:, self.rows, self.cols]
        return vec

    def _cayley_batch(
        self, Q: torch.Tensor, block_size: int, use_cayley_neumann: bool = True, num_neumann_terms: int = 5
    ) -> torch.Tensor:
        """
        Perform the Cayley parametrization on a batch of skew-symmetric matrices.

        Args:
            data: A batch of skew-symmetric matrices of shape (b, r, c).
        """

        b, _ = Q.shape
        previous_dtype = Q.dtype

        # Q_skew = SkewSymmetric.apply(Q, block_size)
        Q_skew = self._pytorch_skew_symmetric(Q, block_size)

        if use_cayley_neumann:
            R = torch.eye(block_size, device=Q.device, dtype=Q.dtype).repeat(b, 1, 1)
            if num_neumann_terms > 1:
                R.add_(Q_skew, alpha=2.0)
                if num_neumann_terms > 2:
                    Q_squared = torch.bmm(Q_skew, Q_skew)
                    R.add_(Q_squared, alpha=2.0)

                    Q_power = Q_squared
                    for i in range(3, num_neumann_terms):
                        Q_power = torch.bmm(Q_power, Q_skew)
                        R.add_(Q_power, alpha=2.0)
        else:
            id_mat = (
                torch.eye(Q_skew.shape[-1], device=Q_skew.device)
                .unsqueeze(0)
                .expand(b, Q_skew.shape[-1], Q_skew.shape[-1])
            )
            R = torch.linalg.solve(id_mat + Q_skew, id_mat - Q_skew, left=False)

        return R.to(previous_dtype)

    # Copied from https://github.com/Zeju1997/oft/blob/84cebb965df69781e3d9c3c875f5980b421eaf24/oft-control/oft.py#L52
    def _project_batch(self, Q, eps=1e-5):
        oft_R = self._pytorch_skew_symmetric(Q, self.block_size)
        # scaling factor for each of the smaller block matrix
        eps = eps * 1 / torch.sqrt(torch.tensor(oft_R.shape[0]))
        I = (  # noqa: E741
            torch.zeros((oft_R.size(1), oft_R.size(1)), device=oft_R.device, dtype=oft_R.dtype)
            .unsqueeze(0)
            .expand_as(oft_R)
        )
        diff = oft_R - I
        norm_diff = torch.norm(oft_R - I, dim=(1, 2), keepdim=True)
        mask = (norm_diff <= eps).bool()
        out = torch.where(mask, oft_R, I + eps * (diff / norm_diff))

        return self._pytorch_skew_symmetric_inv(out, self.block_size)

    # Copied from https://github.com/Zeju1997/oft/blob/84cebb965df69781e3d9c3c875f5980b421eaf24/oft-control/oft.py#L155
    def _block_diagonal(self, oft_R: torch.Tensor, rank: int) -> torch.Tensor:
        if oft_R.shape[0] == 1:
            # block share
            blocks = [oft_R[0, ...] for i in range(rank)]
        else:
            blocks = [oft_R[i, ...] for i in range(rank)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

    def _unfold(self, x):
        """
        Unfold with stride=1, padding=0 to preserve spatial dimensions. Only use kernel_size from base layer to define
        patch size.
        """
        batch_size, in_channels, in_height, in_width = x.shape

        if isinstance(self.kernel_size, int):
            kernel_height, kernel_width = self.kernel_size, self.kernel_size
        else:
            kernel_height, kernel_width = self.kernel_size

        stride_h = stride_w = 1
        pad_h = pad_w = 0

        # output dimensions
        out_height = (in_height + 2 * pad_h - kernel_height) // stride_h + 1
        out_width = (in_width + 2 * pad_w - kernel_width) // stride_w + 1

        # Reshape input from [B, C, H, W] to [B, C, H_out, W_out, K_H, K_W]
        x_unfolded = x.unfold(2, kernel_height, stride_h).unfold(3, kernel_width, stride_w)
        x_unfolded = x_unfolded.permute(0, 2, 3, 1, 4, 5).contiguous()
        x_unfolded = x_unfolded.view(batch_size * out_height * out_width, -1)

        return x_unfolded

    def _fold(self, x_unfolded, orig_shape):
        """
        Fold back to preserve spatial dimensions.
        """
        batch_size, in_channels, in_height, in_width = orig_shape

        if isinstance(self.kernel_size, int):
            kernel_height, kernel_width = self.kernel_size, self.kernel_size
        else:
            kernel_height, kernel_width = self.kernel_size

        # With stride=1, padding=0:
        out_height = in_height - kernel_height + 1
        out_width = in_width - kernel_width + 1

        # Reshape: [B*H_out*W_out, C*K_H*K_W] -> [B, H_out, W_out, C, K_H, K_W]
        x_reshaped = x_unfolded.view(batch_size, out_height, out_width, in_channels, kernel_height, kernel_width)

        # Permute to: [B, C, H_out, W_out, K_H, K_W]
        x_reshaped = x_reshaped.permute(0, 3, 1, 2, 4, 5).contiguous()

        # Use F.fold to reconstruct 4D tensor
        x_folded = F.fold(
            x_reshaped.view(batch_size, in_channels * kernel_height * kernel_width, out_height * out_width),
            output_size=(in_height, in_width),
            kernel_size=(kernel_height, kernel_width),
            stride=(1, 1),
        )

        return x_folded

    def forward(self, x):
        # This module doesn't need to implement the orthogonal transform
        # It's primarily a container for the parameter
        # The actual transformation logic stays in your OFTLayer

        required_dtype = x.dtype
        if required_dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)

        orig_shape = x.shape

        if self.coft:
            with torch.no_grad():
                self.weight.copy_(self._project_batch(self.weight, eps=self.eps))

        orth_rotate = self._cayley_batch(
            self.weight, self.block_size, self.use_cayley_neumann, self.num_cayley_neumann_terms
        )

        # Unfold the input for Conv2d layer
        if len(orig_shape) == 4:
            x = self._unfold(x)

        folded_shape = x.shape
        rank = self.in_features // self.block_size if self.block_share else self.r
        batch_dims = x.shape[:-1]
        x_reshaped = x.reshape(*batch_dims, rank, self.block_size)

        if self.block_share:
            orth_rotate = orth_rotate.repeat(rank, 1, 1)
            x_rotated_reshaped = torch.einsum("...rk,rkc->...rc", x_reshaped, orth_rotate)
        else:
            x_rotated_reshaped = torch.einsum("...rk,rkc->...rc", x_reshaped, orth_rotate)

        x_rotated = x_rotated_reshaped.reshape(*folded_shape)

        if len(orig_shape) == 4:
            x_rotated = self._fold(x_rotated, orig_shape)

        return x_rotated.to(required_dtype)

    def get_weight(self):
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        weight = self.weight

        if self.coft:
            with torch.no_grad():
                weight = self._project_batch(weight, eps=self.eps)
                self.weight.copy_(weight)

        orth_rotate = self._cayley_batch(
            weight, self.block_size, self.use_cayley_neumann, self.num_cayley_neumann_terms
        )

        rank = self.r if not self.block_share else self.in_features // self.block_size
        return self._block_diagonal(orth_rotate, rank)


class OFTLayer(BaseTunerLayer):
    """
    Implements the OFT layer.
    """

    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names: tuple[str, ...] = ("oft_R",)
    # All names of other parameters that may contain adapter-related parameters
    other_param_names: tuple[str, ...] = ("r", "oft_block_size", "oft_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        """
        Initializes the OFT layer.

        Note, currently only support linear layer and convolutional layer, with further support for other layers to be
        added soon.

        Parameters:
        base_layer: the pretrained model layer
        """
        self.base_layer = base_layer
        self.oft_R = nn.ModuleDict({})
        self.oft_block_size = {}
        self.r = {}
        self.oft_block_size = {}
        self.oft_dropout = nn.ModuleDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        # flag to enable/disable casting of input to weight dtype during forward call
        self.cast_input_dtype_enabled = True
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "EetqLinear":
            # Eetq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear":
            # HQQ layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features

    @property
    def _available_adapters(self) -> set[str]:
        return {*self.oft_R}

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return

        warnings.warn("Scaling operation for OFT not supported! Automatically set scale to 1.")

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.oft_R.keys():
                continue

            warnings.warn("Scaling operation for OFT not supported! Automatically set scale to 1.")

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.oft_R.keys():
                continue

            warnings.warn("Unscaling operation for OFT not supported! Keeping scale to 1.")

    def update_layer(
        self,
        adapter_name,
        r,
        oft_block_size,
        module_dropout,
        coft,
        eps,
        block_share,
        init_weights,
        use_cayley_neumann,
        num_cayley_neumann_terms,
    ):
        """
        Update the linear layer with trainable OFT weights. Override for other layer types.
        """
        """Internal function to create oft adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            oft_block_size (`int`): The block size for added adapter.
            module_dropout (`float`):
                The multiplicative dropout probability for disabling adapter blocks during training.
            coft (`bool`): Whether to use the constrained variant of OFT or not.
            eps (`float`):
                The control strength of COFT. The freedom of rotation. Only has an effect if `coft` is set to True.
            block_share (`bool`): Whether to share the OFT parameters between blocks or not.
            init_weights (`bool`): Whether to initialize weights.
        """
        # Initialize the MultiplicativeDropoutLayer for module_dropout > 0.0.
        if module_dropout > 0.0:
            oft_dropout_layer = MultiplicativeDropoutLayer(p=module_dropout)
        else:
            oft_dropout_layer = nn.Identity()
        self.oft_dropout.update(nn.ModuleDict({adapter_name: oft_dropout_layer}))

        if r == 0 and oft_block_size != 0:
            if self.in_features % oft_block_size != 0 or oft_block_size > self.in_features:
                old_oft_block_size = oft_block_size
                oft_block_size = self.adjust_oft_parameters(self.in_features, oft_block_size)
                warnings.warn(
                    f"Invalid `oft_block_size` ({old_oft_block_size})! Adjusted `oft_block_size` to ({oft_block_size})."
                )
            r = int(self.in_features // oft_block_size)
        elif r != 0 and oft_block_size == 0:
            if self.in_features % r != 0 or r > self.in_features:
                old_r = r
                r = self.adjust_oft_parameters(self.in_features, r)
                warnings.warn(f"Invalid `r` ({old_r})! Adjusted `r` to ({r}).")
            oft_block_size = int(self.in_features // r)
        else:
            raise ValueError(
                "Something went wrong, please report this error: https://github.com/huggingface/peft/issues"
            )

        # Create weights with provided shape
        n_elements = oft_block_size * (oft_block_size - 1) // 2
        self.oft_R[adapter_name] = OFTRotationModule(
            r if not block_share else 1,
            n_elements,
            oft_block_size,
            self.in_features,
            coft=coft,
            eps=eps,
            block_share=block_share,
            use_cayley_neumann=use_cayley_neumann,
            num_cayley_neumann_terms=num_cayley_neumann_terms,
        )

        # Initialize weights
        self.reset_oft_parameters(adapter_name, init_weights)

        # set oft r and block size
        self.r[adapter_name] = r
        self.oft_block_size[adapter_name] = oft_block_size

        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_oft_parameters(self, adapter_name, init_weights):
        """
        Reset the OFT parameters.
        """
        if init_weights is False:
            nn.init.normal_(self.oft_R[adapter_name].weight, mean=0.0, std=0.1)
            return

        if adapter_name in self.oft_R.keys():
            if init_weights is True:
                # initialize oft_R to zero
                nn.init.zeros_(self.oft_R[adapter_name].weight)
            else:
                raise ValueError(f"Unknown initialization {init_weights=}")

    def adjust_oft_parameters(self, in_features, params):
        """
        Adjust the OFT parameters to be divisible by the in_features dimension.
        """
        if params < in_features:
            higher_params = params
            while higher_params <= in_features and in_features % higher_params != 0:
                higher_params += 1
        else:
            return in_features

        lower_params = params
        while lower_params > 1 and in_features % lower_params != 0:
            lower_params -= 1

        if (params - lower_params) <= (higher_params - params):
            return lower_params
        else:
            return higher_params


class Linear(nn.Module, OFTLayer):
    """OFT implemented in Linear layer"""

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 8,
        oft_block_size: int = 0,
        module_dropout: float = 0.0,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        use_cayley_neumann: bool = False,
        num_cayley_neumann_terms: int = 5,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        init_weights: Union[bool, str] = True,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        OFTLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

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
            use_cayley_neumann=use_cayley_neumann,
            num_cayley_neumann_terms=num_cayley_neumann_terms,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

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
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    orig_weights = base_layer.weight.data
                    oft_mat = self.get_delta_weight(active_adapter)
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = torch.mm(oft_mat, orig_weights.to(oft_mat.dtype))
                    orig_weights = torch.transpose(orig_weights, 0, 1)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights.contiguous().to(orig_dtype)
                else:
                    orig_weights = base_layer.weight.data
                    oft_mat = self.get_delta_weight(active_adapter)
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = torch.mm(oft_mat, orig_weights.to(oft_mat.dtype))
                    orig_weights = torch.transpose(orig_weights, 0, 1)

                    base_layer.weight.data = orig_weights.contiguous().to(orig_dtype)

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        base_layer = self.get_base_layer()
        orig_dtype = base_layer.weight.dtype
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.oft_R.keys():
                oft_mat = self.get_delta_weight(active_adapter)

                orig_weights = self.get_base_layer().weight.data
                orig_weights = torch.transpose(orig_weights, 0, 1)
                orig_weights = torch.mm(oft_mat.t(), orig_weights.to(oft_mat.dtype))
                orig_weights = torch.transpose(orig_weights, 0, 1)

                base_layer.weight.data = orig_weights.to(orig_dtype)

    def get_delta_weight(self, adapter_name) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """

        return self.oft_R[adapter_name].get_weight()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            for active_adapter in self.active_adapters:
                if active_adapter not in self.oft_R.keys():
                    continue
                oft_R = self.oft_R[active_adapter]

                x = self._cast_input_dtype(x, oft_R.weight.dtype)
                x = oft_R(x)

            result = self.base_layer(x.to(previous_dtype), *args, **kwargs)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "oft." + rep


class Conv2d(nn.Module, OFTLayer):
    """OFT implemented in Conv2d layer"""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 8,
        oft_block_size: int = 0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        module_dropout: float = 0.0,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        init_weights: Union[bool, str] = True,
        use_cayley_neumann: bool = False,
        num_cayley_neumann_terms: int = 5,
        **kwargs,
    ) -> None:
        super().__init__()
        OFTLayer.__init__(self, base_layer)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name

        # Create adapter and set it active
        self.update_layer(
            adapter_name,
            r,
            oft_block_size=oft_block_size,
            module_dropout=module_dropout,
            coft=coft,
            eps=eps,
            block_share=block_share,
            init_weights=init_weights,
            use_cayley_neumann=use_cayley_neumann,
            num_cayley_neumann_terms=num_cayley_neumann_terms,
        )

    def update_layer(
        self,
        adapter_name,
        r,
        oft_block_size,
        module_dropout,
        coft,
        eps,
        block_share,
        init_weights,
        use_cayley_neumann,
        num_cayley_neumann_terms,
    ):
        """
        Update the conv2d layer with trainable OFT weights.
        """
        # Initialize the MultiplicativeDropoutLayer for module_dropout > 0.0.
        if module_dropout > 0.0:
            oft_dropout_layer = MultiplicativeDropoutLayer(p=module_dropout)
        else:
            oft_dropout_layer = nn.Identity()
        self.oft_dropout.update(nn.ModuleDict({adapter_name: oft_dropout_layer}))

        # layer information from the base layer
        base_layer = self.get_base_layer()
        if base_layer.dilation[0] > 1:
            raise ValueError("Conv2d with dilation > 1 is not supported by OFT.")

        conv_filter_dim = self.in_features * base_layer.kernel_size[0] * base_layer.kernel_size[0]

        if r == 0 and oft_block_size != 0:
            if conv_filter_dim % oft_block_size != 0 or oft_block_size > conv_filter_dim:
                old_oft_block_size = oft_block_size
                oft_block_size = self.adjust_oft_parameters(conv_filter_dim, oft_block_size)
                warnings.warn(
                    f"Invalid `oft_block_size` ({old_oft_block_size})! Adjusted `oft_block_size` to ({oft_block_size})."
                )
            r = int(conv_filter_dim // oft_block_size)
        elif r != 0 and oft_block_size == 0:
            if conv_filter_dim % r != 0 or r > conv_filter_dim:
                old_r = r
                r = self.adjust_oft_parameters(conv_filter_dim, r)
                warnings.warn(f"Invalid `r` ({old_r})! Adjusted `r` to ({r}).")
            oft_block_size = int(conv_filter_dim // r)
        else:
            raise ValueError(
                "Something went wrong, please report this error: https://github.com/huggingface/peft/issues"
            )

        # Create weights with provided shape
        n_elements = oft_block_size * (oft_block_size - 1) // 2
        self.oft_R[adapter_name] = OFTRotationModule(
            r if not block_share else 1,
            n_elements,
            oft_block_size,
            conv_filter_dim,
            coft=coft,
            eps=eps,
            block_share=block_share,
            kernel_size=base_layer.kernel_size,
            use_cayley_neumann=use_cayley_neumann,
            num_cayley_neumann_terms=num_cayley_neumann_terms,
        )

        # Initialize weights
        self.reset_oft_parameters(adapter_name, init_weights)

        # set oft r and block size
        self.r[adapter_name] = r
        self.oft_block_size[adapter_name] = oft_block_size

        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.oft_R.keys():
                base_layer = self.get_base_layer()
                orig_dtype = base_layer.weight.dtype
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    oft_mat = self.get_delta_weight(active_adapter)

                    orig_weights = orig_weights.view(
                        self.out_features, self.in_features * base_layer.kernel_size[0] * base_layer.kernel_size[0]
                    )
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = torch.mm(oft_mat, orig_weights.to(oft_mat.dtype))
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = orig_weights.view(
                        self.out_features, self.in_features, base_layer.kernel_size[0], base_layer.kernel_size[0]
                    )

                    base_layer.weight.data = orig_weights.contiguous().to(orig_dtype)
                else:
                    oft_mat = self.get_delta_weight(active_adapter)

                    orig_weights = base_layer.weight.data.clone()
                    orig_weights = orig_weights.view(
                        self.out_features, self.in_features * base_layer.kernel_size[0] * base_layer.kernel_size[0]
                    )
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = torch.mm(oft_mat, orig_weights.to(oft_mat.dtype))
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = orig_weights.view(
                        self.out_features, self.in_features, base_layer.kernel_size[0], base_layer.kernel_size[0]
                    )

                    base_layer.weight.data = orig_weights.contiguous().to(orig_dtype)

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        base_layer = self.get_base_layer()
        orig_dtype = base_layer.weight.dtype
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.oft_R.keys():
                oft_mat = self.get_delta_weight(active_adapter)

                orig_weights = self.get_base_layer().weight.data.clone()
                orig_weights = orig_weights.view(
                    self.out_features,
                    self.in_features * self.get_base_layer().kernel_size[0] * self.get_base_layer().kernel_size[0],
                )
                orig_weights = torch.transpose(orig_weights, 0, 1)
                orig_weights = torch.mm(oft_mat.t(), orig_weights.to(oft_mat.dtype))
                orig_weights = torch.transpose(orig_weights, 0, 1)
                orig_weights = orig_weights.view(
                    self.out_features,
                    self.in_features,
                    self.get_base_layer().kernel_size[0],
                    self.get_base_layer().kernel_size[0],
                )

                base_layer.weight.data = orig_weights.to(orig_dtype)

    def get_delta_weight(self, adapter_name) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """

        return self.oft_R[adapter_name].get_weight()

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            for active_adapter in self.active_adapters:
                if active_adapter not in self.oft_R.keys():
                    continue

                oft_R = self.oft_R[active_adapter]
                x = self._cast_input_dtype(x, oft_R.weight.dtype)
                x = oft_R(x)

            result = self.base_layer(x.to(previous_dtype), *args, **kwargs)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "oft." + rep


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    oft_config: OFTConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Conv2d):
        new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = oft_config.fan_in_fan_out = False
        new_module = Linear(target, adapter_name, **kwargs)

    return new_module
