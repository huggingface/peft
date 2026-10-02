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

# The implementation is based on "Parameter-Efficient Orthogonal Finetuning
# via Butterfly Factorization" (https://huggingface.co/papers/2311.06243) in ICLR 2024.

from __future__ import annotations

import math
import warnings
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer, _get_in_out_features, check_adapters_to_merge
from peft.utils import quantization_extra_repr, resolve_quantization_backend

from .config import BOFTConfig


class MultiplicativeDropoutLayer(nn.Module):
    """
    Implements the multiplicative dropout layer for BOFT.
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
        x (Tensor): The input tensor of shape (N, D, H, H), where `N` is the batch size, `D` represents
                    one additional dimension (In BOFT, the number of BOFT blocks), and `H` is the size of the square
                    blocks along the last two dimensions (In BOFT, the block size).
        """
        if self.training:
            # Ensure the last two dimensions are the same
            if x.shape[-1] != x.shape[-2]:
                raise ValueError("The last two dimensions of input should be the same!")

            N, D, H, _ = x.shape

            # Randomly select one from N
            n_random = torch.randint(0, N, (1,)).item()

            # Create a mask with 1s for matrices to be replaced with identity and 0s otherwise
            num_to_replace = int(self.p * D)
            num_zeros = D - num_to_replace

            # Generate a flat tensor with desired number of 1s and 0s
            mask = torch.cat([torch.ones(num_to_replace, device=x.device), torch.zeros(num_zeros, device=x.device)])

            # Shuffle and reshape the mask
            mask = mask[torch.randperm(D)].view(1, D, 1, 1)

            full_mask = torch.zeros(N, D, 1, 1, device=x.device)
            full_mask[n_random] = mask

            # Use the mask to combine original matrices and identity matrices
            eye_matrix = torch.eye(H, device=x.device).repeat(N, D, 1, 1)
            x = (1 - full_mask) * x + full_mask * eye_matrix
        return x


class BOFTLayer(BaseTunerLayer):
    """
    Implements the BOFT layer.
    """

    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("boft_R", "boft_s")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("boft_block_size", "boft_block_num", "boft_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        """
        Initializes the BOFT layer.

        Note, currently only support linear layer and convolutional layer, with further support for other layers to be
        added soon.

        Parameters:
        base_layer: the pretrained model layer
        """
        self.base_layer = base_layer
        self.quantization_backend = resolve_quantization_backend(
            self.get_base_layer(), get_apply_tensor_subclass=kwargs.get("get_apply_tensor_subclass")
        )
        self.boft_block_size = {}
        self.boft_block_num = {}
        self.boft_dropout = nn.ModuleDict({})
        self.boft_R = nn.ParameterDict({})
        self.boft_s = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        # flag to enable/disable casting of input to weight dtype during forward call
        self.cast_input_dtype_enabled = True
        self.kwargs = kwargs

        base_layer = self.get_base_layer()

        in_features, out_features = _get_in_out_features(base_layer)
        if (in_features is None) or (out_features is None):
            raise TypeError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return

        warnings.warn("Scaling operation for BOFT not supported! Automatically set scale to 1.")

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.boft_R.keys():
                continue

            warnings.warn("Scaling operation for BOFT not supported! Automatically set scale to 1.")

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.boft_R.keys():
                continue

            warnings.warn("Unscaling operation for BOFT not supported! Keeping scale to 1.")

    def update_layer(
        self,
        adapter_name: str,
        config: BOFTConfig,
        **kwargs,
    ):
        """
        Update the linear layer with trainable BOFT weights. Override for other layer types.
        """
        boft_block_size = config.boft_block_size
        boft_block_num = config.boft_block_num
        boft_n_butterfly_factor = config.boft_n_butterfly_factor
        boft_dropout = config.boft_dropout
        init_weights = config.init_weights
        inference_mode = config.inference_mode

        # to be consistent with the paper notation
        boft_n_butterfly_factor = boft_n_butterfly_factor - 1
        if boft_n_butterfly_factor < 0:
            raise ValueError(
                f"You can only specify boft_n_butterfly_factor {boft_n_butterfly_factor + 1} to be a positive integer number."
            )

        # Initialize the MultiplicativeDropoutLayer for boft_dropout > 0.0.
        if boft_dropout > 0.0:
            boft_dropout_layer = MultiplicativeDropoutLayer(p=boft_dropout)
        else:
            boft_dropout_layer = nn.Identity()
        self.boft_dropout.update(nn.ModuleDict({adapter_name: boft_dropout_layer}))

        if boft_block_size == 0 and boft_block_num != 0:
            if self.in_features % boft_block_num != 0:
                raise ValueError(
                    f"in_features ({self.in_features}) must be divisible by boft_block_num ({boft_block_num})!"
                )

            if boft_n_butterfly_factor != 0:
                if boft_n_butterfly_factor > int(math.log2(boft_block_num)):
                    raise ValueError(
                        f"Invalid combination of boft_n_butterfly_factor ({boft_n_butterfly_factor + 1}) and boft_block_num ({boft_block_num})!"
                    )
                if boft_block_num % (2**boft_n_butterfly_factor) != 0:
                    raise ValueError(
                        f"boft_block_num ({boft_block_num}) must be a multiple of 2 raised to the power of boft_n_butterfly_factor ({boft_n_butterfly_factor + 1})!"
                    )

            boft_block_size = int(self.in_features // boft_block_num)

        elif boft_block_size != 0 and boft_block_num == 0:
            if self.in_features % boft_block_size != 0:
                raise ValueError(
                    f"in_features ({self.in_features}) must be divisible by boft_block_size ({boft_block_size})!"
                )

            if boft_n_butterfly_factor != 0:
                if self.in_features < (boft_block_size * (2**boft_n_butterfly_factor)):
                    raise ValueError(
                        f"Invalid combination of in_features ({self.in_features}), boft_n_butterfly_factor ({boft_n_butterfly_factor + 1}) and boft_block_size ({boft_block_size})!"
                    )
                if self.in_features % (boft_block_size * (2**boft_n_butterfly_factor)) != 0:
                    raise ValueError(
                        f"Invalid combination of in_features ({self.in_features}), boft_n_butterfly_factor ({boft_n_butterfly_factor + 1}) and boft_block_size ({boft_block_size})!"
                    )

            boft_block_num = int(self.in_features // boft_block_size)

        else:
            raise ValueError(
                "Something went wrong, please report this error: https://github.com/huggingface/peft/issues"
            )

        # In OFT you can specify the number of blocks to be 1
        if boft_n_butterfly_factor != 0:
            if boft_block_num % 2 != 0:
                raise ValueError(f"boft_block_num ({boft_block_num}) must be an even number!")

            if boft_block_size % 2 != 0:
                raise ValueError(f"boft_block_size ({boft_block_size}) must be an even number!")

        # The permutation P of each butterfly factor is stored as indices, as applying it via indexing is equivalent to,
        # but much faster than, multiplying with the corresponding permutation matrix. If there is no butterfly factor,
        # the permutation is the identity.
        P = torch.empty((boft_n_butterfly_factor + 1, self.in_features), dtype=torch.long)
        for i in range(boft_n_butterfly_factor + 1):
            P[i] = self.block_butterfly_perm(
                self.in_features, int(boft_block_num / (2 ** (i))), int(boft_block_size / 2), boft_n_butterfly_factor
            )

        self.register_buffer("boft_P", P, persistent=False)
        # the inverse permutations, i.e. boft_P_inv[i][boft_P[i]] == arange
        self.register_buffer("boft_P_inv", torch.argsort(P, dim=-1), persistent=False)

        self.boft_R[adapter_name] = nn.Parameter(
            torch.zeros(boft_n_butterfly_factor + 1, boft_block_num, boft_block_size, boft_block_size)
        )
        self.boft_s[adapter_name] = nn.Parameter(torch.ones(int(self.out_features), 1))

        self.reset_boft_parameters(adapter_name, init_weights)

        # set the boft block size and number
        self.boft_block_size[adapter_name] = boft_block_size
        self.boft_block_num[adapter_name] = boft_block_num

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def reset_boft_parameters(self, adapter_name, init_weights):
        """
        Reset the BOFT parameters.
        """
        if init_weights is False:
            nn.init.normal_(self.boft_R[adapter_name], mean=0.0, std=0.1)
            nn.init.normal_(self.boft_s[adapter_name], mean=1.0, std=0.1)
            return

        if adapter_name in self.boft_R.keys():
            if init_weights is True:
                # initialize R to zero
                nn.init.zeros_(self.boft_R[adapter_name])
                nn.init.ones_(self.boft_s[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_weights=}")

    def block_butterfly_perm(self, n, b, r=3, n_butterfly_factor=1):
        """
        Define the permutation matrix for the block butterfly permutation.

        Args:
        n: size of the permutation matrix
        b: desired number of blocks after multiplying with the permutation matrix
        r: base block size of the block diagonal matrix, e.g. 2x2, 3x3, 5x5 etc.
        """

        if n_butterfly_factor == 0:
            return torch.arange(n)

        if b * r * 2 > n:
            raise ValueError("Invalid number of blocks!")

        block_size = int(n // b)
        indices = torch.arange(n)

        def sort_block(b, r):
            step = b / r
            initial_order = torch.arange(b)
            sorted_order = torch.empty(b, dtype=torch.long)

            evens = torch.arange(0, step, 2)
            odds = torch.arange(1, step, 2)
            sorted_seq = torch.cat((evens, odds), dim=0)
            for i, pos in enumerate(sorted_seq):
                sorted_order[int(i * r) : int(i * r + r)] = initial_order[int(pos * r) : int(pos * r + r)]
            return sorted_order

        sorted_order = sort_block(block_size, r)

        for i in range(0, n, block_size):
            block_end = i + block_size
            tmp_indices = indices[i:block_end]
            indices[i:block_end] = tmp_indices[sorted_order]
        return indices

    def cayley_batch(self, data):
        """
        Perform the Cayley parametrization on a batch of skew-symmetric matrices.

        Args:
            data: A batch of skew-symmetric matrices of shape (b, r, c).
        """
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew_mat = 0.5 * (data - data.transpose(1, 2))
        id_mat = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization, must be in float32
        Q = torch.linalg.solve(id_mat + skew_mat, id_mat - skew_mat, left=False)

        return Q.to(data.dtype)


class Linear(nn.Module, BOFTLayer):
    """
    BOFT implemented in a dense layer.
    """

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        config: BOFTConfig,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        BOFTLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = config.fan_in_fan_out

        self._active_adapter = adapter_name

        self.update_layer(adapter_name, config=config)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

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
            if active_adapter in self.boft_R.keys():
                orig_weight = self.get_base_weight().clone()
                orig_dtype = orig_weight.dtype
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    butterfly_oft_mat, boft_s = self.get_delta_weight(active_adapter)
                    orig_weight = torch.transpose(orig_weight, 0, 1)
                    orig_weight = torch.mm(butterfly_oft_mat, orig_weight.to(butterfly_oft_mat.dtype))
                    orig_weight = torch.transpose(orig_weight, 0, 1)
                    orig_weight = orig_weight * boft_s

                    if not torch.isfinite(orig_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    self.set_base_weight(orig_weight.contiguous().to(orig_dtype))
                else:
                    butterfly_oft_mat, boft_s = self.get_delta_weight(active_adapter)
                    orig_weight = torch.transpose(orig_weight, 0, 1)
                    orig_weight = torch.mm(butterfly_oft_mat, orig_weight.to(butterfly_oft_mat.dtype))
                    orig_weight = torch.transpose(orig_weight, 0, 1)
                    orig_weight = orig_weight * boft_s

                    self.set_base_weight(orig_weight.contiguous().to(orig_dtype))

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
            if active_adapter in self.boft_R.keys():
                butterfly_oft_mat, boft_s = self.get_delta_weight(active_adapter)

                orig_weight = self.get_base_weight().clone()
                orig_dtype = orig_weight.dtype
                orig_weight = torch.transpose(orig_weight, 0, 1)
                orig_weight = torch.mm(butterfly_oft_mat.t(), orig_weight.to(butterfly_oft_mat.dtype))
                orig_weight = torch.transpose(orig_weight, 0, 1)

                self.set_base_weight((orig_weight * (1 / boft_s)).to(orig_dtype))

    def get_delta_weight(self, adapter) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        boft_R = self.boft_R[adapter]
        boft_s = self.boft_s[adapter]

        N, D, H, _ = boft_R.shape
        boft_R = boft_R.view(N * D, H, H)
        orth_rotate_butterfly = self.cayley_batch(boft_R)
        orth_rotate_butterfly = orth_rotate_butterfly.view(N, D, H, H)
        # For merging, the rotation matrix of each butterfly factor is materialized: scatter the rotation blocks into
        # a block diagonal matrix B, then apply the permutation from both sides (P @ B @ P^T) by indexing the rows and
        # columns of B.
        block_diagonal_butterfly = orth_rotate_butterfly.new_zeros(N, D, H, D, H)
        block_diagonal_butterfly.diagonal(dim1=1, dim2=3).copy_(orth_rotate_butterfly.permute(0, 2, 3, 1))
        block_diagonal_butterfly = block_diagonal_butterfly.view(N, D * H, D * H)

        boft_P = self.boft_P.to(block_diagonal_butterfly.device)
        butterfly_oft_mat_batch = torch.stack(
            [mat[perm][:, perm] for mat, perm in zip(block_diagonal_butterfly, boft_P)]
        )
        butterfly_oft_mat = butterfly_oft_mat_batch[0]

        for i in range(1, butterfly_oft_mat_batch.shape[0]):
            butterfly_oft_mat = butterfly_oft_mat_batch[i] @ butterfly_oft_mat

        return butterfly_oft_mat, boft_s

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            boft_scale = torch.ones((int(self.out_features), 1), device=x.device, dtype=previous_dtype)

            # The rotation of each adapter is applied to x directly instead of composing the full rotation matrix
            # first. As x is multiplied from the left of the composed rotation, the adapters are iterated in reverse
            # order to preserve the order in which the rotation matrices used to be composed.
            for active_adapter in reversed(self.active_adapters):
                if active_adapter not in self.boft_R.keys():
                    continue
                boft_R = self.boft_R[active_adapter]
                boft_s = self.boft_s[active_adapter]
                dropout = self.boft_dropout[active_adapter]

                N, D, H, _ = boft_R.shape
                boft_R = boft_R.view(N * D, H, H)
                orth_rotate_butterfly = self.cayley_batch(boft_R)
                orth_rotate_butterfly = orth_rotate_butterfly.view(N, D, H, H)
                orth_rotate_butterfly = dropout(orth_rotate_butterfly)
                # cayley_batch and dropout only return fp32 outputs
                orth_rotate_butterfly = orth_rotate_butterfly.to(x)

                boft_P = self.boft_P.to(x.device)
                boft_P_inv = self.boft_P_inv.to(x.device)
                batch_shape = x.shape[:-1]
                # Multiply x with the rotation matrix P_i @ B_i @ P_i^T of each butterfly factor i without materializing
                # it: indexing applies the permutation P_i and the blockwise matmul (analogous to the OFT forward)
                # applies the block diagonal B_i. The factors are iterated in reverse order because x is multiplied from
                # the left of the composed rotation.
                for i in range(N - 1, -1, -1):
                    x = x[..., boft_P_inv[i]]  # x @ P_i
                    x = torch.einsum("...dh,dhk->...dk", x.reshape(*batch_shape, D, H), orth_rotate_butterfly[i])
                    x = x.reshape(*batch_shape, self.in_features)[..., boft_P[i]]  # x @ P_i^T

                boft_scale = boft_s * boft_scale

            result = self.base_layer(x, *args, **kwargs)
            bias = self.base_layer.bias
            if bias is not None:
                result = result - bias
            result = result * boft_scale.squeeze(-1)
            if bias is not None:
                result = result + bias

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "boft." + rep

    def extra_repr(self) -> str:
        return quantization_extra_repr(self)


class Conv2d(nn.Module, BOFTLayer):
    """
    BOFT implemented in a Conv2d layer.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        config: BOFTConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        BOFTLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config=config)

    def update_layer(
        self,
        adapter_name: str,
        config: BOFTConfig,
        **kwargs,
    ):
        """
        Update the conv2d layer with trainable BOFT weights.
        """
        boft_block_size = config.boft_block_size
        boft_block_num = config.boft_block_num
        boft_n_butterfly_factor = config.boft_n_butterfly_factor
        boft_dropout = config.boft_dropout
        init_weights = config.init_weights
        inference_mode = config.inference_mode

        # to be consistent with the paper notation
        boft_n_butterfly_factor = boft_n_butterfly_factor - 1
        if boft_n_butterfly_factor < 0:
            raise ValueError(
                f"You can only specify boft_n_butterfly_factor {boft_n_butterfly_factor + 1} to be a positive integer number."
            )

        # Initialize the MultiplicativeDropoutLayer for boft_dropout > 0.0.
        if boft_dropout > 0.0:
            boft_dropout_layer = MultiplicativeDropoutLayer(p=boft_dropout)
        else:
            boft_dropout_layer = nn.Identity()
        self.boft_dropout.update(nn.ModuleDict({adapter_name: boft_dropout_layer}))

        # layer information from the base layer
        base_layer = self.get_base_layer()
        if base_layer.groups > 1:
            # The rotation below is built over the full in_channels * kernel_size**2, but a grouped conv's
            # weight only holds in_channels // groups there, so forward and merge both crash on shape mismatch.
            raise NotImplementedError(
                f"BOFT does not support {type(base_layer).__name__} layers with groups > 1 "
                f"(got groups={base_layer.groups})."
            )
        conv_filter_dim = self.in_features * base_layer.kernel_size[0] * base_layer.kernel_size[0]

        # Initialize the BOFT parameters.
        if boft_block_size == 0 and boft_block_num != 0:
            if conv_filter_dim % boft_block_num != 0:
                raise ValueError(
                    f"Convolutional kernel dimension ({conv_filter_dim}) must be divisible by boft_block_num ({boft_block_num})!"
                )

            if boft_n_butterfly_factor != 0:
                if boft_n_butterfly_factor > int(math.log2(boft_block_num)):
                    raise ValueError(
                        f"Invalid combination of boft_n_butterfly_factor ({boft_n_butterfly_factor + 1}) and boft_block_num ({boft_block_num})!"
                    )
                if boft_block_num % (2**boft_n_butterfly_factor) != 0:
                    raise ValueError(
                        f"boft_block_num ({boft_block_num}) must be a multiple of 2 raised to the power of boft_n_butterfly_factor ({boft_n_butterfly_factor + 1})!"
                    )

            boft_block_size = int(conv_filter_dim // boft_block_num)

        elif boft_block_size != 0 and boft_block_num == 0:
            if conv_filter_dim % boft_block_size != 0:
                raise ValueError(
                    f"Convolutional kernel dimension ({conv_filter_dim}) must be divisible by boft_block_size ({boft_block_size})!"
                )

            if boft_n_butterfly_factor != 0:
                if conv_filter_dim < (boft_block_size * (2**boft_n_butterfly_factor)):
                    raise ValueError(
                        f"Invalid combination of convolutional kernel dimension ({conv_filter_dim}), boft_n_butterfly_factor ({boft_n_butterfly_factor + 1}) and boft_block_size ({boft_block_size})!"
                    )
                if conv_filter_dim % (boft_block_size * (2**boft_n_butterfly_factor)) != 0:
                    raise ValueError(
                        f"Invalid combination of convolutional kernel dimension ({conv_filter_dim}), boft_n_butterfly_factor ({boft_n_butterfly_factor + 1}) and boft_block_size ({boft_block_size})!"
                    )

            boft_block_num = int(conv_filter_dim // boft_block_size)

        else:
            raise ValueError(
                "Something went wrong, please report this error: https://github.com/huggingface/peft/issues"
            )

        # In OFT you can specify the number of blocks to be 1
        if boft_n_butterfly_factor != 0:
            if boft_block_num % 2 != 0:
                raise ValueError(f"boft_block_num ({boft_block_num}) must be an even number!")

            if boft_block_size % 2 != 0:
                raise ValueError(f"boft_block_size ({boft_block_size}) must be an even number!")

        # The permutation P of each butterfly factor is stored as indices, as applying it via indexing is equivalent to,
        # but much faster than, multiplying with the corresponding permutation matrix. If there is no butterfly factor,
        # the permutation is the identity.
        P = torch.empty((boft_n_butterfly_factor + 1, conv_filter_dim), dtype=torch.long)
        for i in range(boft_n_butterfly_factor + 1):
            P[i] = self.block_butterfly_perm(
                conv_filter_dim, int(boft_block_num / (2 ** (i))), int(boft_block_size / 2), boft_n_butterfly_factor
            )

        self.register_buffer("boft_P", P, persistent=False)
        # the inverse permutations, i.e. boft_P_inv[i][boft_P[i]] == arange
        self.register_buffer("boft_P_inv", torch.argsort(P, dim=-1), persistent=False)

        self.boft_R[adapter_name] = nn.Parameter(
            torch.zeros(boft_n_butterfly_factor + 1, boft_block_num, boft_block_size, boft_block_size)
        )
        self.boft_s[adapter_name] = nn.Parameter(torch.ones(1, int(self.out_features)))

        self.reset_boft_parameters(adapter_name, init_weights)

        # set the boft block size and number
        self.boft_block_size[adapter_name] = boft_block_size
        self.boft_block_num[adapter_name] = boft_block_num

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

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
            if active_adapter in self.boft_R.keys():
                base_layer = self.get_base_layer()
                orig_weight = self.get_base_weight().clone()
                orig_dtype = orig_weight.dtype
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    butterfly_oft_mat, boft_s = self.get_delta_weight(active_adapter)

                    orig_weight = orig_weight.view(
                        self.out_features, self.in_features * base_layer.kernel_size[0] * base_layer.kernel_size[0]
                    )
                    orig_weight = torch.transpose(orig_weight, 0, 1)
                    orig_weight = torch.mm(butterfly_oft_mat, orig_weight.to(butterfly_oft_mat.dtype))
                    orig_weight = torch.transpose(orig_weight, 0, 1)
                    orig_weight = orig_weight * boft_s
                    orig_weight = orig_weight.view(
                        self.out_features, self.in_features, base_layer.kernel_size[0], base_layer.kernel_size[0]
                    )

                    self.set_base_weight(orig_weight.contiguous().to(orig_dtype))
                else:
                    butterfly_oft_mat, boft_s = self.get_delta_weight(active_adapter)

                    orig_weight = orig_weight.view(
                        self.out_features, self.in_features * base_layer.kernel_size[0] * base_layer.kernel_size[0]
                    )
                    orig_weight = torch.transpose(orig_weight, 0, 1)
                    orig_weight = torch.mm(butterfly_oft_mat, orig_weight.to(butterfly_oft_mat.dtype))
                    orig_weight = torch.transpose(orig_weight, 0, 1)
                    orig_weight = orig_weight * boft_s
                    orig_weight = orig_weight.view(
                        self.out_features, self.in_features, base_layer.kernel_size[0], base_layer.kernel_size[0]
                    )

                    self.set_base_weight(orig_weight.contiguous().to(orig_dtype))

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
            base_layer = self.get_base_layer()
            if active_adapter in self.boft_R.keys():
                butterfly_oft_mat, boft_s = self.get_delta_weight(active_adapter)

                orig_weight = self.get_base_weight().clone()
                orig_dtype = orig_weight.dtype
                orig_weight = orig_weight.view(
                    self.out_features,
                    self.in_features * base_layer.kernel_size[0] * base_layer.kernel_size[0],
                )
                orig_weight = torch.transpose(orig_weight, 0, 1)
                orig_weight = torch.mm(butterfly_oft_mat.t(), orig_weight.to(butterfly_oft_mat.dtype))
                orig_weight = torch.transpose(orig_weight, 0, 1)
                orig_weight = orig_weight * (1 / boft_s)
                orig_weight = orig_weight.view(
                    self.out_features,
                    self.in_features,
                    base_layer.kernel_size[0],
                    base_layer.kernel_size[0],
                )

                self.set_base_weight(orig_weight.to(orig_dtype))

    def get_delta_weight(self, adapter) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """

        boft_R = self.boft_R[adapter]
        boft_s = self.boft_s[adapter].transpose(0, 1)

        N, D, H, _ = boft_R.shape
        boft_R = boft_R.view(N * D, H, H)
        orth_rotate_butterfly = self.cayley_batch(boft_R)
        orth_rotate_butterfly = orth_rotate_butterfly.view(N, D, H, H)
        # For merging, the rotation matrix of each butterfly factor is materialized: scatter the rotation blocks into
        # a block diagonal matrix B, then apply the permutation from both sides (P @ B @ P^T) by indexing the rows and
        # columns of B.
        block_diagonal_butterfly = orth_rotate_butterfly.new_zeros(N, D, H, D, H)
        block_diagonal_butterfly.diagonal(dim1=1, dim2=3).copy_(orth_rotate_butterfly.permute(0, 2, 3, 1))
        block_diagonal_butterfly = block_diagonal_butterfly.view(N, D * H, D * H)

        boft_P = self.boft_P.to(block_diagonal_butterfly.device)
        butterfly_oft_mat_batch = torch.stack(
            [mat[perm][:, perm] for mat, perm in zip(block_diagonal_butterfly, boft_P)]
        )
        butterfly_oft_mat = butterfly_oft_mat_batch[0]

        for i in range(1, butterfly_oft_mat_batch.shape[0]):
            butterfly_oft_mat = butterfly_oft_mat_batch[i] @ butterfly_oft_mat

        return butterfly_oft_mat, boft_s

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            boft_scale = torch.ones((int(self.out_features), 1), device=x.device, dtype=x.dtype)

            orig_weight = self.get_base_weight()
            x = x.to(orig_weight.dtype)

            orig_weight = orig_weight.view(
                self.out_features,
                self.in_features * self.base_layer.kernel_size[0] * self.base_layer.kernel_size[0],
            )
            rotated_weight = torch.transpose(orig_weight, 0, 1)

            # The rotation of each adapter is applied to the weight directly instead of composing the full rotation
            # matrix first. As the weight is multiplied from the right of the composed rotation, the adapters are
            # iterated in their original order.
            for active_adapter in self.active_adapters:
                if active_adapter not in self.boft_R.keys():
                    continue
                boft_R = self.boft_R[active_adapter]
                boft_s = self.boft_s[active_adapter].transpose(0, 1)
                dropout = self.boft_dropout[active_adapter]

                N, D, H, _ = boft_R.shape
                boft_R = boft_R.view(N * D, H, H)
                orth_rotate_butterfly = self.cayley_batch(boft_R)
                orth_rotate_butterfly = orth_rotate_butterfly.view(N, D, H, H)
                orth_rotate_butterfly = dropout(orth_rotate_butterfly)
                orth_rotate_butterfly = orth_rotate_butterfly.to(rotated_weight)

                boft_P = self.boft_P.to(rotated_weight.device)
                boft_P_inv = self.boft_P_inv.to(rotated_weight.device)
                # Multiply the weight with the rotation matrix P_i @ B_i @ P_i^T of each butterfly factor i without
                # materializing it: indexing the rows applies the permutation P_i and the blockwise matmul (analogous
                # to the OFT forward) applies the block diagonal B_i.
                for i in range(N):
                    rotated_weight = rotated_weight[boft_P_inv[i]]  # P_i^T @ weight
                    rotated_weight = torch.einsum(
                        "dhk,dko->dho", orth_rotate_butterfly[i], rotated_weight.reshape(D, H, -1)
                    )
                    rotated_weight = rotated_weight.reshape(D * H, -1)[boft_P[i]]  # P_i @ (B_i @ P_i^T @ weight)

                boft_scale = boft_s * boft_scale

            rotated_weight = torch.transpose(rotated_weight, 0, 1)

            scaled_rotated_weight = rotated_weight * boft_scale

            scaled_rotated_weight = scaled_rotated_weight.view(
                self.out_features, self.in_features, self.base_layer.kernel_size[0], self.base_layer.kernel_size[0]
            )
            x = self._cast_input_dtype(x, scaled_rotated_weight.dtype)
            bias = self._cast_input_dtype(self.base_layer.bias, scaled_rotated_weight.dtype)
            result = F.conv2d(
                input=x,
                weight=scaled_rotated_weight,
                bias=bias,
                padding=self.base_layer.padding[0],
                stride=self.base_layer.stride[0],
            )

        result = result.to(previous_dtype)
        return result

    def extra_repr(self) -> str:
        return quantization_extra_repr(self)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "boft." + rep
