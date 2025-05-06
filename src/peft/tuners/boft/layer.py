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
# via Butterfly Factorization" (https://arxiv.org/abs/2311.06243) in ICLR 2024.

from __future__ import annotations

import math
import os
import warnings
from contextlib import contextmanager
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


_FBD_CUDA = None


# this function is a 1:1 copy from accelerate
@contextmanager
def patch_environment(**kwargs):
    """
    A context manager that will add each keyword argument passed to `os.environ` and remove them when exiting.

    Will convert the values in `kwargs` to strings and upper-case all the keys.

    Example:

    ```python
    >>> import os
    >>> from accelerate.utils import patch_environment

    >>> with patch_environment(FOO="bar"):
    ...     print(os.environ["FOO"])  # prints "bar"
    >>> print(os.environ["FOO"])  # raises KeyError
    ```
    """
    existing_vars = {}
    for key, value in kwargs.items():
        key = key.upper()
        if key in os.environ:
            existing_vars[key] = os.environ[key]
        os.environ[key] = str(value)

    yield

    for key in kwargs:
        key = key.upper()
        if key in existing_vars:
            # restore previous value
            os.environ[key] = existing_vars[key]
        else:
            os.environ.pop(key, None)


def get_fbd_cuda():
    global _FBD_CUDA

    if _FBD_CUDA is not None:
        return _FBD_CUDA

    # This import initializes cuda context and should thus be local, see issue 1877
    from torch.utils.cpp_extension import load

    curr_dir = os.path.dirname(__file__)
    # need ninja to build the extension
    try:
        with patch_environment(CC="gcc", CXX="gcc"):
            fbd_cuda = load(
                name="fbd_cuda",
                sources=[f"{curr_dir}/fbd/fbd_cuda.cpp", f"{curr_dir}/fbd/fbd_cuda_kernel.cu"],
                verbose=True,
                # build_directory='/tmp/'  # for debugging
            )
            # extra_cuda_cflags = ['-std=c++14', '-ccbin=$$(which gcc-7)']) # cuda10.2 is not compatible with gcc9. Specify gcc 7
    except Exception as e:
        warnings.warn(f"Failed to load the CUDA extension: {e}, check if ninja is available.")
        warnings.warn("Setting boft_n_butterfly_factor to 1 to speed up the finetuning process.")
        fbd_cuda = None

    _FBD_CUDA = fbd_cuda
    return _FBD_CUDA


class FastBlockDiag(Function):
    """
    Implements a custom autograd Function for a fast block diagonal operation using CUDA.

    This function is optimized for 4D tensors where the last two dimensions are equal, representing block diagonal
    matrices for efficient computation on CUDA devices.
    """

    @staticmethod
    def forward(ctx, input):
        """
        The forward method for FastBlockDiag.

        Computes the block diagonal operation on the input tensor using a CUDA-optimized function. This method assumes
        that the input is a 4D tensor where the last two dimensions are equal, which represent the blocks to be
        diagonalized.

        Parameters:
        ctx: A context object that can be used to stash information for backward computation.
        input (Tensor): The input tensor of shape (N, D, H, H), where `N` is the batch size,
                        `D` represents one additional dimension (In BOFT, the number of BOFT blocks), and `H` is the
                        size of the square blocks along the last two dimensions (In BOFT, the block size).

        Returns:
        Tensor: The resulting tensor after applying the block diagonal operation,
                will have the shape (N, DxH, DxH).
        """
        output = get_fbd_cuda().forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = get_fbd_cuda().backward(grad_output, input)[0]
        return grad_input


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
        self.boft_block_size = {}
        self.boft_block_num = {}
        self.boft_dropout = nn.ModuleDict({})
        self.boft_R = nn.ParameterDict({})
        self.boft_s = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()

        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

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
        self, adapter_name, boft_block_size, boft_block_num, boft_n_butterfly_factor, boft_dropout, init_weights
    ):
        """
        Update the linear layer with trainable BOFT weights. Override for other layer types.
        """
        # Attempt to load the CUDA extension during model initialization
        if not get_fbd_cuda():
            self.fbd_cuda_available = False
            # If the CUDA extension is not available, set the butterfly factor to 1 to speed up the finetuning process
            boft_n_butterfly_factor = 1
        else:
            self.fbd_cuda_available = True

        # to be consistent with the paper notation
        boft_n_butterfly_factor = boft_n_butterfly_factor - 1
        if boft_n_butterfly_factor < 0:
            raise ValueError(
                f"You can only specify boft_n_butterfly_factor {boft_n_butterfly_factor+1} to be a positive integer number."
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
                        f"Invalid combination of boft_n_butterfly_factor ({boft_n_butterfly_factor+1}) and boft_block_num ({boft_block_num})!"
                    )
                if boft_block_num % (2**boft_n_butterfly_factor) != 0:
                    raise ValueError(
                        f"boft_block_num ({boft_block_num}) must be a multiple of 2 raised to the power of boft_n_butterfly_factor ({boft_n_butterfly_factor+1})!"
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
                        f"Invalid combination of in_features ({self.in_features}), boft_n_butterfly_factor ({boft_n_butterfly_factor+1}) and boft_block_size ({boft_block_size})!"
                    )
                if self.in_features % (boft_block_size * (2**boft_n_butterfly_factor)) != 0:
                    raise ValueError(
                        f"Invalid combination of in_features ({self.in_features}), boft_n_butterfly_factor ({boft_n_butterfly_factor+1}) and boft_block_size ({boft_block_size})!"
                    )

            boft_block_num = int(self.in_features // boft_block_size)

        else:
            raise ValueError(
                f"You can only specify either boft_block_size ({boft_block_size}) or boft_block_num ({boft_block_num}), but not both simultaneously or setting both"
                "to be 0, because boft_block_size x boft_block_num != in_features."
            )

        # In OFT you can specify the number of blocks to be 1
        if boft_n_butterfly_factor != 0:
            if boft_block_num % 2 != 0:
                raise ValueError(f"boft_block_num ({boft_block_num}) must be an even number!")

            if boft_block_size % 2 != 0:
                raise ValueError(f"boft_block_size ({boft_block_size}) must be an even number!")

        # If there is no butterfly factor, then permutation matrix P will be an identity matrix.
        P = torch.empty((boft_n_butterfly_factor + 1, self.in_features, self.in_features))
        for i in range(boft_n_butterfly_factor + 1):
            perm = self.block_butterfly_perm(
                self.in_features, int(boft_block_num / (2 ** (i))), int(boft_block_size / 2), boft_n_butterfly_factor
            )
            perm_mat = self.perm2mat(perm)
            P[i] = perm_mat

        self.register_buffer("boft_P", P, persistent=False)

        self.boft_R[adapter_name] = nn.Parameter(
            torch.zeros(boft_n_butterfly_factor + 1, boft_block_num, boft_block_size, boft_block_size)
        )
        self.boft_s[adapter_name] = nn.Parameter(torch.ones(int(self.out_features), 1))

        self.reset_boft_parameters(adapter_name, init_weights)

        # set the boft block size and number
        self.boft_block_size[adapter_name] = boft_block_size
        self.boft_block_num[adapter_name] = boft_block_num

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

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

    def perm2mat(self, indices):
        """
        Convert permutation indices to permutation matrix.

        Args:
        indices: A list of indices representing the permutation.
        """
        # Number of indices determines the size of the square matrix
        n = len(indices)

        # Initialize a matrix of zeros
        perm_mat = torch.zeros((n, n))

        # Set the 1s according to the indices
        for i, idx in enumerate(indices):
            perm_mat[i, idx] = 1

        return perm_mat

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

        # Perform the Cayley parametrization
        Q = torch.linalg.solve(id_mat + skew_mat, id_mat - skew_mat, left=False)

        return Q


class Linear(nn.Module, BOFTLayer):
    """
    BOFT implemented in a dense layer.
    """

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        boft_block_size: int = 8,
        boft_block_num: int = 0,
        boft_n_butterfly_factor: int = 0,
        boft_dropout: float = 0.1,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        init_weights: Union[bool, str] = True,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        BOFTLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name

        self.update_layer(
            adapter_name, boft_block_size, boft_block_num, boft_n_butterfly_factor, boft_dropout, init_weights
        )
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
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weight = base_layer.weight.data.clone()
                    butterfly_oft_mat, boft_s = self.get_delta_weight(active_adapter)
                    orig_weight = torch.transpose(orig_weight, 0, 1)
                    orig_weight = torch.mm(butterfly_oft_mat, orig_weight)
                    orig_weight = torch.transpose(orig_weight, 0, 1)
                    orig_weight = orig_weight * boft_s

                    if not torch.isfinite(orig_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    self.base_layer.weight.data = orig_weight.contiguous()
                else:
                    butterfly_oft_mat, boft_s = self.get_delta_weight(active_adapter)
                    orig_weight = base_layer.weight.data.clone()
                    orig_weight = torch.transpose(orig_weight, 0, 1)
                    orig_weight = torch.mm(butterfly_oft_mat, orig_weight)
                    orig_weight = torch.transpose(orig_weight, 0, 1)
                    orig_weight = orig_weight * boft_s

                    self.base_layer.weight.data = orig_weight.contiguous()

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

                orig_weight = self.get_base_layer().weight.data.clone()
                orig_weight = torch.transpose(orig_weight, 0, 1)
                orig_weight = torch.mm(butterfly_oft_mat.t(), orig_weight)
                orig_weight = torch.transpose(orig_weight, 0, 1)

                self.get_base_layer().weight.data = orig_weight * (1 / boft_s)

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
        if self.fbd_cuda_available:
            block_diagonal_butterfly = FastBlockDiag.apply(orth_rotate_butterfly)
        else:
            orth_rotate_butterfly = orth_rotate_butterfly.squeeze(0)
            block_diagonal_butterfly = torch.block_diag(*torch.unbind(orth_rotate_butterfly))
            block_diagonal_butterfly = block_diagonal_butterfly.unsqueeze(0)

        boft_P = self.boft_P.to(block_diagonal_butterfly.device)
        butterfly_oft_mat_batch = torch.bmm(block_diagonal_butterfly, boft_P.permute(0, 2, 1))
        butterfly_oft_mat_batch = torch.bmm(boft_P, butterfly_oft_mat_batch)
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
            boft_rotation = torch.eye(self.in_features, device=x.device, dtype=previous_dtype)
            boft_scale = torch.ones((int(self.out_features), 1), device=x.device, dtype=previous_dtype)

            for active_adapter in self.active_adapters:
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
                if self.fbd_cuda_available:
                    block_diagonal_butterfly = FastBlockDiag.apply(orth_rotate_butterfly)
                else:
                    orth_rotate_butterfly = orth_rotate_butterfly.squeeze(0)
                    block_diagonal_butterfly = torch.block_diag(*torch.unbind(orth_rotate_butterfly))
                    block_diagonal_butterfly = block_diagonal_butterfly.unsqueeze(0)

                # The BOFT author's cayley_batch, dropout and FastBlockDiag ONLY return fp32 outputs.
                boft_P = self.boft_P.to(x)
                block_diagonal_butterfly = block_diagonal_butterfly.to(x)
                butterfly_oft_mat_batch = torch.bmm(block_diagonal_butterfly, boft_P.permute(0, 2, 1))
                butterfly_oft_mat_batch = torch.bmm(boft_P, butterfly_oft_mat_batch)
                butterfly_oft_mat = butterfly_oft_mat_batch[0]

                for i in range(1, butterfly_oft_mat_batch.shape[0]):
                    butterfly_oft_mat = butterfly_oft_mat_batch[i] @ butterfly_oft_mat

                boft_rotation = butterfly_oft_mat @ boft_rotation
                boft_scale = boft_s * boft_scale

            x = x.to(self.get_base_layer().weight.data.dtype)

            orig_weight = self.get_base_layer().weight.data
            orig_weight = torch.transpose(orig_weight, 0, 1)
            boft_rotation = boft_rotation.to(previous_dtype)
            orig_weight = orig_weight.to(previous_dtype)
            rotated_weight = torch.mm(boft_rotation, orig_weight)
            rotated_weight = torch.transpose(rotated_weight, 0, 1)

            scaled_rotated_weight = rotated_weight * boft_scale

            scaled_rotated_weight = scaled_rotated_weight.to(previous_dtype)
            if self.base_layer.bias is not None:
                self.base_layer.bias = self.base_layer.bias.to(previous_dtype)
            result = F.linear(input=x, weight=scaled_rotated_weight, bias=self.base_layer.bias)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "boft." + rep


class Conv2d(nn.Module, BOFTLayer):
    """
    BOFT implemented in a Conv2d layer.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        boft_block_size: int = 8,
        boft_block_num: int = 0,
        boft_n_butterfly_factor: int = 0,
        boft_dropout: float = 0.1,
        init_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        BOFTLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, boft_block_size, boft_block_num, boft_n_butterfly_factor, boft_dropout, init_weights
        )

    def update_layer(
        self, adapter_name, boft_block_size, boft_block_num, boft_n_butterfly_factor, boft_dropout, init_weights
    ):
        """
        Update the conv2d layer with trainable BOFT weights.
        """

        # Attempt to load the CUDA extension during model initialization
        if not get_fbd_cuda():
            self.fbd_cuda_available = False
            # If the CUDA extension is not available, set the butterfly factor to 1 to speed up the finetuning process
            boft_n_butterfly_factor = 1
        else:
            self.fbd_cuda_available = True

        # to be consistent with the paper notation
        boft_n_butterfly_factor = boft_n_butterfly_factor - 1
        if boft_n_butterfly_factor < 0:
            raise ValueError(
                f"You can only specify boft_n_butterfly_factor {boft_n_butterfly_factor+1} to be a positive integer number."
            )

        # Initialize the MultiplicativeDropoutLayer for boft_dropout > 0.0.
        if boft_dropout > 0.0:
            boft_dropout_layer = MultiplicativeDropoutLayer(p=boft_dropout)
        else:
            boft_dropout_layer = nn.Identity()
        self.boft_dropout.update(nn.ModuleDict({adapter_name: boft_dropout_layer}))

        # layer information from the base layer
        base_layer = self.get_base_layer()
        conv_filter_dim = self.in_features * base_layer.kernel_size[0] * base_layer.kernel_size[0]

        # Initialize the BOFT parameters.
        if not (boft_block_size != 0) ^ (boft_block_num != 0):
            raise ValueError(
                f"You can only specify either boft_block_size ({boft_block_size}) or boft_block_num ({boft_block_num}), but not both simultaneously, because boft_block_size x boft_block_num != in_features."
            )

        if boft_block_size == 0 and boft_block_num != 0:
            if conv_filter_dim % boft_block_num != 0:
                raise ValueError(
                    f"Convolutional kernel dimension ({conv_filter_dim}) must be divisible by boft_block_num ({boft_block_num})!"
                )

            if boft_n_butterfly_factor != 0:
                if boft_n_butterfly_factor > int(math.log2(boft_block_num)):
                    raise ValueError(
                        f"Invalid combination of boft_n_butterfly_factor ({boft_n_butterfly_factor+1}) and boft_block_num ({boft_block_num})!"
                    )
                if boft_block_num % (2**boft_n_butterfly_factor) != 0:
                    raise ValueError(
                        f"boft_block_num ({boft_block_num}) must be a multiple of 2 raised to the power of boft_n_butterfly_factor ({boft_n_butterfly_factor+1})!"
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
                        f"Invalid combination of convolutional kernel dimension ({conv_filter_dim}), boft_n_butterfly_factor ({boft_n_butterfly_factor+1}) and boft_block_size ({boft_block_size})!"
                    )
                if conv_filter_dim % (boft_block_size * (2**boft_n_butterfly_factor)) != 0:
                    raise ValueError(
                        f"Invalid combination of convolutional kernel dimension ({conv_filter_dim}), boft_n_butterfly_factor ({boft_n_butterfly_factor+1}) and boft_block_size ({boft_block_size})!"
                    )

            boft_block_num = int(conv_filter_dim // boft_block_size)

        else:
            raise ValueError("Unknown error!")

        # In OFT you can specify the number of blocks to be 1
        if boft_n_butterfly_factor != 0:
            if boft_block_num % 2 != 0:
                raise ValueError(f"boft_block_num ({boft_block_num}) must be an even number!")

            if boft_block_size % 2 != 0:
                raise ValueError(f"boft_block_size ({boft_block_size}) must be an even number!")

        # If there is no butterfly factor, then permutation matrix P will be an identity matrix.
        P = torch.empty((boft_n_butterfly_factor + 1, conv_filter_dim, conv_filter_dim))
        for i in range(boft_n_butterfly_factor + 1):
            perm = self.block_butterfly_perm(
                conv_filter_dim, int(boft_block_num / (2 ** (i))), int(boft_block_size / 2), boft_n_butterfly_factor
            )
            perm_mat = self.perm2mat(perm)
            P[i] = perm_mat

        self.register_buffer("boft_P", P, persistent=False)

        self.boft_R[adapter_name] = nn.Parameter(
            torch.zeros(boft_n_butterfly_factor + 1, boft_block_num, boft_block_size, boft_block_size)
        )
        self.boft_s[adapter_name] = nn.Parameter(torch.ones(1, int(self.out_features)))

        self.reset_boft_parameters(adapter_name, init_weights)

        # set the boft block size and number
        self.boft_block_size[adapter_name] = boft_block_size
        self.boft_block_num[adapter_name] = boft_block_num

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
            if active_adapter in self.boft_R.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weight = base_layer.weight.data.clone()
                    butterfly_oft_mat, boft_s = self.get_delta_weight(active_adapter)

                    orig_weight = orig_weight.view(
                        self.in_features * base_layer.kernel_size[0] * base_layer.kernel_size[0], self.out_features
                    )
                    orig_weight = torch.mm(butterfly_oft_mat, orig_weight)
                    orig_weight = orig_weight * boft_s
                    orig_weight = orig_weight.view(
                        self.out_features, self.in_features, base_layer.kernel_size[0], base_layer.kernel_size[0]
                    )

                    self.base_layer.weight.data = orig_weight.contiguous()
                else:
                    butterfly_oft_mat, boft_s = self.get_delta_weight(active_adapter)

                    orig_weight = base_layer.weight.data.clone()
                    orig_weight = orig_weight.view(
                        self.in_features * base_layer.kernel_size[0] * base_layer.kernel_size[0], self.out_features
                    )
                    orig_weight = torch.mm(butterfly_oft_mat, orig_weight)
                    orig_weight = orig_weight * boft_s
                    orig_weight = orig_weight.view(
                        self.out_features, self.in_features, base_layer.kernel_size[0], base_layer.kernel_size[0]
                    )

                    self.base_layer.weight.data = orig_weight.contiguous()

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

                orig_weight = self.get_base_layer().weight.data.clone()
                orig_weight = orig_weight.view(
                    self.in_features * self.get_base_layer().kernel_size[0] * self.get_base_layer().kernel_size[0],
                    self.out_features,
                )
                orig_weight = torch.mm(butterfly_oft_mat.t(), orig_weight)
                orig_weight = orig_weight * (1 / boft_s)
                orig_weight = orig_weight.view(
                    self.out_features,
                    self.in_features,
                    self.get_base_layer().kernel_size[0],
                    self.get_base_layer().kernel_size[0],
                )

                self.get_base_layer().weight.data = orig_weight

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
        if self.fbd_cuda_available:
            block_diagonal_butterfly = FastBlockDiag.apply(orth_rotate_butterfly)
        else:
            orth_rotate_butterfly = orth_rotate_butterfly.squeeze(0)
            block_diagonal_butterfly = torch.block_diag(*torch.unbind(orth_rotate_butterfly))
            block_diagonal_butterfly = block_diagonal_butterfly.unsqueeze(0)

        boft_P = self.boft_P.to(block_diagonal_butterfly.device)
        butterfly_oft_mat_batch = torch.bmm(block_diagonal_butterfly, boft_P.permute(0, 2, 1))
        butterfly_oft_mat_batch = torch.bmm(boft_P, butterfly_oft_mat_batch)
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
            boft_rotation = torch.eye(
                self.in_features * self.base_layer.kernel_size[0] * self.base_layer.kernel_size[0],
                device=x.device,
                dtype=x.dtype,
            )
            boft_scale = torch.ones((1, int(self.out_features)), device=x.device, dtype=x.dtype)

            for active_adapter in self.active_adapters:
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
                if self.fbd_cuda_available:
                    block_diagonal_butterfly = FastBlockDiag.apply(orth_rotate_butterfly)
                else:
                    orth_rotate_butterfly = orth_rotate_butterfly.squeeze(0)
                    block_diagonal_butterfly = torch.block_diag(*torch.unbind(orth_rotate_butterfly))
                    block_diagonal_butterfly = block_diagonal_butterfly.unsqueeze(0)

                boft_P = self.boft_P.to(x)
                block_diagonal_butterfly = block_diagonal_butterfly.to(x)
                butterfly_oft_mat_batch = torch.bmm(block_diagonal_butterfly, boft_P.permute(0, 2, 1))
                butterfly_oft_mat_batch = torch.bmm(boft_P, butterfly_oft_mat_batch)
                butterfly_oft_mat = butterfly_oft_mat_batch[0]

                for i in range(1, butterfly_oft_mat_batch.shape[0]):
                    butterfly_oft_mat = butterfly_oft_mat_batch[i] @ butterfly_oft_mat

                boft_rotation = butterfly_oft_mat @ boft_rotation
                boft_scale = boft_s * boft_scale

            x = x.to(self.base_layer.weight.data.dtype)

            orig_weight = self.base_layer.weight.data
            orig_weight = orig_weight.view(
                self.in_features * self.base_layer.kernel_size[0] * self.base_layer.kernel_size[0],
                self.out_features,
            )
            rotated_weight = torch.mm(boft_rotation, orig_weight)

            scaled_rotated_weight = rotated_weight * boft_scale

            scaled_rotated_weight = scaled_rotated_weight.view(
                self.out_features, self.in_features, self.base_layer.kernel_size[0], self.base_layer.kernel_size[0]
            )
            result = F.conv2d(
                input=x,
                weight=scaled_rotated_weight,
                bias=self.base_layer.bias,
                padding=self.base_layer.padding[0],
                stride=self.base_layer.stride[0],
            )

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "boft." + rep
