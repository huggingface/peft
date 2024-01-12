# coding=utf-8
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

import math
import warnings
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load, CUDA_HOME
from torch.autograd import Function

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import transpose

import os

os.environ["CC"] = "gcc"
os.environ["CXX"] = "gcc"
curr_dir = os.path.dirname(__file__)


class FastBlockDiag(Function):
    """
    Implements a custom autograd Function for a fast block diagonal operation using CUDA.

    This function is optimized for 4D tensors where the last two dimensions are equal,
    representing block diagonal matrices for efficient computation on CUDA devices.
    """        
    @staticmethod
    def forward(ctx, input):
        """
        The forward method for FastBlockDiag.

        Computes the block diagonal operation on the input tensor using a CUDA-optimized function.
        This method assumes that the input is a 4D tensor where the last two dimensions are equal,
        which represent the blocks to be diagonalized.

        Parameters:
        ctx: A context object that can be used to stash information for backward computation.
        input (Tensor): The input tensor of shape (N, D, H, H), where `N` is the batch size,
                        `D` represents one additional dimension (In BOFT, the number of BOFT blocks),
                        and `H` is the size of the square blocks along the last two dimensions
                        (In BOFT, the block size).

        Returns:
        Tensor: The resulting tensor after applying the block diagonal operation,
                will have the shape (N, DxH, DxH).
        """
        output = fbd_cuda.forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = fbd_cuda.backward(grad_output, input)[0]
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
        super(MultiplicativeDropoutLayer, self).__init__()
        self.p = p

    def forward(self, x):
        """
        The forward method for MultiplicativeDropoutLayer.

        Applies multiplicative dropout to the input tensor.
        Parameters:
        x (Tensor): The input tensor of shape (N, D, H, H), where `N` is the batch size, `D` represents
                    one additional dimension (In BOFT, the number of BOFT blocks), and `H` is the size
                    of the square blocks along the last two dimensions (In BOFT, the block size).
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

        Note, currently only support linear layer and convolutional layer, with further support for other layers to be added soon.

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
        
        global fbd_cuda
        fbd_cuda = load(
            name="fbd_cuda",
            sources=[f"{curr_dir}/fbd/fbd_cuda.cpp", f"{curr_dir}/fbd/fbd_cuda_kernel.cu"],
            verbose=True,
            # build_directory='/tmp/'  # for debugging
            )
        # extra_cuda_cflags = ['-std=c++14', '-ccbin=$$(which gcc-7)']) # cuda10.2 is not compatible with gcc9. Specify gcc 7
        import fbd_cuda

    def update_layer(
        self, adapter_name, boft_block_size, boft_block_num, boft_n_butterfly_factor, boft_dropout, init_boft_weights
    ):
        """
        Update the linear layer with trainable BOFT weights. Override for other layer types.
        """
        # to be consistent with the paper notation 
        boft_n_butterfly_factor = boft_n_butterfly_factor - 1
        if boft_n_butterfly_factor < 0:
            raise ValueError(
                f"You can only specify boft_n_butterfly_factor {boft_n_butterfly_factor} to be a positive integer number."
            )

        # Initialize the MultiplicativeDropoutLayer for boft_dropout > 0.0.
        if boft_dropout > 0.0:
            boft_dropout_layer = MultiplicativeDropoutLayer(p=boft_dropout)
        else:
            boft_dropout_layer = nn.Identity()
        self.boft_dropout.update(nn.ModuleDict({adapter_name: boft_dropout_layer}))

        # Initialize the BOFT parameters.
        if not (boft_block_size != 0) ^ (boft_block_num != 0):
            raise ValueError(
                f"You can only specify either boft_block_size ({boft_block_size}) or boft_block_num ({boft_block_num}), but not both simultaneously, because boft_block_size x boft_block_num != in_features."
            )

        if boft_block_size == 0 and boft_block_num != 0:
            if self.in_features % boft_block_num != 0:
                raise ValueError(
                    f"in_features ({self.in_features}) must be divisible by boft_block_num ({boft_block_num})!"
                )

            if boft_n_butterfly_factor != 0:
                if boft_n_butterfly_factor > int(math.log2(boft_block_num)):
                    raise ValueError(
                        f"Invalid combination of boft_n_butterfly_factor ({boft_n_butterfly_factor}) and boft_block_num ({boft_block_num})!"
                    )
                if boft_block_num % (2**boft_n_butterfly_factor) != 0:
                    raise ValueError(
                        f"boft_block_num ({boft_block_num}) must be a multiple of 2 raised to the power of boft_n_butterfly_factor ({boft_n_butterfly_factor})!"
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
                        f"Invalid combination of in_features ({self.in_features}), boft_n_butterfly_factor ({boft_n_butterfly_factor}) and boft_block_size ({boft_block_size})!"
                    )
                if self.in_features % (boft_block_size * (2**boft_n_butterfly_factor)) != 0:
                    raise ValueError(
                        f"Invalid combination of in_features ({self.in_features}), boft_n_butterfly_factor ({boft_n_butterfly_factor}) and boft_block_size ({boft_block_size})!"
                    )

            boft_block_num = int(self.in_features // boft_block_size)

        else:
            raise ValueError("Unknown error!")

        # In OFT you can specify the number of blocks to be 1
        if boft_n_butterfly_factor != 0:
            if boft_block_num % 2 != 0:
                raise ValueError(f"boft_block_num ({boft_block_num}) must be an even number!")

            if boft_block_size % 2 != 0:
                raise ValueError(f"boft_block_size ({boft_block_size}) must be an even number!")

        # If there is no butterfly factor, then permutation matrix P will be an identity matrix.
        P = torch.empty((boft_n_butterfly_factor + 1, self.in_features, self.in_features))
        for i in range((boft_n_butterfly_factor + 1)):
            perm = self.block_butterfly_perm(
                self.in_features, int(boft_block_num / (2 ** (i))), int(boft_block_size / 2), boft_n_butterfly_factor
            )
            perm_mat = self.perm2mat(perm)
            P[i] = perm_mat

        self.register_buffer("boft_P", P)

        self.boft_R[adapter_name] = nn.Parameter(
            torch.zeros(boft_n_butterfly_factor + 1, boft_block_num, boft_block_size, boft_block_size)
        )
        self.boft_s[adapter_name] = nn.Parameter(torch.ones(int(self.out_features), 1))

        if init_boft_weights:
            self.reset_boft_parameters(adapter_name, init_boft_weights)

        weight = getattr(self, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

        # set the boft block size and number
        self.boft_block_size[adapter_name] = boft_block_size
        self.boft_block_num[adapter_name] = boft_block_num

    def reset_boft_parameters(self, adapter_name, init_boft_weights):
        """
        Reset the BOFT parameters.
        """
        if adapter_name in self.boft_R.keys():
            if init_boft_weights is True:
                # initialize R to zero
                nn.init.zeros_(self.boft_R[adapter_name])
                nn.init.ones_(self.boft_s[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_boft_weights=}")

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
        init_boft_weights: Union[bool, str] = True,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        BOFTLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name

        self.update_layer(
            adapter_name, boft_block_size, boft_block_num, boft_n_butterfly_factor, boft_dropout, init_boft_weights
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
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
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )

        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in self.active_adapters:
            if active_adapter in self.boft_R.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    butterfly_oft_mat, boft_s = self.get_delta_weight(active_adapter)
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = torch.mm(butterfly_oft_mat, orig_weights)
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = orig_weights * boft_s

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    self.base_layer.weight.data = orig_weights
                else:
                    butterfly_oft_mat, boft_s = self.get_delta_weight(active_adapter)
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = torch.mm(butterfly_oft_mat, orig_weights)
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    self.base_layer.weight.data = orig_weights * boft_s

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
                rotated_weight = torch.mm(butterfly_oft_mat.t(), orig_weight)
                rotated_weight = torch.transpose(rotated_weight, 0, 1)

                self.get_base_layer().weight.data = rotated_weight * (1 / boft_s)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        # device = self.boft_R[adapter].device
        # dtype = self.boft_R[adapter].dtype

        boft_R = self.boft_R[adapter]
        boft_s = self.boft_s[adapter]

        N, D, H, _ = boft_R.shape
        boft_R = boft_R.view(N * D, H, H)
        orth_rotate_butterfly = self.cayley_batch(boft_R)
        orth_rotate_butterfly = orth_rotate_butterfly.view(N, D, H, H)
        block_diagonal_butterfly = FastBlockDiag.apply(orth_rotate_butterfly)

        butterfly_oft_mat_batch = torch.bmm(block_diagonal_butterfly, self.boft_P.permute(0, 2, 1))
        butterfly_oft_mat_batch = torch.bmm(self.boft_P, butterfly_oft_mat_batch)
        butterfly_oft_mat = butterfly_oft_mat_batch[0]

        for i in range(1, butterfly_oft_mat_batch.shape[0]):
            butterfly_oft_mat = butterfly_oft_mat_batch[i] @ butterfly_oft_mat

        return butterfly_oft_mat, boft_s

    def cayley_batch(self, data):
        """
        Perform the Cayley parametrization on a batch of skew-symmetric matrices.
        Args:
            data: A batch of skew-symmetric matrices of shape (b, r, c).
        """
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.bmm(I - skew, torch.inverse(I + skew))
        # Q = torch.linalg.solve(I + skew, I - skew, left=False)

        return Q

    def angle2rot(self, alphas):
        """
        Convert the angle to rotation matrix.
        Only applicable for BOFT block size 2.
        """
        c = torch.cos(alphas)
        s = torch.sin(alphas)
        rot_mats = torch.cat([c, -s, s, c], dim=-1).view(alphas.shape[0], alphas.shape[1], 2, 2)
        return rot_mats

    def is_orthogonal(self, R, eps=1e-3):
        """
        Check if the matrix is orthogonal.
        """
        R = R.float()
        with torch.no_grad():
            RtR = torch.matmul(R.t(), R)
            diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return torch.all(diff < eps)

    def is_identity_matrix(self, tensor):
        """
        Check if the matrix is identity.
        """
        if not torch.is_tensor(tensor):
            raise TypeError("Input must be a PyTorch tensor.")
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            return False
        identity = torch.eye(tensor.shape[0], device=tensor.device)
        return torch.all(torch.eq(tensor, identity))

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result_initialized = False  # Initialize the flag
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
                block_diagonal_butterfly = FastBlockDiag.apply(orth_rotate_butterfly)

                butterfly_oft_mat_batch = torch.bmm(block_diagonal_butterfly, self.boft_P.permute(0, 2, 1))
                butterfly_oft_mat_batch = torch.bmm(self.boft_P, butterfly_oft_mat_batch)
                butterfly_oft_mat = butterfly_oft_mat_batch[0]

                for i in range(1, butterfly_oft_mat_batch.shape[0]):
                    butterfly_oft_mat = butterfly_oft_mat_batch[i] @ butterfly_oft_mat

                x = x.to(boft_R.data.dtype)

                orig_weight = self.base_layer.weight.data
                orig_weight = torch.transpose(orig_weight, 0, 1)
                rotated_weight = torch.mm(butterfly_oft_mat, orig_weight)
                rotated_weight = torch.transpose(rotated_weight, 0, 1)

                scaled_rotated_weight = rotated_weight * boft_s

                result = F.linear(input=x, weight=scaled_rotated_weight, bias=self.base_layer.bias)
                result_initialized = True

            # handle the case when there are no activate adapter in BOFT
            if not result_initialized:
                result = self.base_layer(x, *args, **kwargs)

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
        init_boft_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        BOFTLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, boft_block_size, boft_block_num, boft_n_butterfly_factor, boft_dropout, init_boft_weights
        )

    def update_layer(
        self, adapter_name, boft_block_size, boft_block_num, boft_n_butterfly_factor, boft_dropout, init_boft_weights
    ):
        """
        Update the linear layer with trainable BOFT weights.
        """
        # to be consistent with the paper notation 
        boft_n_butterfly_factor = boft_n_butterfly_factor - 1
        if boft_n_butterfly_factor < 0:
            raise ValueError(
                f"You can only specify boft_n_butterfly_factor {boft_n_butterfly_factor} to be a positive integer number."
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
                        f"Invalid combination of boft_n_butterfly_factor ({boft_n_butterfly_factor}) and boft_block_num ({boft_block_num})!"
                    )
                if boft_block_num % (2**boft_n_butterfly_factor) != 0:
                    raise ValueError(
                        f"boft_block_num ({boft_block_num}) must be a multiple of 2 raised to the power of boft_n_butterfly_factor ({boft_n_butterfly_factor})!"
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
                        f"Invalid combination of convolutional kernel dimension ({conv_filter_dim}), boft_n_butterfly_factor ({boft_n_butterfly_factor}) and boft_block_size ({boft_block_size})!"
                    )
                if conv_filter_dim % (boft_block_size * (2**boft_n_butterfly_factor)) != 0:
                    raise ValueError(
                        f"Invalid combination of convolutional kernel dimension ({conv_filter_dim}), boft_n_butterfly_factor ({boft_n_butterfly_factor}) and boft_block_size ({boft_block_size})!"
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
        for i in range((boft_n_butterfly_factor + 1)):
            perm = self.block_butterfly_perm(
                conv_filter_dim, int(boft_block_num / (2 ** (i))), int(boft_block_size / 2), boft_n_butterfly_factor
            )
            perm_mat = self.perm2mat(perm)
            P[i] = perm_mat

        self.register_buffer("boft_P", P)

        self.boft_R[adapter_name] = nn.Parameter(
            torch.zeros(boft_n_butterfly_factor + 1, boft_block_num, boft_block_size, boft_block_size)
        )
        self.boft_s[adapter_name] = nn.Parameter(torch.ones(int(self.out_features), 1))

        if init_boft_weights:
            self.reset_boft_parameters(adapter_name, init_boft_weights)

        weight = getattr(self, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

        # set the boft block size and number
        self.boft_block_size[adapter_name] = boft_block_size
        self.boft_block_num[adapter_name] = boft_block_num

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
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
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )

        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in self.active_adapters:
            if active_adapter in self.boft_R.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    butterfly_oft_mat, boft_s = self.get_delta_weight(active_adapter)
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = torch.mm(butterfly_oft_mat, orig_weights)
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = orig_weights * boft_s

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    self.base_layer.weight.data = orig_weights
                else:
                    butterfly_oft_mat, boft_s = self.get_delta_weight(active_adapter)
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = torch.mm(butterfly_oft_mat, orig_weights)
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    self.base_layer.weight.data = orig_weights * boft_s

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
                rotated_weight = torch.mm(butterfly_oft_mat.t(), orig_weight)
                rotated_weight = torch.transpose(rotated_weight, 0, 1)

                self.get_base_layer().weight.data = rotated_weight * (1 / boft_s)


    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        # device = self.boft_R[adapter].device
        # dtype = self.boft_R[adapter].dtype

        boft_R = self.boft_R[adapter]
        boft_s = self.boft_s[adapter]

        N, D, H, _ = boft_R.shape
        boft_R = boft_R.view(N * D, H, H)
        orth_rotate_butterfly = self.cayley_batch(boft_R)
        orth_rotate_butterfly = orth_rotate_butterfly.view(N, D, H, H)
        block_diagonal_butterfly = FastBlockDiag.apply(orth_rotate_butterfly)

        butterfly_oft_mat_batch = torch.bmm(block_diagonal_butterfly, self.boft_P.permute(0, 2, 1))
        butterfly_oft_mat_batch = torch.bmm(self.boft_P, butterfly_oft_mat_batch)
        butterfly_oft_mat = butterfly_oft_mat_batch[0]

        for i in range(1, butterfly_oft_mat_batch.shape[0]):
            butterfly_oft_mat = butterfly_oft_mat_batch[i] @ butterfly_oft_mat

        return butterfly_oft_mat, boft_s


    def cayley_batch(self, data):
        """
        Perform the Cayley parametrization on a batch of skew-symmetric matrices.
        Args:
            data: A batch of skew-symmetric matrices of shape (b, r, c).
        """
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.bmm(I - skew, torch.inverse(I + skew))
        # Q = torch.linalg.solve(I + skew, I - skew, left=False)

        return Q

    def angle2rot(self, alphas):
        """
        Convert the angle to rotation matrix.
        Only applicable for BOFT block size 2.
        """
        c = torch.cos(alphas)
        s = torch.sin(alphas)
        rot_mats = torch.cat([c, -s, s, c], dim=-1).view(alphas.shape[0], alphas.shape[1], 2, 2)
        return rot_mats

    def is_orthogonal(self, R, eps=1e-3):
        """
        Check if the matrix is orthogonal.
        """
        R = R.float()
        with torch.no_grad():
            RtR = torch.matmul(R.t(), R)
            diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return torch.all(diff < eps)

    def is_identity_matrix(self, tensor):
        """
        Check if the matrix is identity.
        """
        if not torch.is_tensor(tensor):
            raise TypeError("Input must be a PyTorch tensor.")
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            return False
        identity = torch.eye(tensor.shape[0], device=tensor.device)
        return torch.all(torch.eq(tensor, identity))

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result_initialized = False  # Initialize the flag
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
                block_diagonal_butterfly = FastBlockDiag.apply(orth_rotate_butterfly)

                butterfly_oft_mat_batch = torch.bmm(block_diagonal_butterfly, self.boft_P.permute(0, 2, 1))
                butterfly_oft_mat_batch = torch.bmm(self.boft_P, butterfly_oft_mat_batch)
                butterfly_oft_mat = butterfly_oft_mat_batch[0]

                for i in range(1, butterfly_oft_mat_batch.shape[0]):
                    butterfly_oft_mat = butterfly_oft_mat_batch[i] @ butterfly_oft_mat

                x = x.to(boft_R.data.dtype)

                orig_weight = self.base_layer.weight.data
                orig_weight = torch.transpose(orig_weight, 0, 1)
                rotated_weight = torch.mm(butterfly_oft_mat, orig_weight)
                rotated_weight = torch.transpose(rotated_weight, 0, 1)

                scaled_rotated_weight = rotated_weight * boft_s
                
                scaled_rotated_weight = scaled_rotated_weight.view(self.out_features, self.in_features, self.base_layer.kernel_size[0], self.base_layer.kernel_size[0])
                result = F.conv2d(input=x, filters=scaled_rotated_weight, bias=self.base_layer.bias, padding=self.base_layer.padding, stride=self.base_layer.stride)
                result_initialized = True

            # handle the case when there are no activate adapter in BOFT
            if not result_initialized:
                result = self.base_layer(x, *args, **kwargs)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "boft." + rep
