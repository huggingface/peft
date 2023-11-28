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
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load, CUDA_HOME
from torch.autograd import Function

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import transpose

os.environ["CC"] = "gcc"
os.environ["CXX"] = "gcc"
curr_dir = os.path.dirname(__file__)
fbd_cuda = \
    load(name='fbd_cuda', 
        sources=[f'{curr_dir}/fbd/fbd_cuda.cpp', f'{curr_dir}/fbd/fbd_cuda_kernel.cu'], verbose=True,
        build_directory='/tmp/'
        )
        # extra_cuda_cflags = ['-std=c++14', '-ccbin=$$(which gcc-7)']) # cuda10.2 is not compatible with gcc9. Specify gcc 7 

import fbd_cuda

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
        input, = ctx.saved_tensors
        grad_input = fbd_cuda.backward(
            grad_output, input)[0]
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
            assert x.shape[-1] == x.shape[-2], "The last two dimensions of input should be the same!"

            N, D, H, _ = x.shape

            # Randomly select one from N
            n_random = torch.randint(0, N, (1,)).item()

            # Create a mask with 1s for matrices to be replaced with identity and 0s otherwise
            num_to_replace = int(self.p * D)
            num_zeros = D - num_to_replace

            # Generate a flat tensor with desired number of 1s and 0s
            mask = torch.cat([torch.ones(num_to_replace, device=x.device), torch.zeros(num_zeros, device=x.device)])

            # Shuffle and reshape the mask
            mask = mask[torch.randperm(D)].view(1, Z, 1, 1)

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
    def __init__(self, in_features: int, out_features: int, **kwargs):
        """
        Initializes the BOFT layer.

        Note, currently only support linear layer and convolutional layer, with further support for other layers to be added soon.

        Parameters:
        in_features (int): The dimension of the input tensor.
        out_features (int): The dimension of the output tensor.
        """
        self.boft_block_size = {}
        self.boft_block_num = {}
        self.boft_dropout = nn.ModuleDict({})
        self.boft_R = nn.ParameterDict({})
        self.boft_s = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def _init_empty_weights(self, cls, *args, **kwargs) -> None:
        # A helper method that allows to initialize the layer of the given class without spending time to initialize the
        # model weights. The implementation is inspired by
        # https://pytorch.org/docs/stable/generated/torch.nn.utils.skip_init.html but this function cannot be used
        # directly.
        # Instead of this approach, it would be possible to bypass the __init__ of the class but that runs the risk of
        # omitting important logic inside that __init__.
        kwargs = kwargs.copy()
        final_device = kwargs.pop("device", "cpu")
        cls.__init__(self, *args, device="meta", **kwargs)
        self.to_empty(device=final_device)

    def update_layer(self, adapter_name, boft_block_size, boft_block_num, boft_dropout, init_boft_weights):
        """
        Update the linear layer with trainable BOFT weights.
        """
        # Initialize the MultiplicativeDropoutLayer for boft_dropout > 0.0.
        if boft_dropout > 0.0:
            boft_dropout_layer = MultiplicativeDropoutLayer(p=boft_dropout)
        else:
            boft_dropout_layer = nn.Identity()
        self.boft_dropout.update(nn.ModuleDict({adapter_name: boft_dropout_layer}))

        # Initialize the BOFT parameters.
        assert (boft_block_size != 0) ^ (boft_block_num != 0), "You can only specify either boft_block_size or boft_block_num, but not both simultaneously, because boft_block_size x boft_block_num != in_features."
        assert boft_block_size % 2 == 0, "You must set the boft_block_size to be an even number!"

        if boft_block_size == 0 and boft_block_num != 0:
            assert self.in_features % boft_block_num == 0, "in_features must be divisible by boft_block_num"
            if self.kwargs["boft_n_butterfly_factor"] != 0:
                assert self.kwargs["boft_n_butterfly_factor"] <= int(math.log2(boft_block_num)), "invalid combination of boft_n_butterfly_factor and boft_block_num"
                assert boft_block_num % (2**self.kwargs["boft_n_butterfly_factor"]) == 0, "boft_block_num must be a power of 2"
            boft_block_size = int(self.in_features // boft_block_num)

        elif boft_block_size != 0 and boft_block_num == 0:
            assert self.in_features % boft_block_size == 0, "in_features must be divisible by boft_block_size"
            if self.kwargs["boft_n_butterfly_factor"] != 0:
                assert self.in_features >= boft_block_size * (2**self.kwargs["boft_n_butterfly_factor"]), "invalid combination of boft_n_butterfly_factor and boft_block_size"
                assert self.in_features % (boft_block_size * (2**self.kwargs["boft_n_butterfly_factor"])) == 0, "invalid combination of boft_n_butterfly_factor and boft_block_size"
            boft_block_num = int(self.in_features // boft_block_size)

        else:
            print('Unknown error!')
            sys.exit()

        # If there is no butterfly factor, then permutation matrix P will be an identity matrix.
        P = torch.empty((self.kwargs["boft_n_butterfly_factor"]+1, self.in_features, self.in_features))
        for i in range((self.kwargs["boft_n_butterfly_factor"]+1)):
            perm = self.block_butterfly_perm(self.in_features, int(boft_block_num/(2**(i))), int(boft_block_size / 2))
            perm_mat = self.perm2mat(perm)
            P[i] = perm_mat

        self.register_buffer('boft_P', P)

        self.boft_R[adapter_name] = nn.Parameter(torch.zeros(self.kwargs["boft_n_butterfly_factor"]+1, boft_block_num, boft_block_size, boft_block_size))
        self.boft_s[adapter_name] = nn.Parameter(torch.ones(int(self.out_features), 1))

        if init_boft_weights:
            self.reset_boft_parameters(adapter_name)

        weight = getattr(self, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

        self.boft_block_size[adapter_name] = boft_block_size
        self.boft_block_num[adapter_name] = boft_block_num

    def reset_boft_parameters(self, adapter_name):
        """
        Reset the BOFT parameters.
        """
        if adapter_name in self.boft_R.keys():
            # initialize R to zero   
            nn.init.zeros_(self.boft_R[adapter_name])
            nn.init.ones_(self.boft_s[adapter_name])
            nn.init.ones_(self.boft_b[adapter_name])

    def perm2mat(self, indices):
        """
        Convert permutation indices to permutation matrix.
        """
        # Number of indices determines the size of the square matrix
        n = len(indices)
        
        # Initialize a matrix of zeros
        perm_mat = torch.zeros((n, n))
        
        # Set the 1s according to the indices
        for i, idx in enumerate(indices):
            perm_mat[i, idx] = 1
        
        return perm_mat

    def block_butterfly_perm(self, n, b, r=3):
        """
        Define the permutation matrix for the block butterfly permutation.

        Args:
        n: size of the permutation matrix
        b: desired number of blocks after multiplying with the permutation matrix
        r: base block size of the block diagonal matrix, e.g. 2x2, 3x3, 5x5 etc.
        """

        assert b * r * 2 <= n, "Invalid number of blocks!"

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
                sorted_order[int(i*r):int(i*r+r)] = initial_order[int(pos*r):int(pos*r+r)]
            return sorted_order

        sorted_order = sort_block(block_size, r)

        for i in range(0, n, block_size):
            block_end = i + block_size
            tmp_indices = indices[i:block_end]
            indices[i:block_end] = tmp_indices[sorted_order]
        return indices


class Linear(nn.Linear, BOFTLayer):
    """
    BOFT implemented in a dense layer.
    """
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        boft_dropout: float = 0.1,
        boft_block_size: int = 8,
        boft_block_num: int = 0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        boft_n_butterfly_factor: int = 0,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        init_boft_weights = kwargs.pop("init_boft_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        BOFTLayer.__init__(self, in_features=in_features, out_features=out_features, boft_n_butterfly_factor=boft_n_butterfly_factor)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        self.boft_n_butterfly_factor = boft_n_butterfly_factor

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, boft_block_size, boft_block_num, boft_dropout, init_boft_weights)
        self.active_adapter = adapter_name
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )
        for active_adapter in self.active_adapters:
            if active_adapter in self.boft_R.keys():
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = self.weight.data.clone()
                    butterfly_oft_mat, boft_s = self.get_delta_weight(active_adapter)
                    orig_weight = torch.transpose(orig_weight, 0, 1)
                    orig_weight = torch.mm(butterfly_oft_mat, orig_weight)
                    orig_weight = torch.transpose(orig_weight, 0, 1)
                    orig_weight = orig_weight * boft_s

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    self.weight.data = orig_weights
                else:
                    butterfly_oft_mat, boft_s = self.get_delta_weight(active_adapter)
                    self.weight.data = torch.transpose(self.weight.data, 0, 1)
                    self.weight.data = torch.mm(butterfly_oft_mat, self.weight.data)
                    self.weight.data = torch.transpose(self.weight.data, 0, 1)
                    self.weight.data = self.weight.data * boft_s
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        Delete the BOFT adapter weights with the base model.
        """
        if self.active_adapter not in self.boft_R.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.boft_block_size[self.active_adapter] > 0 and self.boft_block_num[self.active_adapter] > 0:
            # self.weight.data -= self.get_delta_weight(self.active_adapter)
            orig_weight = self.weight.data 
            butterfly_oft_mat, boft_s, boft_b = self.get_delta_weight(self.active_adapter)

            orig_weight = torch.transpose(orig_weight, 0, 1)
            rotated_weight = torch.mm(butterfly_oft_mat.t(), orig_weight)
            rotated_weight = torch.transpose(rotated_weight, 0, 1) 

            self.weight.data = rotated_weight * (1 / boft_s)
            self.bias.data = self.bias.data / boft_b
            self.merged = False

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        boft_R = self.boft_R[adapter]
        boft_s = self.boft_s[adapter]

        N, Z, b, _ = boft_R.shape
        boft_R = boft_R.view(N * Z, b, b)
        orth_rotate_butterfly = self.cayley_batch(boft_R)
        orth_rotate_butterfly = orth_rotate_butterfly.view(N, Z, b, b)
        block_diagonal_butterfly = FastBlockDiag.apply(orth_rotate_butterfly)

        butterfly_oft_mat_batch = torch.bmm(block_diagonal_butterfly, self.boft_P.permute(0, 2, 1))
        butterfly_oft_mat_batch = torch.bmm(self.boft_P, butterfly_oft_mat_batch)
        butterfly_oft_mat = butterfly_oft_mat_batch[0]

        for i in range(1, butterfly_oft_mat_batch.shape[0]):
            butterfly_oft_mat = butterfly_oft_mat_batch[i] @ butterfly_oft_mat

        return butterfly_oft_mat, boft_s

    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._linear(x)
        elif self.merged:
            result = self._linear(x)
        else:
            boft_R = self.boft_R[self.active_adapter]
            boft_s = self.boft_s[self.active_adapter]
            dropout = self.boft_dropout[self.active_adapter]

            N, Z, b, _ = boft_R.shape
            boft_R = boft_R.view(N * Z, b, b)
            orth_rotate_butterfly = self.cayley_batch(boft_R)
            orth_rotate_butterfly = orth_rotate_butterfly.view(N, Z, b, b)
            orth_rotate_butterfly = dropout(orth_rotate_butterfly)
            block_diagonal_butterfly = FastBlockDiag.apply(orth_rotate_butterfly)

            butterfly_oft_mat_batch = torch.bmm(block_diagonal_butterfly, self.boft_P.permute(0, 2, 1))
            butterfly_oft_mat_batch = torch.bmm(self.boft_P, butterfly_oft_mat_batch)
            butterfly_oft_mat = butterfly_oft_mat_batch[0]

            for i in range(1, butterfly_oft_mat_batch.shape[0]):
                butterfly_oft_mat = butterfly_oft_mat_batch[i] @ butterfly_oft_mat

            x = x.to(boft_R.data.dtype)
            
            orig_weight = self.weight.data
            orig_weight = torch.transpose(orig_weight, 0, 1)
            rotated_weight = torch.mm(butterfly_oft_mat, orig_weight)
            rotated_weight = torch.transpose(rotated_weight, 0, 1)

            scaled_rotated_weight = rotated_weight * boft_s

            result = F.linear(input=x, weight=scaled_rotated_weight, bias=self.bias.data)

        result = result.to(previous_dtype)
        return result