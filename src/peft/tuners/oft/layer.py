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
from typing import Any, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge



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
                    one additional dimension (In OFT, the number of OFT blocks), and `H` is the size of the square
                    blocks along the last two dimensions (In OFT, the block size).
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


class OFTLayer(BaseTunerLayer):
    """
    Implements the OFT layer.
    """

    # All names of layers that may contain adapter weights
    adapter_layer_names = ("oft_r", "oft_s")
    # other_param_names is defined on parent class
    other_param_names = ("r", "oft_block_size", "oft_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        """
        Initializes the OFT layer.

        Note, currently only support linear layer and convolutional layer, with further support for other layers to be
        added soon.

        Parameters:
        base_layer: the pretrained model layer
        """
        self.base_layer = base_layer
        # OFT info
        self.oft_r = nn.ParameterDict({})
        self.oft_s = nn.ParameterDict({})
        self.r = {}
        self.oft_block_size = {}
        self.oft_dropout = nn.ModuleDict({})
        self.coft = {}
        self.eps = {}
        self.block_share = {}
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

    @property
    def _available_adapters(self) -> Set[str]:
        return {*self.oft_r}

    def create_adapter_parameters(self, adapter_name: str, r: int, shape: Tuple[int, ...], block_share: bool):
        if block_share:
            self.oft_r[adapter_name] = nn.Parameter(torch.empty(1, math.ceil(shape[0] / r), math.ceil(shape[0] / r)))
        else:
            self.oft_r[adapter_name] = nn.Parameter(torch.empty(r, math.ceil(shape[0] / r), math.ceil(shape[0] / r)))

    def reset_oft_parameters(self, adapter_name, init_weights):
        """
        Reset the OFT parameters.
        """
        if init_weights is False:
            nn.init.normal_(self.oft_r[adapter_name], mean=0.0, std=0.1)
            nn.init.normal_(self.oft_s[adapter_name], mean=1.0, std=0.1)
            return

        if adapter_name in self.oft_r.keys():
            if init_weights is True:
                # initialize R to zero
                nn.init.zeros_(self.oft_r[adapter_name])
                nn.init.ones_(self.oft_s[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_weights=}")


    '''
    def update_layer(
        self,
        adapter_name: str,
        r: int,
        module_dropout: float,
        init_weights: bool,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        **kwargs,
    ) -> None:
    '''

    def update_layer(
        self, adapter_name, r, oft_block_size, module_dropout, coft, eps, block_share, init_weights
    ):
        """
        Update the linear layer with trainable OFT weights. Override for other layer types.
        """ 
        """Internal function to create oft adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            oft_block_size (`int`): The block size for added adapter.
            module_dropout (`float`): The multiplicative dropout probability for disabling adapter blocks during training.
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
            if self.in_features % oft_block_size != 0:
                raise ValueError(f"Input features ({self.in_features}) should be divisible by `oft_block_size` ({oft_block_size})")
            r = int(self.in_features // oft_block_size)
        elif r != 0 and oft_block_size == 0:
            if self.in_features % r != 0:
                raise ValueError(f"Input features ({self.in_features}) should be divisible by `r` ({r})!")
            oft_block_size = int(self.in_features // r)
        elif r != 0 and oft_block_size != 0:
            raise ValueError(f"You can only specify either r ({r}) or oft_block_size ({oft_block_size}), but not both simultaneously.")
        else:
            raise ValueError(f"Either `r` or `oft_block_size` must be non-zero. Currently, r = {r} and oft_block_size = {oft_block_size}.")
                
        self.module_dropout[adapter_name] = module_dropout
        self.coft[adapter_name] = coft
        self.block_share[adapter_name] = block_share

        # Determine shape of OFT weights
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            shape = tuple(base_layer.weight.shape)
        elif isinstance(base_layer, nn.Conv2d):
            shape = (
                base_layer.out_channels,
                base_layer.in_channels * base_layer.kernel_size[0] * base_layer.kernel_size[1],
            )
        else:
            raise TypeError(f"OFT is not implemented for base layers of type {type(base_layer).__name__}")

        self.eps[adapter_name] = eps * math.ceil(shape[0] / r) * math.ceil(shape[0] / r)

        print(self.in_features, r)
        exit()

        # Create weights with provided shape
        if block_share:
            self.oft_r[adapter_name] = nn.Parameter(torch.empty(1, math.ceil(self.in_features / r), math.ceil(self.in_features / r)))
        else:
            self.oft_r[adapter_name] = nn.Parameter(torch.empty(r, math.ceil(self.in_features / r), math.ceil(self.in_features / r)))
        self.oft_s[adapter_name] = nn.Parameter(torch.empty(int(self.out_features), 1))

        # Initialize weights
        self.reset_oft_parameters(adapter_name, init_weights)

        # set oft r and block size
        self.r[adapter_name] = r
        self.oft_block_size[adapter_name] = oft_block_size

        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def unscale_layer(self, scale=None) -> None:
        # scale is not used
        pass

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
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

                orig_weights = base_layer.weight.data
                if isinstance(base_layer, nn.Linear):
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                elif isinstance(base_layer, nn.Conv2d):
                    orig_weights = orig_weights.view(
                        [
                            base_layer.out_channels,
                            base_layer.in_channels * base_layer.kernel_size[0] * base_layer.kernel_size[1],
                        ]
                    )
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                delta_weight = self.get_delta_weight(active_adapter)
                if orig_weights.shape[1] != delta_weight.shape[1]:
                    # when in channels is not divisible by r
                    delta_weight = delta_weight[: orig_weights.shape[1], : orig_weights.shape[1]]
                new_weights = torch.mm(orig_weights, delta_weight)
                if isinstance(base_layer, nn.Linear):
                    new_weights = torch.transpose(new_weights, 0, 1)
                elif isinstance(base_layer, nn.Conv2d):
                    new_weights = torch.transpose(new_weights, 0, 1)
                    new_weights = new_weights.view(
                        [
                            base_layer.out_channels,
                            base_layer.in_channels,
                            base_layer.kernel_size[0],
                            base_layer.kernel_size[1],
                        ]
                    )

                if safe_merge and not torch.isfinite(new_weights).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )

                base_layer.weight.data = new_weights
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
                base_layer = self.get_base_layer()
                new_weights = base_layer.weight.data
                if isinstance(base_layer, nn.Linear):
                    new_weights = torch.transpose(new_weights, 0, 1)
                elif isinstance(base_layer, nn.Conv2d):
                    new_weights = new_weights.view(
                        [
                            base_layer.out_channels,
                            base_layer.in_channels * base_layer.kernel_size[0] * base_layer.kernel_size[1],
                        ]
                    )
                    new_weights = torch.transpose(new_weights, 0, 1)
                delta_weight = self.get_delta_weight(active_adapter)
                if new_weights.shape[1] != delta_weight.shape[1]:
                    # when in channels is not divisible by r
                    delta_weight = delta_weight[: new_weights.shape[1], : new_weights.shape[1]]
                delta_inv = torch.inverse(delta_weight)
                orig_weights = torch.mm(new_weights, delta_inv)

                if isinstance(base_layer, nn.Linear):
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                elif isinstance(base_layer, nn.Conv2d):
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = orig_weights.reshape(
                        [
                            base_layer.out_channels,
                            base_layer.in_channels,
                            base_layer.kernel_size[0],
                            base_layer.kernel_size[1],
                        ]
                    )
                base_layer.weight.data = orig_weights

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        rank = self.r[adapter_name]
        coft = self.coft[adapter_name]
        eps = self.eps[adapter_name]
        opt_r = self.oft_r[adapter_name]

        if coft:
            with torch.no_grad():
                opt_r.copy_(self._project_batch(opt_r, eps=eps))

        orth_rotate = self._cayley_batch(opt_r)
        weight = self._block_diagonal(orth_rotate, rank)

        return weight

    # Copied from https://github.com/Zeju1997/oft/blob/84cebb965df69781e3d9c3c875f5980b421eaf24/oft-control/oft.py#L144
    def _cayley_batch(self, data: torch.Tensor) -> torch.Tensor:
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)  # noqa: E741

        # Perform the Cayley parametrization
        Q = torch.bmm(I - skew, torch.inverse(I + skew))

        return Q

    # Copied from https://github.com/Zeju1997/oft/blob/84cebb965df69781e3d9c3c875f5980b421eaf24/oft-control/oft.py#L155
    def _block_diagonal(self, oft_r: torch.Tensor, rank: int) -> torch.Tensor:
        if oft_r.shape[0] == 1:
            # block share
            blocks = [oft_r[0, ...] for i in range(rank)]
        else:
            blocks = [oft_r[i, ...] for i in range(rank)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

    # Copied from https://github.com/Zeju1997/oft/blob/84cebb965df69781e3d9c3c875f5980b421eaf24/oft-control/oft.py#L52
    def _project_batch(self, oft_r, eps=1e-5):
        # scaling factor for each of the smaller block matrix
        eps = eps * 1 / torch.sqrt(torch.tensor(oft_r.shape[0]))
        I = (  # noqa: E741
            torch.zeros((oft_r.size(1), oft_r.size(1)), device=oft_r.device, dtype=oft_r.dtype)
            .unsqueeze(0)
            .expand_as(oft_r)
        )
        diff = oft_r - I
        norm_diff = torch.norm(oft_r - I, dim=(1, 2), keepdim=True)
        mask = (norm_diff <= eps).bool()
        out = torch.where(mask, oft_r, I + eps * (diff / norm_diff))
        return out

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            if len(result.shape) == 4:
                result = result.permute(0, 2, 3, 1)

            base_layer = self.get_base_layer()
            base_bias = base_layer.bias
            if base_bias is not None:
                # Bias should be added after OFT forward
                result = result - base_bias.data

            # Execute all the adapters
            for active_adapter in self.active_adapters:
                if active_adapter not in self._available_adapters:
                    continue

                module_dropout = self.module_dropout[active_adapter]

                # Modify current execution weights
                if (not self.training) or (self.training and torch.rand(1) > module_dropout):
                    result = self._get_delta_activations(active_adapter, result, *args, **kwargs)

            if base_bias is not None:
                result = result + base_bias.data
            if len(result.shape) == 4:
                result = result.permute(0, 3, 1, 2)

        result = result.to(previous_dtype)
        return result


class Linear(OFTLayer):
    """OFT implemented in Linear layer"""

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 8,
        oft_block_size: int = 0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        module_dropout: float = 0.0,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        init_weights: Union[bool, str] = True,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        OFTLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name

        self.update_layer(
            adapter_name, r, oft_block_size, module_dropout, coft, eps, block_share, init_weights
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

                    self.base_layer.weight.data = orig_weight
                else:
                    butterfly_oft_mat, boft_s = self.get_delta_weight(active_adapter)
                    orig_weight = base_layer.weight.data.clone()
                    orig_weight = torch.transpose(orig_weight, 0, 1)
                    orig_weight = torch.mm(butterfly_oft_mat, orig_weight)
                    orig_weight = torch.transpose(orig_weight, 0, 1)
                    orig_weight = orig_weight * boft_s

                    self.base_layer.weight.data = orig_weight

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
            boft_rotation = torch.eye(self.in_features, device=x.device)
            boft_scale = torch.ones((int(self.out_features), 1), device=x.device)

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

                boft_P = self.boft_P.to(block_diagonal_butterfly.device)
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
            rotated_weight = torch.mm(boft_rotation, orig_weight)
            rotated_weight = torch.transpose(rotated_weight, 0, 1)

            scaled_rotated_weight = rotated_weight * boft_scale

            result = F.linear(input=x, weight=scaled_rotated_weight, bias=self.base_layer.bias)

        result = result.to(previous_dtype)
        return result

    def _get_delta_activations(
        self, adapter_name: str, input: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        delta_weight = self.get_delta_weight(adapter_name)

        base_layer = self.get_base_layer()
        base_weight = base_layer.weight.data
        delta_weight = delta_weight[: base_weight.shape[0], : base_weight.shape[0]]

        # don't add bias here, because the bias will be added after OFT forward
        return torch.matmul(input, delta_weight)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "oft." + rep


class Conv2d(OFTLayer):
    """OFT implemented in Conv2d layer"""

    def __init__(
        self,
        base_layer,
        adapter_name,
        r: int = 8,
        oft_block_size: int = 0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        module_dropout: float = 0.0,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        init_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        OFTLayer.__init__(self, base_layer)
        self.fan_in_fan_out = fan_in_fan_out

        # Create adapter and set it active
        self.update_layer(
            adapter_name, r, oft_block_size, module_dropout, coft, eps, block_share, init_weights
        )

    def update_layer(
        self, adapter_name, boft_block_size, boft_block_num, boft_n_butterfly_factor, boft_dropout, init_weights
    ):
        """
        Update the conv2d layer with trainable BOFT weights.
        """
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

        elif boft_block_size != 0 and boft_block_num != 0:
            raise ValueError(f"You can only specify either boft_block_size ({boft_block_size}) or boft_block_num ({boft_block_num}), but not both simultaneously.")
        
        else:
            raise ValueError(f"Either `boft_block_size` or `boft_block_num` must be non-zero. Currently, boft_block_size = {boft_block_size} and boft_block_num = {boft_block_num}.")


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

        self.register_buffer("boft_P", P)

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

                    self.base_layer.weight.data = orig_weight
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

                    self.base_layer.weight.data = orig_weight

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
                self.in_features * self.base_layer.kernel_size[0] * self.base_layer.kernel_size[0], device=x.device
            )
            boft_scale = torch.ones((1, int(self.out_features)), device=x.device)

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

                boft_P = self.boft_P.to(block_diagonal_butterfly.device)
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

    def _get_delta_activations(
        self, adapter_name: str, input: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        delta_weight = self.get_delta_weight(adapter_name)

        base_layer = self.get_base_layer()
        base_weight = base_layer.weight.data
        delta_weight = delta_weight[: base_weight.shape[0], : base_weight.shape[0]]

        # don't add bias here, because the bias will be added after OFT forward
        return torch.matmul(input, delta_weight)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "oft." + rep
