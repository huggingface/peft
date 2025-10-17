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
from typing import Any, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.lycoris_utils import LycorisLayer


class LoHaLayer(nn.Module, LycorisLayer):
    # All names of layers that may contain adapter weights
    adapter_layer_names = ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b", "hada_t1", "hada_t2")
    # Override other_param_names to include ABBA-specific parameters
    other_param_names = ("r", "alpha", "scaling", "rank_dropout", "module_dropout", "r2", "use_khatri_rao", "scaling1", "scaling2")

    def __init__(self, base_layer: nn.Module):
        super().__init__()
        LycorisLayer.__init__(self, base_layer)

        # LoHa info
        self.hada_w1_a = nn.ParameterDict({})
        self.hada_w1_b = nn.ParameterDict({})
        self.hada_w2_a = nn.ParameterDict({})
        self.hada_w2_b = nn.ParameterDict({})
        self.hada_t1 = nn.ParameterDict({})
        self.hada_t2 = nn.ParameterDict({})
        
        # Khatri-Rao optimization flag
        self.use_khatri_rao = {}
        
        # Store second rank for ABBA (r is first component, r2 is second component, defaults to r)
        self.r2 = {}
        
        # Separate scaling factors for ABBA (α/√r and α/√r₂)
        self.scaling1 = {}
        self.scaling2 = {}

    @property
    def _available_adapters(self) -> set[str]:
        return {*self.hada_w1_a, *self.hada_w1_b, *self.hada_w2_a, *self.hada_w2_b, *self.hada_t1, *self.hada_t2}

    def create_adapter_parameters(self, adapter_name: str, r1: int, r2: int, shape: tuple[int, ...]):
        # https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/loha.py#L130C9-L143C75
        # Support different ranks for the two Hadamard components (ABBA-style)
        if len(shape) == 4:  # Conv2d
            self.hada_t1[adapter_name] = nn.Parameter(torch.empty(r1, r1, shape[2], shape[3]))
            self.hada_w1_a[adapter_name] = nn.Parameter(torch.empty(r1, shape[0]))  # out_dim, 1-mode
            self.hada_w1_b[adapter_name] = nn.Parameter(torch.empty(r1, shape[1]))  # in_dim , 2-mode

            self.hada_t2[adapter_name] = nn.Parameter(torch.empty(r2, r2, shape[2], shape[3]))
            self.hada_w2_a[adapter_name] = nn.Parameter(torch.empty(r2, shape[0]))  # out_dim, 1-mode
            self.hada_w2_b[adapter_name] = nn.Parameter(torch.empty(r2, shape[1]))  # in_dim , 2-mode
        elif len(shape) == 3:  # Conv1d
            self.hada_t1[adapter_name] = nn.Parameter(torch.empty(r1, r1, shape[2], 1))
            self.hada_w1_a[adapter_name] = nn.Parameter(torch.empty(r1, shape[0]))  # out_dim, 1-mode
            self.hada_w1_b[adapter_name] = nn.Parameter(torch.empty(r1, shape[1]))  # in_dim , 2-mode

            self.hada_t2[adapter_name] = nn.Parameter(torch.empty(r2, r2, shape[2], 1))
            self.hada_w2_a[adapter_name] = nn.Parameter(torch.empty(r2, shape[0]))  # out_dim, 1-mode
            self.hada_w2_b[adapter_name] = nn.Parameter(torch.empty(r2, shape[1]))  # in_dim , 2-mode
        else:  # Linear
            self.hada_w1_a[adapter_name] = nn.Parameter(torch.empty(shape[0], r1))
            self.hada_w1_b[adapter_name] = nn.Parameter(torch.empty(r1, shape[1]))

            self.hada_w2_a[adapter_name] = nn.Parameter(torch.empty(shape[0], r2))
            self.hada_w2_b[adapter_name] = nn.Parameter(torch.empty(r2, shape[1]))

    def reset_adapter_parameters(self, adapter_name: str):
        # Original implementation performs initialization with normal distribution
        # https://github.com/KohakuBlueleaf/LyCORIS/blob/3549fdef8f564761d68b695a08ef88b1122fdedc/lycoris/modules/loha.py#L158

        # FedPara paper proposes to perform He initialization, let's stick with it
        # It is enough to initialize only single matrix with zeros to make adapter do nothing after initialization
        if adapter_name in self.hada_w1_a.keys():
            nn.init.kaiming_uniform_(self.hada_w1_a[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_w1_b[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_w2_a[adapter_name], a=math.sqrt(5))
            nn.init.zeros_(self.hada_w2_b[adapter_name])
        if adapter_name in self.hada_t1.keys():
            nn.init.kaiming_uniform_(self.hada_t1[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_t2[adapter_name], a=math.sqrt(5))

    def reset_adapter_parameters_random(self, adapter_name: str):
        # Original implementation performs initialization with normal distribution
        # https://github.com/KohakuBlueleaf/LyCORIS/blob/3549fdef8f564761d68b695a08ef88b1122fdedc/lycoris/modules/loha.py#L158

        # FedPara paper proposes to perform He initialization, let's stick with it
        # It is enough to initialize only single matrix with zeros to make adapter do nothing after initialization
        if adapter_name in self.hada_w1_a.keys():
            nn.init.kaiming_uniform_(self.hada_w1_a[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_w1_b[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_w2_a[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_w2_b[adapter_name], a=math.sqrt(5))
        if adapter_name in self.hada_t1.keys():
            nn.init.kaiming_uniform_(self.hada_t1[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_t2[adapter_name], a=math.sqrt(5))

    def reset_adapter_parameters_abba(self, adapter_name: str):
        """
        ABBA initialization: Initialize LoHa weights to approximate the pretrained weights.
        This is based on the ABBA paper which proposes initializing adapters to approximate
        the identity function or the pretrained weights.
        
        For LoHa with separate ranks: (w1a @ w1b) ⊙ (w2a @ w2b)
        where w1a,w1b have rank r and w2a,w2b have rank r2 (defaults to r).
        We want this to approximate the pretrained weight W.
        
        Strategy: Use SVD to split the weight matrix, allocating r singular values
        to the first component and r2 to the second component.
        """
        if adapter_name in self.hada_w1_a.keys():
            base_layer = self.get_base_layer()
            
            # Get the ranks (r for first component, r2 for second component)
            r1 = self.r[adapter_name]  # First component uses r
            r2 = self.r2[adapter_name]  # Second component uses r2
            
            # Step 1: Get weight tensor
            weight = base_layer.weight
            
            # ABBA doesn't support quantized models yet
            is_quantized = hasattr(weight, "quant_state") or type(weight).__name__ in ("Params4bit", "Int8Params")
            if is_quantized:
                raise NotImplementedError(
                    f"ABBA initialization does not support quantized models (int4/int8) yet. "
                    f"Please use dtype='float32', 'float16', or 'bfloat16' instead of quantized dtypes."
                )
            
            # Get weight data (should be float32, bfloat16, or float16)
            weight = weight.data if hasattr(weight, "data") else weight
            
            # Step 2: Prepare weight for SVD
            # For Linear layers, weight is already 2D with shape (out_features, in_features)
            # For Conv layers, flatten to 2D
            if isinstance(base_layer, nn.Linear):
                W = weight
            else:
                # For conv layers, flatten to 2D: (out_channels, in_channels * kernel_size)
                W = weight.reshape(weight.shape[0], -1)
            
            # Step 3: Always cast to float32 for SVD
            # PyTorch's torch.linalg.svd does NOT support: float16, bfloat16, or any integer types 
            if W.dtype != torch.float32:
                W = W.float()
            
            # Step 4: Perform SVD on GPU (results are in float32)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            
            # Split singular values between r1 and r2
            # Take top r1+r2 singular values and split them
            total_r = min(r1 + r2, len(S))
            actual_r1 = min(r1, total_r)
            actual_r2 = min(r2, total_r)
            
            # Get components for first Hadamard term (rank r1)
            U_r1 = U[:, :actual_r1]  # (m, r1)
            S_r1 = S[:actual_r1]  # (r1,)
            Vh_r1 = Vh[:actual_r1, :]  # (r1, n)
            
            # Get components for second Hadamard term (rank r2)
            # Use next r2 singular values or reuse if not enough
            if actual_r1 + actual_r2 <= len(S):
                U_r2 = U[:, actual_r1:actual_r1 + actual_r2]  # (m, r2)
                S_r2 = S[actual_r1:actual_r1 + actual_r2]  # (r2,)
                Vh_r2 = Vh[actual_r1:actual_r1 + actual_r2, :]  # (r2, n)
            else:
                # Reuse early singular values if needed
                U_r2 = U[:, :actual_r2]  # (m, r2)
                S_r2 = S[:actual_r2]  # (r2,)
                Vh_r2 = Vh[:actual_r2, :]  # (r2, n)
            
            # Step 5: Initialize adapter parameters from SVD results
            # Use fourth root so that (w1a @ w1b) ⊙ (w2a @ w2b) ≈ W
            
            # Get adapter dtype from PEFT config (respects user configuration)
            # Adapters can be float32 (default), bfloat16, float16, etc.
            adapter_dtype = self.hada_w1_a[adapter_name].dtype
            
            # Initialize first Hadamard component: w1a @ w1b
            fourth_root_S1 = torch.pow(S_r1, 0.25)
            w1a_init = U_r1 * fourth_root_S1
            w1b_init = fourth_root_S1.unsqueeze(1) * Vh_r1
            
            # Cast from float32 (SVD output) to adapter dtype and copy
            self.hada_w1_a[adapter_name].data.copy_(w1a_init.to(adapter_dtype))
            self.hada_w1_b[adapter_name].data.copy_(w1b_init.to(adapter_dtype))
            
            # Initialize second Hadamard component: w2a @ w2b
            fourth_root_S2 = torch.pow(S_r2, 0.25)
            w2a_init = U_r2 * fourth_root_S2
            w2b_init = fourth_root_S2.unsqueeze(1) * Vh_r2
            
            # Cast from float32 (SVD output) to adapter dtype and copy
            self.hada_w2_a[adapter_name].data.copy_(w2a_init.to(adapter_dtype))
            self.hada_w2_b[adapter_name].data.copy_(w2b_init.to(adapter_dtype))
        
        if adapter_name in self.hada_t1.keys():
            # For convolutional layers with effective decomposition, use random init
            # ABBA initialization for CP decomposition is more complex
            nn.init.kaiming_uniform_(self.hada_t1[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_t2[adapter_name], a=math.sqrt(5))

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        alpha: float,
        rank_dropout: float,
        module_dropout: float,
        init_weights: Union[bool, Literal["abba"]],
        use_effective_conv2d: bool = False,
        use_khatri_rao: Union[bool, Literal["auto"]] = "auto",
        r2: int = None,
        inference_mode: bool = False,
        **kwargs,
    ) -> None:
        """Internal function to create loha adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter. For standard LoHa, both Hadamard components use 
                this rank. For ABBA mode, this is the rank of the first Hadamard component (the second 
                component's rank is controlled by r2).
            alpha (`float`): Alpha for the added adapter.
            rank_dropout (`float`): The dropout probability for rank dimension during training.
            module_dropout (`float`): The dropout probability for disabling adapter during training.
            init_weights (`Union[bool, Literal["abba"]]`): How to initialize weights. 
                `True` for default initialization (one matrix initialized to zeros), 
                `False` for random initialization, or `"abba"` for ABBA initialization which 
                approximates the pretrained weights using SVD decomposition. ABBA initialization 
                enables the adapter to start with behavior close to the original model, potentially 
                improving training stability and convergence.
                Based on the ABBA paper: https://arxiv.org/pdf/2505.14238
                See https://github.com/huggingface/peft/issues/2587 for implementation details.
            use_effective_conv2d (`bool`, *optional*, defaults to `False`):
                Use parameter effective decomposition for Conv2d with ksize > 1.
            use_khatri_rao (`Union[bool, Literal["auto"]]`, *optional*, defaults to `"auto"`):
                Use Khatri-Rao product optimization to reduce memory overhead. When set to `"auto"`, 
                it is enabled for ABBA initialization (recommended by the paper) and disabled for 
                standard LoHa. Set to `True` or `False` to explicitly control this behavior.
            r2 (`int`, *optional*): Rank for the second Hadamard component. If None, defaults to r 
                (symmetric ranks). Only relevant when using different ranks for the two components.
        """
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        # Determine r2
        # If not specified, r2 defaults to r (symmetric ranks)
        if r2 is None:
            r2 = r
            
        # Ensure at least rank 1
        r = max(1, r)
        r2 = max(1, r2)

        self.r[adapter_name] = r
        self.r2[adapter_name] = r2
        self.alpha[adapter_name] = alpha
        self.scaling[adapter_name] = alpha / r  # Original scaling (for backward compatibility)
        
        # ABBA paper: separate scaling factors α/√r and α/√r₂
        self.scaling1[adapter_name] = alpha / math.sqrt(r)
        self.scaling2[adapter_name] = alpha / math.sqrt(r2)
        
        self.rank_dropout[adapter_name] = rank_dropout
        self.module_dropout[adapter_name] = module_dropout
        
        # Handle use_khatri_rao: "auto" enables it for ABBA, disables for standard LoHa
        # User can explicitly set True/False to override
        is_abba = isinstance(init_weights, str) and init_weights.lower() == "abba"
        if use_khatri_rao == "auto":
            use_khatri_rao = is_abba  # True for ABBA, False for standard LoHa
        self.use_khatri_rao[adapter_name] = use_khatri_rao

        # Determine shape of LoHa weights
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            shape = tuple(base_layer.weight.shape)
        elif isinstance(base_layer, nn.Conv2d):
            # For 1x1 convolutions, disable effective_conv2d to avoid unnecessary tensor reshaping overhead.
            # Since 1x1 convolutions are essentially pointwise operations (matrix multiplications),
            # they can be more efficiently handled with the flattened weight representation,
            # similar to how Linear layers work. This optimization reduces computational cost
            # without affecting the mathematical equivalence of the operation.
            use_effective_conv2d = use_effective_conv2d and base_layer.kernel_size != (1, 1)
            if use_effective_conv2d:
                shape = (base_layer.out_channels, base_layer.in_channels, *base_layer.kernel_size)
            else:
                shape = (
                    base_layer.out_channels,
                    base_layer.in_channels * base_layer.kernel_size[0] * base_layer.kernel_size[1],
                )
        elif isinstance(base_layer, nn.Conv1d):
            # For Conv1d with kernel_size=1, disable effective_conv2d for the same optimization reasons
            # as 1x1 Conv2d. Kernel size 1 means no spatial/temporal context, making it equivalent
            # to a Linear layer applied across the channel dimension. Using flattened representation
            # avoids unnecessary reshaping and improves computational efficiency.
            use_effective_conv2d = use_effective_conv2d and base_layer.kernel_size[0] != 1
            if use_effective_conv2d:
                shape = (base_layer.out_channels, base_layer.in_channels, base_layer.kernel_size[0])
            else:
                shape = (
                    base_layer.out_channels,
                    base_layer.in_channels * base_layer.kernel_size[0],
                )
        else:
            raise TypeError(f"LoHa is not implemented for base layers of type {type(base_layer).__name__}")

        # Create weights with provided shape (using r and r2)
        self.create_adapter_parameters(adapter_name, r, r2, shape)

        # Initialize weights
        if isinstance(init_weights, str) and init_weights.lower() == "abba":
            self.reset_adapter_parameters_abba(adapter_name)
        elif init_weights:
            self.reset_adapter_parameters(adapter_name)
        else:
            self.reset_adapter_parameters_random(adapter_name)

        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        # https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/loha.py#L178
        if adapter_name in self.hada_t1.keys():
            weight = make_weight_cp(
                self.hada_t1[adapter_name],
                self.hada_w1_a[adapter_name],
                self.hada_w1_b[adapter_name],
                self.hada_t2[adapter_name],
                self.hada_w2_a[adapter_name],
                self.hada_w2_b[adapter_name],
                scale=torch.tensor(self.scaling[adapter_name]),
            )
        else:
            # Check if Khatri-Rao optimization is enabled
            use_kr = self.use_khatri_rao.get(adapter_name, False)
            
            if use_kr:
                # Use ABBA paper formula with separate scales: α/√r₁ and α/√r₂
                weight = make_weight_kr(
                    self.hada_w1_a[adapter_name],
                    self.hada_w1_b[adapter_name],
                    self.hada_w2_a[adapter_name],
                    self.hada_w2_b[adapter_name],
                    scale1=torch.tensor(self.scaling1[adapter_name]),
                    scale2=torch.tensor(self.scaling2[adapter_name]),
                )
            else:
                weight = make_weight(
                    self.hada_w1_a[adapter_name],
                    self.hada_w1_b[adapter_name],
                    self.hada_w2_a[adapter_name],
                    self.hada_w2_b[adapter_name],
                    scale=torch.tensor(self.scaling[adapter_name]),
                )

        base_layer = self.get_base_layer()

        # Reshape to match base layer shape
        weight = weight.reshape(base_layer.weight.shape)

        # Perform rank dropout during training - drop rows of addition weights
        rank_dropout = self.rank_dropout[adapter_name]
        if self.training and rank_dropout:
            drop = (torch.rand(weight.size(0)) > rank_dropout).to(weight.dtype)
            drop = drop.view(-1, *[1] * len(weight.shape[1:])).to(weight.device)
            # TODO: Investigate if there should be a scaler like in normal dropout during training
            # Original implementation doesn't have it
            # https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/loha.py#L193
            drop /= drop.mean()
            weight *= drop

        return weight

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

            # Execute all the adapters
            for active_adapter in self.active_adapters:
                if active_adapter not in self._available_adapters:
                    continue

                module_dropout = self.module_dropout[active_adapter]

                # Modify current execution weights
                if (not self.training) or (self.training and torch.rand(1) > module_dropout):
                    result = result + self._get_delta_activations(active_adapter, x, *args, **kwargs)

        result = result.to(previous_dtype)
        return result


class Linear(LoHaLayer):
    """LoHa implemented in Linear layer"""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str = "default",
        r: int = 0,
        alpha: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        init_weights: Union[bool, Literal["abba"]] = True,
        use_khatri_rao: Union[bool, Literal["auto"]] = "auto",
        r2: int = None,
        **kwargs,
    ):
        super().__init__(base_layer)

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, alpha, rank_dropout, module_dropout, init_weights, use_khatri_rao=use_khatri_rao, r2=r2, **kwargs)

    def _get_delta_activations(
        self, adapter_name: str, input: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        delta_weight = self.get_delta_weight(adapter_name)
        input = self._cast_input_dtype(input, delta_weight.dtype)
        # don't add bias here, because the bias is already included in the output of the base_layer
        return F.linear(input, delta_weight)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "loha." + rep


class Conv2d(LoHaLayer):
    """LoHa implemented in Conv2d layer"""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str = "default",
        r: int = 0,
        alpha: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        use_effective_conv2d: bool = False,
        init_weights: Union[bool, Literal["abba"]] = True,
        use_khatri_rao: Union[bool, Literal["auto"]] = "auto",
        r2: int = None,
        **kwargs,
    ):
        super().__init__(base_layer)

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, r, alpha, rank_dropout, module_dropout, init_weights, use_effective_conv2d, use_khatri_rao=use_khatri_rao, r2=r2, **kwargs
        )

    def _get_delta_activations(
        self, adapter_name: str, input: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        delta_weight = self.get_delta_weight(adapter_name)
        input = self._cast_input_dtype(input, delta_weight.dtype)
        # don't add bias here, because the bias is already included in the output of the base_layer
        base_layer = self.get_base_layer()
        return F.conv2d(
            input,
            delta_weight,
            stride=base_layer.stride,
            padding=base_layer.padding,
            dilation=base_layer.dilation,
            groups=base_layer.groups,
        )

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "loha." + rep


class Conv1d(LoHaLayer):
    """LoHa implemented in Conv1d layer"""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str = "default",
        r: int = 0,
        alpha: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        use_effective_conv2d: bool = False,
        init_weights: Union[bool, Literal["abba"]] = True,
        use_khatri_rao: Union[bool, Literal["auto"]] = "auto",
        r2: int = None,
        **kwargs,
    ):
        super().__init__(base_layer)

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, r, alpha, rank_dropout, module_dropout, init_weights, use_effective_conv2d, use_khatri_rao=use_khatri_rao, r2=r2, **kwargs
        )

    def _get_delta_activations(
        self, adapter_name: str, input: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        delta_weight = self.get_delta_weight(adapter_name)
        input = self._cast_input_dtype(input, delta_weight.dtype)
        # don't add bias here, because the bias is already included in the output of the base_layer
        base_layer = self.get_base_layer()
        return F.conv1d(
            input,
            delta_weight,
            stride=base_layer.stride,
            padding=base_layer.padding,
            dilation=base_layer.dilation,
            groups=base_layer.groups,
        )

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "loha." + rep


# For abba
class HadaWeightKR(torch.autograd.Function):
    """
    Khatri-Rao optimized version of HadaWeight that avoids materializing
    the full B1A1 and B2A2 matrices, significantly reducing memory overhead.
    
    Key Innovation:
    Instead of computing (w1a @ w1b) * (w2a @ w2b) which requires storing two
    m×n matrices, we compute the result row-by-row (or in chunks), never storing
    the full intermediate matrices in memory.
    
    ABBA paper formula:
    ΔW = (α/√r₁ · B₁A₁) ⊙ (α/√r₂ · B₂A₂)
    where scale1 = α/√r₁ and scale2 = α/√r₂
    
    Mathematical equivalence:
    result[i,j] = scale1 * (sum_k w1a[i,k]*w1b[k,j]) * scale2 * (sum_k w2a[i,k]*w2b[k,j])
    
    This can be computed without materializing full matrices by processing
    one row at a time or using einsum with no intermediate storage.
    
    Memory savings: O(m*n) -> O(n) for forward pass (processing row by row)
    """
    @staticmethod
    def forward(ctx, w1a, w1b, w2a, w2b, scale1=torch.tensor(1), scale2=torch.tensor(1)):
        ctx.save_for_backward(w1a, w1b, w2a, w2b, scale1, scale2)
        
        # Handle different ranks: w1a/w1b may have rank r1, w2a/w2b may have rank r2
        # w1a: (m, r1), w1b: (r1, n)
        # w2a: (m, r2), w2b: (r2, n)
        
        m = w1a.shape[0]
        n = w1b.shape[1]
        
        # Allocate output
        diff_weight = torch.empty(m, n, dtype=w1a.dtype, device=w1a.device)
        
        # Process in chunks to save memory (chunk_size can be tuned)
        # Smaller chunk_size = less memory, but more overhead
        chunk_size = min(128, m)  # Process 128 rows at a time
        
        for i in range(0, m, chunk_size):
            end_i = min(i + chunk_size, m)
            # Compute chunk of term1: scale1 * (w1a[i:end_i] @ w1b) -> (chunk_size, n)
            term1_chunk = scale1 * (w1a[i:end_i] @ w1b)  # Only materialize chunk_size × n
            # Compute chunk of term2: scale2 * (w2a[i:end_i] @ w2b) -> (chunk_size, n)
            term2_chunk = scale2 * (w2a[i:end_i] @ w2b)  # Only materialize chunk_size × n
            # Element-wise multiply and store
            # Result: (α/√r₁ · B₁A₁) ⊙ (α/√r₂ · B₂A₂)
            diff_weight[i:end_i] = term1_chunk * term2_chunk
            # These chunks are automatically freed after use
        
        return diff_weight

    @staticmethod
    def backward(ctx, grad_out):
        (w1a, w1b, w2a, w2b, scale1, scale2) = ctx.saved_tensors
        
        # Handle different ranks: w1a/w1b may have rank r1, w2a/w2b may have rank r2
        # w1a: (m, r1), w1b: (r1, n)
        # w2a: (m, r2), w2b: (r2, n)
        m = w1a.shape[0]
        n = w1b.shape[1]
        
        # Initialize gradients
        grad_w1a = torch.zeros_like(w1a)
        grad_w1b = torch.zeros_like(w1b)
        grad_w2a = torch.zeros_like(w2a)
        grad_w2b = torch.zeros_like(w2b)
        
        # Process in chunks to save memory
        chunk_size = min(128, m)
        
        for i in range(0, m, chunk_size):
            end_i = min(i + chunk_size, m)
            
            # Recompute forward pass chunks (trade computation for memory)
            # term1_chunk = scale1 * (w1a @ w1b), term2_chunk = scale2 * (w2a @ w2b)
            term1_chunk = scale1 * (w1a[i:end_i] @ w1b)  # (chunk_size, n)
            term2_chunk = scale2 * (w2a[i:end_i] @ w2b)  # (chunk_size, n)
            
            grad_out_chunk = grad_out[i:end_i]  # (chunk_size, n)
            
            # Gradients for w1a and w1b
            # d(ΔW)/d(B₁A₁) = grad_out ⊙ scale1 ⊙ (scale2 · B₂A₂)
            # Chain rule: d/dw1a = scale1 * (grad_out ⊙ term2_chunk) @ w1b.T
            grad_term1_chunk = scale1 * (grad_out_chunk * term2_chunk)  # (chunk_size, n)
            grad_w1a[i:end_i] = grad_term1_chunk @ w1b.T  # (chunk_size, r1)
            grad_w1b += w1a[i:end_i].T @ grad_term1_chunk  # (r1, n)
            
            # Gradients for w2a and w2b
            # d(ΔW)/d(B₂A₂) = grad_out ⊙ scale2 ⊙ (scale1 · B₁A₁)
            # Chain rule: d/dw2a = scale2 * (grad_out ⊙ term1_chunk) @ w2b.T
            grad_term2_chunk = scale2 * (grad_out_chunk * term1_chunk)  # (chunk_size, n)
            grad_w2a[i:end_i] = grad_term2_chunk @ w2b.T  # (chunk_size, r2)
            grad_w2b += w2a[i:end_i].T @ grad_term2_chunk  # (r2, n)
            
            # Chunks are freed here
        
        return grad_w1a, grad_w1b, grad_w2a, grad_w2b, None, None

def make_weight_kr(w1a, w1b, w2a, w2b, scale1, scale2):
    """
    Generate weights using Khatri-Rao optimization with separate scaling.
    
    ABBA paper formula: ΔW = (α/√r₁ · B₁A₁) ⊙ (α/√r₂ · B₂A₂)
    where scale1 = α/√r₁ and scale2 = α/√r₂
    """
    return HadaWeightKR.apply(w1a, w1b, w2a, w2b, scale1, scale2)

    
# Below code is a direct copy from https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/loha.py#L9


class HadaWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w1a, w1b, w2a, w2b, scale=torch.tensor(1)):
        ctx.save_for_backward(w1a, w1b, w2a, w2b, scale)
        diff_weight = ((w1a @ w1b) * (w2a @ w2b)) * scale
        return diff_weight

    @staticmethod
    def backward(ctx, grad_out):
        (w1a, w1b, w2a, w2b, scale) = ctx.saved_tensors
        grad_out = grad_out * scale
        temp = grad_out * (w2a @ w2b)
        grad_w1a = temp @ w1b.T
        grad_w1b = w1a.T @ temp

        temp = grad_out * (w1a @ w1b)
        grad_w2a = temp @ w2b.T
        grad_w2b = w2a.T @ temp

        del temp
        return grad_w1a, grad_w1b, grad_w2a, grad_w2b, None


class HadaWeightCP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t1, w1a, w1b, t2, w2a, w2b, scale=torch.tensor(1)):
        ctx.save_for_backward(t1, w1a, w1b, t2, w2a, w2b, scale)

        rebuild1 = torch.einsum("i j k l, j r, i p -> p r k l", t1, w1b, w1a)
        rebuild2 = torch.einsum("i j k l, j r, i p -> p r k l", t2, w2b, w2a)

        return rebuild1 * rebuild2 * scale

    @staticmethod
    def backward(ctx, grad_out):
        (t1, w1a, w1b, t2, w2a, w2b, scale) = ctx.saved_tensors
        grad_out = grad_out * scale

        temp = torch.einsum("i j k l, j r -> i r k l", t2, w2b)
        rebuild = torch.einsum("i j k l, i r -> r j k l", temp, w2a)

        grad_w = rebuild * grad_out
        del rebuild

        grad_w1a = torch.einsum("r j k l, i j k l -> r i", temp, grad_w)
        grad_temp = torch.einsum("i j k l, i r -> r j k l", grad_w, w1a.T)
        del grad_w, temp

        grad_w1b = torch.einsum("i r k l, i j k l -> r j", t1, grad_temp)
        grad_t1 = torch.einsum("i j k l, j r -> i r k l", grad_temp, w1b.T)
        del grad_temp

        temp = torch.einsum("i j k l, j r -> i r k l", t1, w1b)
        rebuild = torch.einsum("i j k l, i r -> r j k l", temp, w1a)

        grad_w = rebuild * grad_out
        del rebuild

        grad_w2a = torch.einsum("r j k l, i j k l -> r i", temp, grad_w)
        grad_temp = torch.einsum("i j k l, i r -> r j k l", grad_w, w2a.T)
        del grad_w, temp

        grad_w2b = torch.einsum("i r k l, i j k l -> r j", t2, grad_temp)
        grad_t2 = torch.einsum("i j k l, j r -> i r k l", grad_temp, w2b.T)
        del grad_temp
        return grad_t1, grad_w1a, grad_w1b, grad_t2, grad_w2a, grad_w2b, None


def make_weight(w1a, w1b, w2a, w2b, scale):
    return HadaWeight.apply(w1a, w1b, w2a, w2b, scale)


def make_weight_cp(t1, w1a, w1b, t2, w2a, w2b, scale):
    return HadaWeightCP.apply(t1, w1a, w1b, t2, w2a, w2b, scale)