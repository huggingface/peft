from typing import Any

import torch
import torch.nn as nn
import torch.nn.init as init

from .layer import LoraLayer


class LoraParallelLinear(nn.Module, LoraLayer):
    """
    When the target layer parallel_linear is RowParallelLinear, in order to keep the input and output shapes
    consistent, we need to split the lora matrix A into rows, and the lora_B at this time should be a complete linear
    layer; In the same way, when the target layer is ColumnParallelLinear, we perform column segmentation on lora_B,
    while lora_A is still a complete linear layer.
    """

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        backend,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        **kwargs,
    ):
        super().__init__()
        LoraLayer.__init__(self, base_layer=base_layer)

        self.backend = backend
        self.is_paralle_a = isinstance(base_layer, backend.RowParallelLinear)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name

        megatron_config = kwargs["megatron_config"]
        parallel_linear_kwargs = {"megatron_config": megatron_config}
        init_method = init.xavier_normal_
        if hasattr(megatron_config, "init_method"):
            init_method = megatron_config.init_method
        input_is_parallel = True
        gather_output = False
        if isinstance(base_layer, self.backend.RowParallelLinear):
            input_is_parallel = base_layer.input_is_parallel
        else:
            gather_output = base_layer.gather_output
        self.update_layer(
            adapter_name,
            r,
            lora_alpha,
            lora_dropout,
            init_lora_weights,
            init_method,
            input_is_parallel,
            gather_output,
            **parallel_linear_kwargs,
        )

        self.is_target_conv_1d_layer = False

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        init_method=init.xavier_normal_,
        input_is_parallel=True,
        gather_output=False,
        **parallel_linear_kwargs,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer

        megatron_config = parallel_linear_kwargs["megatron_config"]
        # lora needs to be forced to upgrade to 32-bit precision, otherwise it will overflow
        megatron_config.params_dtype = torch.float32
        if self.is_paralle_a:
            lora_a = self.backend.RowParallelLinear(
                input_size=self.in_features,
                output_size=r,
                bias=False,
                input_is_parallel=input_is_parallel,
                skip_bias_add=True,
                init_method=init_method,
                config=megatron_config,
            )
            lora_b = nn.Linear(in_features=r, out_features=self.out_features, bias=False, dtype=torch.float32)
        else:
            lora_a = nn.Linear(in_features=self.in_features, out_features=r, bias=False, dtype=torch.float32)
            lora_b = self.backend.ColumnParallelLinear(
                input_size=r,
                output_size=self.out_features,
                bias=False,
                gather_output=gather_output,
                init_method=init_method,
                config=megatron_config,
            )
        self.lora_A[adapter_name] = lora_a
        self.lora_B[adapter_name] = lora_b
        self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        previous_dtype = x.dtype
        # If weight is used for matrix multiplication here, the final aggregation operation of the original
        # parallel_linear layer will be missing, so we need to directly call its forward function to obtain the
        # output of the original parallel_linear layer.
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result, bias = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result, bias = self.base_layer(x, *args, **kwargs)
        else:
            result, bias = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                lora_result = lora_A(dropout(x))
                if isinstance(lora_result, tuple):
                    lora_result = lora_result[0]
                lora_result = lora_B(lora_result)
                if isinstance(lora_result, tuple):
                    lora_result = lora_result[0]
                lora_result = lora_result * scaling

                result = result + lora_result

        result = result.to(previous_dtype)
        return result, bias
