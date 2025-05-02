# https://github.com/IBM/activated-lora/blob/main/alora/layer.py
import warnings
from typing import Any, Optional, list

import torch
from torch import Tensor, nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.alora.config import ALoraConfig
from peft.tuners.lora.layer import LoraLayer, MultiheadAttention
from peft.tuners.tuners_utils import BaseTunerLayer


class ALoraLayer(LoraLayer):
    """
    aLora layer class. Inherits from LoraLayer.
        It subclasses PEFT's LoraLayer, and modifies the forward method to include the aLoRA activation logic.
    """

    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, *args, **kwargs):
        super().__init__(base_layer, ephemeral_gpu_offload, *args, **kwargs)
        self.base_layer = base_layer
        base_layer = self.get_base_layer()

        if isinstance(base_layer, BaseTunerLayer):
            # we need to get the base layer of the BaseTunerLayer
            base_layer = base_layer.get_base_layer()

            if isinstance(base_layer, nn.Linear):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            elif isinstance(base_layer, nn.Conv1d):
                raise NotImplementedError("Support for nn.Conv1d is not implemented.")
            elif isinstance(base_layer, nn.Conv2d):
                raise NotImplementedError("Support for nn.Conv2d is not implemented.")
            elif isinstance(base_layer, nn.Conv3d):
                raise NotImplementedError("Support for nn.Conv3d is not implemented.")
            elif isinstance(base_layer, nn.Embedding):
                raise NotImplementedError("Support for nn.Embedding is not implemented.")
            elif isinstance(base_layer, Conv1D):
                raise NotImplementedError("Support for Conv1D is not implemented.")
            elif isinstance(base_layer, nn.MultiheadAttention):
                if not base_layer._qkv_same_embed_dim:
                    raise ValueError(f"Only same dim for query/key/value is supported as of now for {self.__class__}.")
                in_features, out_features = base_layer.embed_dim, 3 * base_layer.embed_dim
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
            elif base_layer.__class__.__name__ == "PatchedLinear":
                # INC layers
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

    def _mixed_batch_forward(
        self, x, *args: Any, adapter_names: list[str], alora_offsets: list[int], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch: Tensor = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
            if len(alora_offsets) > 1:
                alora_offsets_batch = alora_offsets[sub_batch_indices_list[i]]
                for j in range(len(alora_offsets_batch)):
                    k = min(alora_offsets_batch[j], result.shape[1])
                    lora_output = lora_B(lora_A(dropout(sub_batch[j, -k:, :]))) * scaling
                    result[sub_batch_indices_list[i][j], -k:, :] += lora_output.to(torch_result_dtype)
            else:
                # all batches have the same alora offset
                alora_offsets_batch = alora_offsets
                k = min(result.shape[1], alora_offsets_batch[0])
                lora_output = lora_B(lora_A(dropout(sub_batch[:, -k:, :]))) * scaling
                result[sub_batch_indices_list[i], -k:, :] += lora_output.to(torch_result_dtype)

        return result


class ALoraLinear(nn.Module, ALoraLayer):
    """
    aLora Linear layer class. Inherits from LoraLayer.
        It subclasses PEFT's LoraLayer, and modifies the forward method to include the aLoRA activation logic.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def merge(self, safe_merge=False, adapter_names=None):
        raise NotImplementedError(
            "The 'merge' operation is not supported for aLoRA layers. "
            "aLoRA uses dynamic activation patterns that cannot be statically merged since it is only applied upon invocation string"
        )

    def unmerge(self):
        raise NotImplementedError("The 'unmerge' operation is not supported for aLoRA layers. aLoRA cannot be merged")

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass for the aLoRA layer. It applies the lora weights after the offset position.
        """
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        alora_offsets = kwargs.pop("alora_offsets", [1])  # returns 1 if no offset

        if self.disable_adapters:
            # Commented because alora doesnt support model merge or unmerge
            # if self.merged:
            #     self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        # Commented because alora doesnt support model merge or unmerge
        # elif self.merged:
        #     result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            lora_A_keys = self.lora_A.keys()
            for active_adapter in self.active_adapters:
                if active_adapter not in lora_A_keys:
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = self._cast_input_dtype(x, lora_A.weight.dtype)
                if active_adapter not in self.lora_variant:  # vanilla LoRA
                    if (len(alora_offsets) == 1) and (alora_offsets[0] >= 0):
                        # all batches have the same alora offset, starting at 1 so acts as vanilla LoRA
                        result = result + lora_B(lora_A(dropout(x))) * scaling

                    if len(alora_offsets) == 1 and (alora_offsets[0] > 1):
                        # all batches have the same alora offset
                        k = min(result.shape[1], alora_offsets[0])
                        lora_output = lora_B(lora_A(dropout(x[:, -k:, :]))) * scaling
                        result[:, -k:, :] += lora_output.to(torch_result_dtype)

                    elif len(alora_offsets) > 1 and all(alora_offsets[i] > 0 for i in range(len(alora_offsets))):
                        # different batches have different alora offsets
                        for j in range(len(alora_offsets)):
                            k = min(result.shape[1], alora_offsets[j])
                            lora_output = lora_B(lora_A(dropout(x[j, -k:, :]))) * scaling
                            result[j, -k:, :] += lora_output.to(torch_result_dtype)

                else:
                    warnings.warn(
                        "Using aLoRA with LoRA variant. This is not recommended and may lead to unexpected results."
                    )
                    result = self.lora_variant[active_adapter].forward(
                        self,
                        active_adapter=active_adapter,
                        x=x,
                        result=result,
                    )

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


def dispatch_alora(
    target: torch.nn.Module,
    adapter_name: str,
    alora_config: ALoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    # Only support Linear and MultiheadAttention layers
    if isinstance(target_base_layer, torch.nn.MultiheadAttention):
        kwargs.update(alora_config.loftq_config)
        new_module = MultiheadAttention(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = alora_config.fan_in_fan_out = False
        kwargs.update(alora_config.loftq_config)
        new_module = ALoraLinear(target, adapter_name, **kwargs)

    return new_module
