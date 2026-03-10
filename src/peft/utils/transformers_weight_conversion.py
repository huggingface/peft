# Copyright 2026-present the HuggingFace Inc. team.
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

# NOTE: don't import from this module unless transformers v5+ is used
import re

import torch

from transformers.core_model_loading import (
    Concatenate,
    ConversionOps,
    MergeModulelist,
    Transpose,
    WeightConverter,
    WeightRenaming,
)


def _block_diag_3d(tensors: list[torch.Tensor]) -> torch.Tensor:
    if len(tensors) < 2:
        raise ValueError(f"_block_diag_3d expects at least 2 tensors, got {len(tensors)}")

    if any(t.dim() != 3 for t in tensors):
        raise ValueError("_block_diag_3d expects all tensors to be 3d.")

    num_experts = tensors[0].shape[0]
    if any(t.shape[0] != num_experts for t in tensors):
        raise ValueError("All tensors passed to _block_diag_3d must have the same number of experts.")

    lora_b_block_diag = []
    for i in range(num_experts):
        lora_b_block_diag.append(torch.block_diag(*[tensor[i] for tensor in tensors]))
    return torch.stack(lora_b_block_diag, dim=0)


class PeftConcatenate(Concatenate):
    """Convert per-expert LoRA weights to merged weights.

    When the base weights are fused, e.g. W01 = [W0, W1], the LoRA weights also need to be fused. To achieve this
    correctly, concatenate the LoRA A weights along the r (rank) dimension. This doesn't require a new Operation. But
    for LoRA B, the weights need to be merged in a block diagonal fashion to achieve the correct result.

    To illustrate:

    Before
    W0' = W0 + A0 @ B0
    W1' = W1 + A1 @ B1

    After
    W01' = W01 + A01 @ B01_bd
        where
        A01 = [A0, A1]
        B01_bd = [[B0,  0],
                  [0,  B1]]

    This class is responsible for merging LoRA B in this block-diagonal fashion. Assuming that we fuse N weights, it
    should look like this:

    1. LoRA B is 2-dim
    Normal LoRA weight of shape (out_feat, rank), the output shape should be (N * out_feat, N * rank).

    2. LoRA B is 3-dim
    MoE LoRA weight of shape (experts, out_feat, rank), the output shape should be (experts, N * out_feat, N * rank).

    After this, the experts x rank dimension are flattened, as PEFT expects 2d tensors for LoRA.
    """

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        full_layer_name: str,
        **kwargs,
    ) -> dict[str, list[torch.Tensor]]:
        dims = [v.dim() for v in input_dict.values()]
        if set(dims) not in ({2}, {3}):
            raise ValueError(
                f"To convert this LoRA adapter, the LoRA weights all need to have either 2 or 3 dims, got {set(dims)}"
            )

        # Keep source order stable (e.g. w1 before w3 for Mixtral) to preserve gate/up semantics.
        ordered_tensors = [
            input_dict[source_pattern] for source_pattern in source_patterns if source_pattern in input_dict
        ]
        if len(ordered_tensors) != len(input_dict):
            missing = set(input_dict) - set(source_patterns)
            raise ValueError(
                "Collected tensors contain keys not present in source_patterns. "
                f"Unexpected keys: {sorted(missing)}; source_patterns={source_patterns}"
            )

        if set(dims) == {2}:
            output_dict = {full_layer_name: torch.block_diag(*ordered_tensors)}
        else:
            # with r being the LoRA rank and n being the number of fused weights:
            out = _block_diag_3d(ordered_tensors)  # shape = experts, n*out_feat, 2*r
            out = torch.permute(out, (2, 0, 1))  # shape = 2*r, experts, n*out_feat
            out = out.flatten(0, 1)  # shape = 2*r * experts, n*out_feat
            out = out.T
            output_dict = {full_layer_name: out}
        return output_dict

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("Reversing PEFT LoRA MoE conversions is not supported yet.")


class FlattenDims(ConversionOps):
    """
    Flatten the tensors along the given dimensions
    """

    def __init__(self, dims: int | tuple[int, ...]):
        if isinstance(dims, int):
            dims = (dims,)
        self.dims = dims

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        config,
        **kwargs,
    ) -> dict[str, list[torch.Tensor]]:
        output_dict = {k: v.flatten(*self.dims) for k, v in input_dict.items()}
        return output_dict

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("Reversing flatteing operatio is not supported.")

    def __repr__(self):
        return f"{self.__class__.__name__}(dims={self.dims})"


class PermuteDims(ConversionOps):
    """
    Permute the tensors along the given dimensions
    """

    def __init__(self, dims: tuple[int, ...]):
        self.dims = dims

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        config,
        **kwargs,
    ) -> dict[str, list[torch.Tensor]]:
        output_dict = {k: v.permute(*self.dims) for k, v in input_dict.items()}
        return output_dict

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("Reversing flatteing operatio is not supported yet.")

    def __repr__(self):
        return f"{self.__class__.__name__}(dims={self.dims})"


def build_peft_weight_mapping(
    weight_conversions: list[WeightConverter | WeightRenaming] | None, adapter_name: str, peft_config=None
) -> list[WeightConverter | WeightRenaming]:
    # We iterate over all the operations of the original model and simply edit them to apply to the PEFT adapter when
    # appropriate.
    # Note: This function is used in PEFT, changing it requires coordination.
    if not weight_conversions:
        return []

    # strip "base_model.model" and add adapter name
    new_weight_conversions = [WeightRenaming("base_model.model.model.", "model.")]

    prefixes = set()
    from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING

    peft_type = getattr(peft_config, "peft_type", None)
    if peft_type in PEFT_TYPE_TO_PREFIX_MAPPING:
        prefixes.add(PEFT_TYPE_TO_PREFIX_MAPPING[peft_type])
    else:
        prefixes.update(PEFT_TYPE_TO_PREFIX_MAPPING.values())

    for prefix in sorted(prefixes):
        escaped_prefix = re.escape(prefix)
        new_weight_conversions.append(
            WeightRenaming(
                source_patterns=rf"({escaped_prefix}[^\.]*)",
                target_patterns=rf"\1.{adapter_name}",
            )
        )

    for orig_conversion in weight_conversions:
        if isinstance(orig_conversion, WeightRenaming):
            new_weight_conversions.append(orig_conversion)
            continue

        if len(orig_conversion.target_patterns) == 1 and orig_conversion.target_patterns[0].endswith("gate_up_proj"):
            # gate_up_proj requires both merging the experts and concatenating for the fusion of w1 and w3
            for lora in ("lora_A", "lora_B"):  # TODO: lora_embedding_A and lora_embedding_B
                # deal with operations
                peft_weight_operations = []
                for op in orig_conversion.operations:
                    if isinstance(op, Concatenate):
                        if lora == "lora_B":  # block diagonal concat
                            peft_weight_operations.append(PeftConcatenate(dim=op.dim))
                        else:  # normal concat + flatten
                            peft_weight_operations.append(op)
                            peft_weight_operations.append(FlattenDims(dims=(0, 1)))
                    elif isinstance(op, MergeModulelist):
                        peft_weight_operations.append(op)

                # TODO: this assumption may not hold for models != mixtral
                # For source, we capture the original weights + the lora weights
                new_source_patterns = []
                for pat in list(orig_conversion.source_patterns):
                    # we replace the weight pattern to colllect loras
                    pat = pat.rsplit(".", 1)[0]
                    # note: the source state_dict does *not* contain the adapter name
                    new_source_patterns.append(f"{pat}.{lora}.*")

                # the gate_up_proj is the innner PEFT ParamWrapper, so we need to use base_layer
                pat = orig_conversion.target_patterns[0]
                pat = pat.replace("gate_up_proj", "base_layer")
                # we make sure the target key is correct, add '.weight' because the parameter is targeted directly
                new_target_patterns = [f"{pat}.{lora}.{adapter_name}.weight"]

                # Instantiate a new object that correctly post process patterns if needed
                new_conversion = orig_conversion.__class__(
                    source_patterns=new_source_patterns,
                    target_patterns=new_target_patterns,
                    distributed_operation=orig_conversion.distributed_operation,
                    quantization_operation=orig_conversion.quantization_operation,
                    operations=peft_weight_operations,
                )
                new_weight_conversions.append(new_conversion)

        elif len(orig_conversion.target_patterns) == 1 and orig_conversion.target_patterns[0].endswith("down_proj"):
            # down_proj only requires merging of experts
            for lora in ("lora_A", "lora_B"):  # TODO: lora_embedding_A and lora_embedding_B
                peft_weight_operations = []
                for op in orig_conversion.operations:
                    if isinstance(op, MergeModulelist):
                        peft_weight_operations.append(op)
                        if lora == "lora_A":
                            peft_weight_operations.append(FlattenDims(dims=(0, 1)))
                        else:
                            peft_weight_operations.append(PermuteDims(dims=(2, 0, 1)))
                            peft_weight_operations.append(FlattenDims(dims=(0, 1)))
                            peft_weight_operations.append(Transpose(dim0=0, dim1=1))

                # TODO: this assumption may not hold for models != mixtral
                # For source, we capture the original weights + the lora weights
                new_source_patterns = []
                for pat in list(orig_conversion.source_patterns):
                    # we replace the weight pattern to colllect loras
                    pat = pat.rsplit(".", 1)[0]
                    # note: the source state_dict does *not* contain the adapter name
                    new_source_patterns.append(f"{pat}.{lora}.*")

                # the down_proj is the outer PEFT ParamWrapper, so we remove the prefix
                pat = orig_conversion.target_patterns[0]
                pat = pat.replace(".down_proj", "")
                # we make sure the target key is correct, add '.weight' because the parameter is targeted directly
                new_target_patterns = [f"{pat}.{lora}.{adapter_name}.weight"]

                # Instantiate a new object that correctly post process patterns if needed
                new_conversion = orig_conversion.__class__(
                    source_patterns=new_source_patterns,
                    target_patterns=new_target_patterns,
                    distributed_operation=orig_conversion.distributed_operation,
                    quantization_operation=orig_conversion.quantization_operation,
                    operations=peft_weight_operations,
                )
                new_weight_conversions.append(new_conversion)

    return new_weight_conversions
