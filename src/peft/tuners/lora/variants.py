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

from typing import Any

import torch
from accelerate.utils.imports import is_xpu_available
from torch import nn

from peft.utils.other import transpose
import torch.nn.functional as F

from .dora import DoraConv1dLayer, DoraConv2dLayer, DoraConv3dLayer, DoraEmbeddingLayer, DoraLinearLayer
from .layer import Conv1d, Conv2d, Conv3d, Embedding, Linear, LoraVariant, _ConvNd


class DoraLinearVariant(LoraVariant):
    @staticmethod
    def init(module: Linear, adapter_name: str, **kwargs: Any) -> None:
        if not module.lora_magnitude_vector:
            # first dora layer being added, add lora_magnitude_vector to the list of learnable parameters
            module.adapter_layer_names = module.adapter_layer_names[:] + ("lora_magnitude_vector",)

        dora_layer = DoraLinearLayer(fan_in_fan_out=getattr(module, "fan_in_fan_out", False))
        lora_A = module.lora_A[adapter_name].weight
        lora_B = module.lora_B[adapter_name].weight
        place_on_cpu = module.ephemeral_gpu_offload and (lora_A.device.type == "cpu" or lora_B.device.type == "cpu")
        if module.ephemeral_gpu_offload:
            if lora_A.device.type in ["cuda", "xpu"]:
                lora_B = lora_B.to(lora_A.device)
            else:
                if lora_B.device.type not in ["cuda", "xpu"]:
                    if is_xpu_available():
                        lora_B = lora_B.to("xpu")
                    else:
                        lora_B = lora_B.to("cuda")
                lora_A = lora_A.to(lora_B.device)
        scaling = module.scaling[adapter_name]
        dora_layer.update_layer(
            base_layer=module.get_base_layer(),
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=scaling,
            place_on_cpu=place_on_cpu,
        )
        module.lora_magnitude_vector[adapter_name] = dora_layer

    @staticmethod
    def merge_safe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)

        # since delta_weight already includes scaling, set it to 1 here
        weight_norm = (
            module.lora_magnitude_vector[active_adapter]
            .get_weight_norm(orig_weight, transpose(delta_weight, module.fan_in_fan_out), scaling=1)
            .detach()
        )
        # We need to cache weight_norm because it has to be based on the original weights. We
        # cannot calculate it on the fly based on the merged weights when unmerging because its a
        # different value
        module._cache_store(f"{active_adapter}-weight_norm", weight_norm)
        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm
        dora_factor = transpose(dora_factor.view(-1, 1), module.fan_in_fan_out)
        new_weight = dora_factor * (orig_weight + delta_weight)
        new_weight = new_weight.to(orig_dtype)
        return new_weight

    @staticmethod
    def merge_unsafe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> None:
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)
        weight_norm = (
            module.lora_magnitude_vector[active_adapter]
            .get_weight_norm(orig_weight, transpose(delta_weight, module.fan_in_fan_out), scaling=1)
            .detach()
        )
        # We need to cache weight_norm because it has to be based on the original weights. We
        # cannot calculate it on the fly based on the merged weights when unmerging because its a
        # different value
        module._cache_store(f"{active_adapter}-weight_norm", weight_norm)
        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm
        dora_factor = transpose(dora_factor.view(-1, 1), module.fan_in_fan_out)
        new_weight = dora_factor * (orig_weight.data + delta_weight)
        new_weight = new_weight.to(orig_dtype)
        orig_weight.data = new_weight

    @staticmethod
    def unmerge(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)
        weight_norm = module._cache_pop(f"{active_adapter}-weight_norm")
        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm
        new_weight = orig_weight.data / dora_factor.view(-1, 1) - delta_weight
        new_weight = new_weight.to(orig_dtype)
        return new_weight

    @staticmethod
    def forward(module: Linear, active_adapter: str, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]

        if isinstance(dropout, nn.Identity) or not module.training:
            base_result = result
        else:
            x = dropout(x)
            base_result = None

        result = result + module.lora_magnitude_vector[active_adapter](
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=scaling,
            base_layer=module.get_base_layer(),
            base_result=base_result,
        )
        return result


class DoraEmbeddingVariant(DoraLinearVariant):
    @staticmethod
    def init(module: Embedding, adapter_name: str, **kwargs: Any) -> None:
        if module.lora_magnitude_vector is None:
            # first dora layer being added, add lora_magnitude_vector to the list of learnable parameters
            module.adapter_layer_names = module.adapter_layer_names[:] + ("lora_magnitude_vector",)

        dora_layer = DoraEmbeddingLayer(fan_in_fan_out=True)
        lora_embedding_A = module.lora_embedding_A[adapter_name]
        lora_embedding_B = module.lora_embedding_B[adapter_name]
        scaling = module.scaling[adapter_name]
        dora_layer.update_layer(
            base_layer=module.get_base_layer(), lora_A=lora_embedding_A, lora_B=lora_embedding_B, scaling=scaling
        )
        module.lora_magnitude_vector[adapter_name] = dora_layer

    @staticmethod
    def merge_safe(module: Embedding, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)

        # since delta_weight already includes scaling, set it to 1 here
        weight_norm = (
            module.lora_magnitude_vector[active_adapter]
            .get_weight_norm(orig_weight, delta_weight.T, scaling=1)
            .detach()
        )
        # We need to cache weight_norm because it has to be based on the original weights. We
        # cannot calculate it on the fly based on the merged weights when unmerging because its a
        # different value
        module._cache_store(f"{active_adapter}-weight_norm", weight_norm)
        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm
        dora_factor = dora_factor.view(1, -1)
        new_weight = dora_factor * (orig_weight + delta_weight)
        new_weight = new_weight.to(orig_dtype)
        return new_weight

    @staticmethod
    def merge_unsafe(module: Embedding, active_adapter: str, orig_weight: torch.Tensor) -> None:
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)
        weight_norm = (
            module.lora_magnitude_vector[active_adapter]
            .get_weight_norm(orig_weight, delta_weight.T, scaling=1)
            .detach()
        )
        # We need to cache weight_norm because it has to be based on the original weights. We
        # cannot calculate it on the fly based on the merged weights when unmerging because its a
        # different value
        module._cache_store(f"{active_adapter}-weight_norm", weight_norm)
        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm
        dora_factor = dora_factor.view(1, -1)
        new_weight = dora_factor * (orig_weight.data + delta_weight)
        new_weight = new_weight.to(orig_dtype)
        orig_weight.data = new_weight

    @staticmethod
    def unmerge(module: Embedding, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)
        weight_norm = module._cache_pop(f"{active_adapter}-weight_norm")
        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm
        new_weight = orig_weight.data / dora_factor.view(1, -1) - delta_weight
        new_weight = new_weight.to(orig_dtype)
        return new_weight

    @staticmethod
    def forward(module: Embedding, active_adapter: str, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        embedding_A = module.lora_embedding_A[active_adapter].T
        embedding_B = module.lora_embedding_B[active_adapter].T
        scaling = module.scaling[active_adapter]

        mag_norm_scale, dora_result = module.lora_magnitude_vector[active_adapter](
            x,
            lora_A=embedding_A,
            lora_B=embedding_B,
            scaling=scaling,
            base_layer=module.get_base_layer(),
            embed_fn=module._embed,
        )
        result = mag_norm_scale * result + dora_result
        return result


class _DoraConvNdVariant(LoraVariant):
    @staticmethod
    def init_convd_variant(module: _ConvNd, adapter_name: str, dora_layer: nn.Module) -> None:
        if module.lora_magnitude_vector is None:
            # first dora layer being added, add lora_magnitude_vector to the list of learnable parameters
            module.adapter_layer_names = module.adapter_layer_names[:] + ("lora_magnitude_vector",)

        lora_A = module.lora_A[adapter_name].weight
        lora_B = module.lora_B[adapter_name].weight
        scaling = module.scaling[adapter_name]
        dora_layer.update_layer(base_layer=module.get_base_layer(), lora_A=lora_A, lora_B=lora_B, scaling=scaling)
        module.lora_magnitude_vector[adapter_name] = dora_layer

    @staticmethod
    def merge_safe(module: _ConvNd, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)

        # since delta_weight already includes scaling, set it to 1 here
        weight_norm = (
            module.lora_magnitude_vector[active_adapter].get_weight_norm(orig_weight, delta_weight, scaling=1).detach()
        )
        # We need to cache weight_norm because it has to be based on the original weights. We
        # cannot calculate it on the fly based on the merged weights when unmerging because its a
        # different value
        module._cache_store(f"{active_adapter}-weight_norm", weight_norm)
        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm
        new_weight = dora_factor.view(*module._get_dora_factor_view()) * (orig_weight + delta_weight)
        new_weight = new_weight.to(orig_dtype)
        return new_weight

    @staticmethod
    def merge_unsafe(module: _ConvNd, active_adapter: str, orig_weight: torch.Tensor) -> None:
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)
        # since delta_weight already includes scaling, set it to 1 here
        weight_norm = (
            module.lora_magnitude_vector[active_adapter].get_weight_norm(orig_weight, delta_weight, scaling=1).detach()
        )
        # We need to cache weight_norm because it has to be based on the original weights. We
        # cannot calculate it on the fly based on the merged weights when unmerging because its a
        # different value
        module._cache_store(f"{active_adapter}-weight_norm", weight_norm)
        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm
        new_weight = dora_factor.view(*module._get_dora_factor_view()) * (orig_weight.data + delta_weight)
        new_weight = new_weight.to(orig_dtype)
        orig_weight.data = new_weight

    @staticmethod
    def unmerge(module: _ConvNd, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)
        weight_norm = module._cache_pop(f"{active_adapter}-weight_norm")
        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm
        new_weight = orig_weight.data / dora_factor.view(*module._get_dora_factor_view()) - delta_weight
        new_weight = new_weight.to(orig_dtype)
        return new_weight

    @staticmethod
    def forward(module: _ConvNd, active_adapter: str, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]

        if isinstance(dropout, nn.Identity) or not module.training:
            base_result = result
        else:
            x = dropout(x)
            base_result = None

        result = result + module.lora_magnitude_vector[active_adapter](
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=scaling,
            base_layer=module.get_base_layer(),
            base_result=base_result,
        )
        return result


class DoraConv1dVariant(_DoraConvNdVariant):
    @staticmethod
    def init(module: Conv1d, adapter_name: str, **kwargs: Any) -> None:
        dora_layer = DoraConv1dLayer(fan_in_fan_out=False)
        _DoraConvNdVariant.init_convd_variant(module, adapter_name, dora_layer=dora_layer)


class DoraConv2dVariant(_DoraConvNdVariant):
    @staticmethod
    def init(module: Conv2d, adapter_name: str, **kwargs: Any) -> None:
        dora_layer = DoraConv2dLayer(fan_in_fan_out=False)
        _DoraConvNdVariant.init_convd_variant(module, adapter_name, dora_layer=dora_layer)


class DoraConv3dVariant(_DoraConvNdVariant):
    @staticmethod
    def init(module: Conv3d, adapter_name: str, **kwargs: Any) -> None:
        dora_layer = DoraConv3dLayer(fan_in_fan_out=False)
        _DoraConvNdVariant.init_convd_variant(module, adapter_name, dora_layer=dora_layer)


class QALoraLinearVariant(LoraVariant):
    @staticmethod
    def init(module: Linear, adapter_name: str, **kwargs: Any) -> None:
        """
        Initializes QALoRA specific parameters for a given adapter.

        Args:
            module (Linear): The linear module to be adapted.
            adapter_name (str): The name of the adapter.
            **kwargs: Additional keyword arguments.
                qalora_group_size (int): The size of groups for pooling. This is expected to be passed.
        """
        if "qalora_group_size" not in kwargs:
            raise ValueError(
                "`use_qalora=True` requires 'qalora_group_size' to be provided in kwargs."
                " Please ensure it is passed from the LoraConfig."
            )

        if module.in_features is not None and module.in_features % kwargs["qalora_group_size"] != 0:
            raise ValueError(
                f"`use_qalora=True` requires `module.in_features` ({module.in_features}) to be"
                f"divisible by 'qalora_group_size' ({kwargs['qalora_group_size']})"
            )
        qalora_group_size = kwargs["qalora_group_size"]

        if "qalora_group_size" not in module.other_param_names:
            module.other_param_names = module.other_param_names + ("qalora_group_size",)

        if not hasattr(module, "qalora_group_size"):
            module.qalora_group_size = {}
        module.qalora_group_size[adapter_name] = qalora_group_size

        old_lora_A_layer = module.lora_A[adapter_name]
        r = old_lora_A_layer.out_features
        device = old_lora_A_layer.weight.device
        dtype = old_lora_A_layer.weight.dtype

        new_lora_A_layer = nn.Linear(
            old_lora_A_layer.in_features // module.qalora_group_size[adapter_name],
            r,
            bias=False,
            device=device,
            dtype=dtype,
        )
        # module.lora_A[adapter_name] = new_lora_A_layer

    @staticmethod
    def get_delta_weight(module: Linear, active_adapter: str) -> torch.Tensor:
        raise NotImplementedError("QALoRA for GPTQ layers does not support 'get_delta_weight'.")

    @staticmethod
    def merge_safe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("QALoRA for GPTQ layers does not support 'safe_merge'.")

    @staticmethod
    def merge_unsafe(module: Linear, active_adapter: str, orig_weight: torch.Tensor, **kwargs) -> None:
        """
        Merges the QALoRA adapter by first "un-pooling" the LoRA contribution to match
        the original weight dimensions, and then merging this full-resolution adjustment
        into the qzeros.

        Args:
            module (Linear): The quantized linear layer to be merged.
            active_adapter (str): The name of the adapter to merge.
            orig_weight (torch.Tensor): Unused in this variant.
            **kwargs: Additional keyword arguments for merging.
                - amplification_factor (float): Factor to amplify scale. Default: 4.0.
        """
        if not (
            hasattr(module, "base_layer")
            and hasattr(module.base_layer, "qzeros")
            and hasattr(module.base_layer, "scales")
        ):
            return

        # --- 1. Parameter sammeln ---
        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]
        lora_r = module.r[active_adapter]
        lora_alpha = module.lora_alpha[active_adapter]
        scales = module.base_layer.scales
        qzeros_packed = module.base_layer.qzeros

        # Holen Sie die Gruppengrößen aus den Modul-Attributen
        qalora_group_size = module.qalora_group_size[active_adapter]
        gptq_group_size = getattr(module.base_layer, "group_size", 32)

        amplification_factor = kwargs.get("amplification_factor", 4.0)
        effective_scale = (lora_alpha / lora_r) * amplification_factor

        with torch.no_grad():
            # --- 2. LoRA-Beitrag auf der gepoolten Ebene berechnen ---
            # Dies ergibt eine "low-resolution" Delta-Matrix: delta_W_pooled
            # Shape: [out_features, in_features / qalora_group_size]
            delta_W_pooled = lora_B.weight @ lora_A.weight

            # --- 3. "Un-Pooling": Den Beitrag auf die volle Dimension hochskalieren ---
            # Wir wiederholen jede Spalte (die einem Pool entspricht) `qalora_group_size` mal,
            # um die ursprüngliche `in_features`-Dimension wiederherzustellen.
            # Shape: [out_features, in_features]
            delta_W_full = delta_W_pooled.repeat_interleave(qalora_group_size, dim=1)

            # --- 4. Den "Full-Resolution-Shift" für die qzeros berechnen ---
            # Zuerst transponieren wir, um die Form an die `scales` anzupassen.
            # Shape: [in_features, out_features]
            lora_contribution_full = delta_W_full.t()

            # Jetzt gruppieren wir diesen vollen Beitrag gemäß der GPTQ-Gruppengröße,
            # indem wir den Mittelwert über jede Gruppe bilden.
            # Shape: [in_features / gptq_group_size, out_features]
            num_groups = scales.shape[0]
            lora_contribution_grouped = lora_contribution_full.view(num_groups, gptq_group_size, -1).mean(dim=1)

            # --- 5. LoRA-Adjustment im Gewichtsraum berechnen ---
            # adjustment = grouped_lora_contribution * effective_scale
            # Dies ist bereits im Gewichtsraum, nicht im quantisierten Raum
            weight_adjustment = lora_contribution_grouped * effective_scale

            # --- 6. Originale qzeros entpacken und vollständig dequantisieren ---
            bits = getattr(module.base_layer, "bits", 4)
            mask = (2**bits) - 1

            if qzeros_packed.dtype != torch.int32:
                print("Warning: qzeros are not in packed int32 format. Skipping merge.")
                return

            elements_per_packed_val = 32 // bits
            shifts = torch.arange(
                0, elements_per_packed_val * bits, bits, device=qzeros_packed.device, dtype=torch.int32
            ).unsqueeze(0)

            # Entpacken der quantisierten qzeros (0-15 für 4-bit)
            unpacked_qzeros = (qzeros_packed.unsqueeze(-1) >> shifts) & mask
            unpacked_qzeros = unpacked_qzeros.view(qzeros_packed.shape[0], -1)
            unpacked_qzeros = unpacked_qzeros[:, : scales.shape[1]]

            # WICHTIG: Vollständige Dequantisierung der qzeros in den Gewichtsraum
            # Dies konvertiert von quantisierten Werten (0-15) zu echten Zero-Point-Werten
            dequantized_qzeros = unpacked_qzeros.to(torch.float16) * scales

            # --- 7. LoRA-Shift auf dequantisierte qzeros anwenden ---
            # Jetzt können wir den weight_adjustment direkt subtrahieren
            new_qzeros_fp16 = dequantized_qzeros - weight_adjustment.to(torch.float16)

            # --- 8. Alten qzeros-Parameter durch den neuen ersetzen ---
            del module.base_layer.qzeros
            module.base_layer.register_parameter("qzeros", torch.nn.Parameter(new_qzeros_fp16, requires_grad=False))

            print(
                f"Merged adapter into qzeros for layer. New qzeros shape: {new_qzeros_fp16.shape}, dtype: {new_qzeros_fp16.dtype}"
            )

    @staticmethod
    def unmerge(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("QALoRA for GPTQ layers does not support 'unmerge'.")

    @staticmethod
    def forward(module: Linear, active_adapter: str, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        # ================================================================= #
        # TEIL 1: Bestehender QA-LoRA Code (bleibt 100% identisch)
        # ================================================================= #
        lora_A_weight = module.lora_A[active_adapter].weight
        lora_B_weight = module.lora_B[active_adapter].weight
        dropout = module.lora_dropout[active_adapter]
        lora_scaling_coefficient = module.scaling[active_adapter]
        group_size = module.qalora_group_size[active_adapter]

        x_dropped = dropout(x) if module.training and not isinstance(dropout, nn.Identity) else x
        orig_shape = x_dropped.shape

        if len(orig_shape) > 2:
            x_flat = x_dropped.view(-1, module.in_features)
        else:
            x_flat = x_dropped

        batch_size, in_features = x_flat.shape
        pooled_features = in_features // group_size

        x_pooled = x_flat.view(batch_size, pooled_features, group_size).mean(dim=2)
        paper_scaling_factor = in_features / group_size
        x_pooled_scaled = x_pooled * paper_scaling_factor

        lora_A_weight_reshaped = lora_A_weight.view(lora_A_weight.shape[0], pooled_features, group_size)
        lora_A_pooled_weight = lora_A_weight_reshaped.mean(dim=2)

        intermediate = x_pooled_scaled @ lora_A_pooled_weight.t()
        delta = intermediate @ lora_B_weight.t() * lora_scaling_coefficient

        if len(orig_shape) > 2:
            delta = delta.view(orig_shape[:-1] + (delta.size(-1),))
            
        # Das 'result' hier ist der Output der quantisierten Basisschicht.
        # 'delta' ist der Beitrag von QA-LoRA.
        final_result = result + delta
        
        # ================================================================= #
        # TEIL 2: IHR NEUER CODE - DER OUTLIER-BEITRAG (KORRIGIERTE LOGIK)
        # ================================================================= #
        # Prüfen, ob die Outlier-Attribute existieren, die wir injiziert haben.
        if hasattr(module.base_layer, "outlier_weights") and module.base_layer.outlier_indices.numel() > 0:
            
            input_tensor = x_dropped 
            
            # --- 1. Hochpräzisen Beitrag berechnen ---
            # Erstelle eine temporäre Matrix in der Form (out, in), für die die Indizes gelten.
            # WICHTIG: Leite dtype und device direkt von den outlier_weights ab, um den Fehler zu vermeiden.
            temp_hp_matrix = torch.zeros(
                (module.base_layer.out_features, module.base_layer.in_features),
                device=module.base_layer.outlier_weights.device,
                dtype=module.base_layer.outlier_weights.dtype
            )
            
            # Jetzt stimmen die Datentypen überein.
            temp_hp_matrix.view(-1).scatter_(
                0,
                module.base_layer.outlier_indices,
                module.base_layer.outlier_weights
            )
            
            # Transponiere sie zur (in, out) Form für die Multiplikation
            sparse_outlier_matrix_hp = temp_hp_matrix.t()
            
            # Direkte Multiplikation: input @ weight
            outlier_contribution_hp = input_tensor @ sparse_outlier_matrix_hp

            # --- 2. Niedrigpräzisen Beitrag der Outlier berechnen ---
            dequantized_weight_lp = module.base_layer.dequantize_weight() # Shape: (in_features, out_features)
            
            # Erstelle eine temporäre (out, in) Matrix
            temp_lp_matrix = torch.zeros(
                (module.base_layer.out_features, module.base_layer.in_features),
                device=dequantized_weight_lp.device,
                dtype=dequantized_weight_lp.dtype # Leite dtype von der dequantisierten Matrix ab
            )
            
            # Extrahiere die LP-Werte aus der (transponierten) dequantisierten Matrix
            lp_outlier_values = dequantized_weight_lp.t().contiguous().view(-1)[module.base_layer.outlier_indices]
            
            # Fülle die temporäre Matrix
            temp_lp_matrix.view(-1).scatter_(0, module.base_layer.outlier_indices, lp_outlier_values)

            # Transponiere sie zur (in, out) Form
            sparse_outlier_matrix_lp = temp_lp_matrix.t()

            # Direkte Multiplikation: input @ weight
            outlier_contribution_lp = input_tensor @ sparse_outlier_matrix_lp.to(dtype=input_tensor.dtype)
            
            # --- 3. Das Ergebnis korrigieren ---
            # Addiere den hochpräzisen Beitrag und subtrahiere den niedrigpräzisen Beitrag.
            final_result = final_result + outlier_contribution_hp - outlier_contribution_lp
                
        return final_result