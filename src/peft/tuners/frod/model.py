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

from __future__ import annotations

import warnings
from collections import defaultdict

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_FROD_TARGET_MODULES_MAPPING

from .._buffer_dict import BufferDict
from ..tuners_utils import _maybe_include_all_linear_layers
from .config import FrodConfig
from .layer import FrodLayer, Linear


def _category_from_key(key: str) -> str:
    """Infer the projection-sharing category from a dotted module key.

    FRoD shares projection buffers across modules that play the same role in different transformer blocks. This helper
    assumes keys follow the dotted paths returned by `named_modules()` and derives the role from the final path
    components. For example, `encoder.layer.0.attention.self.query` maps to `self_query`, while
    `vision_model.encoder.layers.0.self_attn.q_proj` maps to `self_attn_q_proj`. The BERT-style attention output key
    `encoder.layer.0.attention.output.dense` is normalized to `attention_output` so it does not collide with MLP
    `output.dense` modules.
    """
    parts = key.split(".")
    if len(parts) == 1:
        return parts[0]
    if parts[-2].isdigit():
        return parts[-1]
    category = f"{parts[-2]}_{parts[-1]}"
    if (category == "output_dense") and (len(parts) >= 3) and (parts[-3] == "attention"):
        return "attention_output"
    return category


def _layer_index_from_key(key: str, fallback: int) -> int:
    """Infer the transformer block index from a dotted module key.

    Many decoder and vision models use paths like `model.layers.3.self_attn.q_proj`, so the first preference is the
    integer immediately after a `layers` path component. Encoder models often use paths like
    `encoder.layer.11.attention.self.query`; for those, the first numeric path component is used. If no numeric layer
    id is present, e.g. for `classifier.dense`, `fallback` keeps the projection initialization order deterministic.
    """
    parts = key.split(".")
    if "layers" in parts:
        try:
            return int(parts[parts.index("layers") + 1])
        except (ValueError, IndexError):
            pass
    for part in parts:
        if part.isdigit():
            return int(part)
    return fallback


def _projection_from_weights(matrices: list[torch.Tensor], regularization_alpha: float) -> torch.Tensor:
    stacked = torch.cat(matrices, dim=0)
    if stacked.shape[0] < stacked.shape[1]:
        _, _, vh = torch.linalg.svd(stacked, full_matrices=True)
        return vh.T

    q_matrix, r_matrix = torch.linalg.qr(stacked)
    q_slices = []
    start = 0
    for matrix in matrices:
        rows = matrix.shape[0]
        q_slices.append(q_matrix[start : start + rows, :])
        start += rows

    dim = r_matrix.shape[1]
    t_pi = torch.zeros((dim, dim), dtype=r_matrix.dtype)
    # Layers of the same projection category can be highly correlated; this ridge term keeps the inverse stable.
    for q_slice in q_slices:
        q_term = q_slice.T @ q_slice + regularization_alpha * torch.eye(dim, dtype=r_matrix.dtype)
        t_pi += torch.linalg.inv(q_term)
    t_pi /= len(q_slices)

    _, eigenvectors = torch.linalg.eigh(t_pi)
    return r_matrix.T @ eigenvectors


class FrodModel(BaseTuner):
    prefix: str = "frod_"
    tuner_layer_cls = FrodLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_FROD_TARGET_MODULES_MAPPING

    def _init_frod_projections(self, config: FrodConfig, adapter_name: str) -> None:
        weights = defaultdict(dict)
        model_config = self.get_model_config(self.model)
        peft_config = self._prepare_adapter_config(config, model_config)
        peft_config = _maybe_include_all_linear_layers(peft_config, self.model)

        fallback_index = 0
        for key, module in self.model.named_modules():
            if not self._check_target_module_exists(peft_config, key):
                continue

            if isinstance(module, nn.Linear):
                weight = module.weight
            elif isinstance(module, Conv1D):
                weight = module.weight.T
            else:
                continue

            category = _category_from_key(key)
            layer_idx = _layer_index_from_key(key, fallback_index)
            fallback_index += 1
            weights[layer_idx][category] = weight

        if not weights:
            raise ValueError(
                "No layer types compatible with FRoD were found. Please check `peft_config.target_modules`."
            )

        # BaseTuner.__init__() enters the pre-injection flow before a FrodModel subclass
        # could assign ModuleDicts after super().__init__(), so create these containers lazily here.
        if not hasattr(self, "frod_V"):
            self.frod_V = nn.ModuleDict()
            self.frod_s_indices = nn.ModuleDict()
            self.frod_s_size = nn.ModuleDict()

        generator = torch.Generator(device="cpu").manual_seed(config.projection_prng_key)
        categories = {category for layer_dict in weights.values() for category in layer_dict}
        for category in sorted(categories):
            matrices = [
                layer_dict[category].detach().to(torch.float32).cpu()
                for _, layer_dict in sorted(weights.items())
                if category in layer_dict
            ]
            if not matrices:
                continue

            v_matrix = _projection_from_weights(matrices, config.regularization_alpha)
            example_weight = next(layer_dict[category] for layer_dict in weights.values() if category in layer_dict)
            v_tensor = v_matrix.to(dtype=example_weight.dtype, device="cpu")

            if category not in self.frod_V:
                self.frod_V[category] = BufferDict({}, persistent=config.save_projection)
            self.frod_V[category][adapter_name] = v_tensor

            in_dim = v_tensor.shape[0]
            rows, cols = torch.meshgrid(torch.arange(in_dim), torch.arange(in_dim), indexing="ij")
            mask_indices = torch.stack([rows.flatten(), cols.flatten()], dim=1)
            non_diag_indices = mask_indices[mask_indices[:, 0] != mask_indices[:, 1]]
            nnz = min(int(in_dim * in_dim * config.sparse_rate), non_diag_indices.shape[0])
            if (config.sparse_rate > 0) and (non_diag_indices.shape[0] > 0):
                nnz = max(1, nnz)
            if nnz:
                perm = torch.randperm(non_diag_indices.shape[0], generator=generator)[:nnz]
                indices = non_diag_indices[perm].t().contiguous()
            else:
                indices = torch.empty(2, 0, dtype=torch.long)
            size = torch.tensor([in_dim, in_dim], dtype=torch.long)

            if category not in self.frod_s_indices:
                self.frod_s_indices[category] = BufferDict({}, persistent=config.save_projection)
            self.frod_s_indices[category][adapter_name] = indices.to(torch.long)
            if category not in self.frod_s_size:
                self.frod_s_size[category] = BufferDict({}, persistent=config.save_projection)
            self.frod_s_size[category][adapter_name] = size

    def _pre_injection_hook(self, model: nn.Module, config: FrodConfig, adapter_name: str) -> None:
        self._init_frod_projections(config, adapter_name)

    def _check_new_adapter_config(self, config: FrodConfig) -> None:
        super()._check_new_adapter_config(config)

        for existing_config in self.peft_config.values():
            if existing_config is config:
                continue
            if existing_config.projection_prng_key != config.projection_prng_key:
                raise ValueError(
                    f"FRoD projection initialization key must be the same for all adapters. Got "
                    f"{config.projection_prng_key=} but previous config had "
                    f"{existing_config.projection_prng_key}."
                )

        save_projection_values = sorted({config.save_projection for config in self.peft_config.values()})
        if len(save_projection_values) > 1:
            raise ValueError(
                "FRoD projection weights must be saved for all adapters or none, but got multiple different values: "
                f"{save_projection_values}"
            )

    def _create_and_replace(
        self,
        frod_config: FrodConfig,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        category = _category_from_key(current_key)
        if category not in self.frod_V:
            self._init_frod_projections(frod_config, adapter_name)
        bias = hasattr(target, "bias") and target.bias is not None

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name,
                self.frod_V[category],
                self.frod_s_indices[category],
                self.frod_s_size[category],
                config=frod_config,
            )
        else:
            new_module = self._create_new_module(
                frod_config,
                self.frod_V[category],
                self.frod_s_indices[category],
                self.frod_s_size[category],
                adapter_name,
                target,
                bias=bias,
            )
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(
        frod_config: FrodConfig,
        frod_V,
        frod_s_indices,
        frod_s_size,
        adapter_name,
        target,
        **kwargs,
    ):
        bias = kwargs.pop("bias", False)

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if frod_config.fan_in_fan_out:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                frod_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not frod_config.fan_in_fan_out:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
                )
                frod_config.fan_in_fan_out = True
        else:
            raise TypeError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `transformers.pytorch_utils.Conv1D`."
            )

        return Linear(
            target,
            frod_V,
            frod_s_indices,
            frod_s_size,
            adapter_name,
            config=frod_config,
            bias=bias,
            **kwargs,
        )
