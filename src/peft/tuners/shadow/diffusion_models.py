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

from copy import copy as shallow_copy
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn


_FLUX_EMBEDDING_MODULES = ("x_embedder", "context_embedder", "time_guidance_embed", "pos_embed")


class _SharedModuleProxy(nn.Module):
    """Call a shared module without registering it again on the shadow model."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._shared_module: list[nn.Module] = [module]

    @property
    def shared_module(self) -> nn.Module:
        return self._shared_module[0]

    def forward(self, *args: Any, **kwargs: Any):
        return self.shared_module(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "_shared_module":
                raise
            return getattr(self.shared_module, name)


def _select_layer_indices(total: int, count: int) -> list[int]:
    if count < 1 or count > total:
        raise ValueError(f"The reduced Flux shadow needs between 1 and {total} layers, got {count}.")
    if count == 1:
        return [total - 1]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def _copy_overlap(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    if target.ndim != source.ndim or any(
        target_dim > source_dim for target_dim, source_dim in zip(target.shape, source.shape)
    ):
        raise ValueError(
            f"Cannot initialize a Flux shadow tensor of shape {tuple(target.shape)} from {tuple(source.shape)}."
        )
    slices = tuple(slice(0, size) for size in target.shape)
    return source[slices].clone()


def _copy_grouped_rows(
    source: torch.Tensor, *, groups: int, source_group_size: int, target_group_size: int, target_in: int
) -> torch.Tensor:
    return torch.cat(
        [source[i * source_group_size : i * source_group_size + target_group_size, :target_in] for i in range(groups)],
        dim=0,
    ).clone()


def _copy_flux_tensor(
    key: str,
    target: torch.Tensor,
    source: torch.Tensor,
    *,
    base_hidden: int,
    shadow_hidden: int,
    base_intermediate: int,
    shadow_intermediate: int,
) -> torch.Tensor:
    """Copy a full-width Flux tensor while preserving fused projection segment boundaries."""
    if target.shape == source.shape:
        return source.clone()

    if key.endswith("to_qkv_mlp_proj.weight"):
        source_segments = (
            source[:base_hidden],
            source[base_hidden : 2 * base_hidden],
            source[2 * base_hidden : 3 * base_hidden],
            source[3 * base_hidden : 3 * base_hidden + base_intermediate],
            source[3 * base_hidden + base_intermediate :],
        )
        target_sizes = (
            shadow_hidden,
            shadow_hidden,
            shadow_hidden,
            shadow_intermediate,
            shadow_intermediate,
        )
        return torch.cat(
            [segment[:size, :shadow_hidden] for segment, size in zip(source_segments, target_sizes)], dim=0
        ).clone()

    if key == "attn.to_out.weight" or (
        ".single_transformer_blocks." in f".{key}" and key.endswith(".attn.to_out.weight")
    ):
        attention = source[:shadow_hidden, :shadow_hidden]
        mlp = source[:shadow_hidden, base_hidden : base_hidden + shadow_intermediate]
        return torch.cat((attention, mlp), dim=1).clone()

    if key.endswith(("double_stream_modulation_img.linear.weight", "double_stream_modulation_txt.linear.weight")):
        return _copy_grouped_rows(
            source,
            groups=6,
            source_group_size=base_hidden,
            target_group_size=shadow_hidden,
            target_in=shadow_hidden,
        )
    if key.endswith("single_stream_modulation.linear.weight"):
        return _copy_grouped_rows(
            source,
            groups=3,
            source_group_size=base_hidden,
            target_group_size=shadow_hidden,
            target_in=shadow_hidden,
        )
    if key.endswith("norm_out.linear.weight"):
        return _copy_grouped_rows(
            source,
            groups=2,
            source_group_size=base_hidden,
            target_group_size=shadow_hidden,
            target_in=shadow_hidden,
        )
    if key.endswith((".ff.linear_in.weight", ".ff_context.linear_in.weight")):
        return _copy_grouped_rows(
            source,
            groups=2,
            source_group_size=base_intermediate,
            target_group_size=shadow_intermediate,
            target_in=shadow_hidden,
        )
    return _copy_overlap(target, source)


def _load_reduced_flux_state(
    target: nn.Module,
    source: nn.Module,
    *,
    base_hidden: int,
    shadow_hidden: int,
    base_intermediate: int,
    shadow_intermediate: int,
) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    initialized = {}
    for key, target_tensor in target_state.items():
        if key not in source_state:
            raise KeyError(f"Missing pretrained Flux tensor '{key}'.")
        initialized[key] = _copy_flux_tensor(
            key,
            target_tensor,
            source_state[key],
            base_hidden=base_hidden,
            shadow_hidden=shadow_hidden,
            base_intermediate=base_intermediate,
            shadow_intermediate=shadow_intermediate,
        )
        if initialized[key].shape != target_tensor.shape:
            raise ValueError(
                f"Initialized Flux tensor '{key}' has shape {tuple(initialized[key].shape)}, expected "
                f"{tuple(target_tensor.shape)}."
            )
    target.load_state_dict(initialized, strict=True)


class DetachedFluxShadowModel(nn.Module):
    """A compact, detachable Flux shadow initialized from pretrained base blocks."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.config = model.config

    @classmethod
    def from_base(
        cls,
        base_model: nn.Module,
        *,
        num_layers: int,
        num_single_layers: int,
        share_embeddings: bool,
        hidden_size: int | None = None,
        num_attention_heads: int | None = None,
        intermediate_size: int | None = None,
    ) -> "DetachedFluxShadowModel":
        base_double = base_model.transformer_blocks
        base_single = base_model.single_transformer_blocks
        double_indices = _select_layer_indices(len(base_double), num_layers)
        single_indices = _select_layer_indices(len(base_single), num_single_layers)

        base_heads = int(base_model.config.num_attention_heads)
        head_dim = int(base_model.config.attention_head_dim)
        base_hidden = base_heads * head_dim
        base_intermediate = int(base_hidden * float(base_model.config.mlp_ratio))

        if hidden_size is None and num_attention_heads is None:
            shadow_heads = base_heads
            shadow_hidden = base_hidden
        elif hidden_size is None:
            shadow_heads = int(num_attention_heads)
            shadow_hidden = shadow_heads * head_dim
        elif num_attention_heads is None:
            shadow_hidden = int(hidden_size)
            if shadow_hidden % head_dim:
                raise ValueError(
                    f"`shadow_hidden_size` ({shadow_hidden}) must be divisible by the Flux2 base attention head "
                    f"dimension ({head_dim})."
                )
            shadow_heads = shadow_hidden // head_dim
        else:
            shadow_hidden = int(hidden_size)
            shadow_heads = int(num_attention_heads)
            if shadow_hidden != shadow_heads * head_dim:
                raise ValueError(
                    "A pretrained-initialized Flux2 shadow must preserve the base attention head dimension: "
                    f"expected shadow_hidden_size == shadow_num_attention_heads * {head_dim}, got "
                    f"{shadow_hidden} and {shadow_heads}."
                )

        if not 1 <= shadow_heads <= base_heads:
            raise ValueError(
                f"`shadow_num_attention_heads` must be between 1 and the base model's {base_heads}, got {shadow_heads}."
            )
        shadow_intermediate = base_intermediate if intermediate_size is None else int(intermediate_size)
        if not 1 <= shadow_intermediate <= base_intermediate:
            raise ValueError(
                f"`shadow_intermediate_size` must be between 1 and the base model's {base_intermediate}, got "
                f"{shadow_intermediate}."
            )

        reduced_config = dict(base_model.config)
        reduced_config["num_layers"] = num_layers
        reduced_config["num_single_layers"] = num_single_layers
        reduced_config["num_attention_heads"] = shadow_heads
        reduced_config["attention_head_dim"] = head_dim
        reduced_config["mlp_ratio"] = shadow_intermediate / shadow_hidden
        reduced = base_model.__class__.from_config(reduced_config)

        # Copy every pretrained component, preserving logical segments in fused modulation/QKV/MLP projections.
        shell_source = shallow_copy(base_model)
        shell_source._modules = base_model._modules.copy()
        shell_source.transformer_blocks = nn.ModuleList()
        shell_source.single_transformer_blocks = nn.ModuleList()
        shell_target = shallow_copy(reduced)
        shell_target._modules = reduced._modules.copy()
        shell_target.transformer_blocks = nn.ModuleList()
        shell_target.single_transformer_blocks = nn.ModuleList()
        _load_reduced_flux_state(
            shell_target,
            shell_source,
            base_hidden=base_hidden,
            shadow_hidden=shadow_hidden,
            base_intermediate=base_intermediate,
            shadow_intermediate=shadow_intermediate,
        )
        for target, source_idx in zip(reduced.transformer_blocks, double_indices):
            source = getattr(base_double[source_idx], "base_layer", base_double[source_idx])
            _load_reduced_flux_state(
                target,
                source,
                base_hidden=base_hidden,
                shadow_hidden=shadow_hidden,
                base_intermediate=base_intermediate,
                shadow_intermediate=shadow_intermediate,
            )
        for target, source_idx in zip(reduced.single_transformer_blocks, single_indices):
            source = getattr(base_single[source_idx], "base_layer", base_single[source_idx])
            _load_reduced_flux_state(
                target,
                source,
                base_hidden=base_hidden,
                shadow_hidden=shadow_hidden,
                base_intermediate=base_intermediate,
                shadow_intermediate=shadow_intermediate,
            )

        if share_embeddings and shadow_hidden == base_hidden:
            for name in _FLUX_EMBEDDING_MODULES:
                base_embedding = getattr(base_model, name, None)
                if isinstance(base_embedding, nn.Module):
                    setattr(reduced, name, _SharedModuleProxy(base_embedding))

        if getattr(base_model, "gradient_checkpointing", False):
            reduced.enable_gradient_checkpointing()
        return cls(reduced)

    def forward(self, input_ids=None, inputs_embeds=None, block_kwargs: dict[str, Any] | None = None, **kwargs):
        """Produce the initial shadow state using the reduced pretrained single-stream stack."""
        if inputs_embeds is None:
            raise ValueError("The reduced Flux shadow backbone requires `inputs_embeds`.")
        block_kwargs = block_kwargs or {}
        hidden_states = inputs_embeds
        shadow_hidden = int(self.model.config.num_attention_heads) * int(self.model.config.attention_head_dim)
        if hidden_states.size(-1) < shadow_hidden:
            raise ValueError(
                f"The Flux shadow received hidden size {hidden_states.size(-1)}, smaller than its {shadow_hidden}."
            )
        hidden_states = hidden_states[..., :shadow_hidden]
        encoder_hidden_states = block_kwargs.get("encoder_hidden_states")
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states[..., :shadow_hidden]
        temb_mod = block_kwargs.get("temb_mod")
        if temb_mod is not None and temb_mod.size(-1) != 3 * shadow_hidden:
            base_hidden = temb_mod.size(-1) // 3
            temb_mod = torch.cat(
                [temb_mod[..., i * base_hidden : i * base_hidden + shadow_hidden] for i in range(3)], dim=-1
            )
        for block in self.model.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod=temb_mod,
                image_rotary_emb=block_kwargs.get("image_rotary_emb"),
                joint_attention_kwargs=block_kwargs.get("joint_attention_kwargs"),
            )
        return SimpleNamespace(last_hidden_state=hidden_states, past_key_values=None)


def validate_pretrained_flux_shadow(base_model: nn.Module, shadow_model: nn.Module) -> DetachedFluxShadowModel:
    """Validate and wrap an independently pretrained Flux2 transformer for attached Shadow use."""
    exact_attributes = (
        "in_channels",
        "out_channels",
        "joint_attention_dim",
        "timestep_guidance_channels",
        "attention_head_dim",
        "axes_dims_rope",
        "guidance_embeds",
    )
    for attribute in exact_attributes:
        base_value = getattr(base_model.config, attribute, None)
        shadow_value = getattr(shadow_model.config, attribute, None)
        normalized_base = tuple(base_value) if isinstance(base_value, (list, tuple)) else base_value
        normalized_shadow = tuple(shadow_value) if isinstance(shadow_value, (list, tuple)) else shadow_value
        if normalized_base != normalized_shadow:
            raise ValueError(
                f"The pretrained Flux2 shadow has incompatible `{attribute}`: "
                f"expected {base_value!r}, got {shadow_value!r}."
            )

    bounded_attributes = ("num_attention_heads", "num_layers", "num_single_layers")
    for attribute in bounded_attributes:
        base_value = int(getattr(base_model.config, attribute))
        shadow_value = int(getattr(shadow_model.config, attribute))
        if not 1 <= shadow_value <= base_value:
            raise ValueError(
                f"The pretrained Flux2 shadow `{attribute}` must be between 1 and the base value "
                f"{base_value}, got {shadow_value}."
            )
    base_hidden = int(base_model.config.num_attention_heads) * int(base_model.config.attention_head_dim)
    shadow_hidden = int(shadow_model.config.num_attention_heads) * int(shadow_model.config.attention_head_dim)
    if shadow_hidden > base_hidden:
        raise ValueError(
            f"The pretrained Flux2 shadow hidden size ({shadow_hidden}) cannot exceed the base hidden size "
            f"({base_hidden})."
        )
    base_intermediate = int(base_hidden * float(base_model.config.mlp_ratio))
    shadow_intermediate = int(shadow_hidden * float(shadow_model.config.mlp_ratio))
    if shadow_intermediate > base_intermediate:
        raise ValueError(
            f"The pretrained Flux2 shadow intermediate size ({shadow_intermediate}) cannot exceed the base "
            f"intermediate size ({base_intermediate})."
        )
    return DetachedFluxShadowModel(shadow_model)


def freeze_flux_stem_embeddings(backbone: DetachedFluxShadowModel) -> None:
    """Freeze pretrained input/conditioning stems while leaving blocks and output layers trainable."""
    for name in _FLUX_EMBEDDING_MODULES:
        module = getattr(backbone.model, name, None)
        if isinstance(module, nn.Module):
            module.requires_grad_(False)


class DiffusionShadowBackend:
    """Architecture-specific operations needed by an architecture-alike diffusion shadow."""

    @classmethod
    def supports(cls, model: nn.Module) -> bool:
        raise NotImplementedError

    @classmethod
    def build_mirror(cls, model: nn.Module, config: Any) -> nn.Module:
        raise NotImplementedError

    @classmethod
    def load_pretrained(cls, model: nn.Module, model_id_or_path: str) -> nn.Module:
        raise NotImplementedError

    @classmethod
    def prepare_block_kwargs(cls, args: tuple, kwargs: dict[str, Any]) -> dict[str, Any]:
        return {}

    @classmethod
    def forward_backbone(cls, backbone: nn.Module, inputs_embeds: torch.Tensor, block_kwargs: dict[str, Any]) -> Any:
        return backbone(inputs_embeds=inputs_embeds, **block_kwargs)

    @classmethod
    def initialize_projection(cls, projection: nn.Linear, shadow_hidden: int) -> None:
        pass

    @classmethod
    def freeze_embeddings(cls, backbone: nn.Module) -> None:
        pass


class Flux2ShadowBackend(DiffusionShadowBackend):
    """Reduced, pretrained-initialized Flux2 shadow backend."""

    @classmethod
    def supports(cls, model: nn.Module) -> bool:
        return model.__class__.__name__ == "Flux2Transformer2DModel"

    @classmethod
    def build_mirror(cls, model: nn.Module, config: Any) -> DetachedFluxShadowModel:
        num_layers = config.shadow_num_hidden_layers or 1
        return DetachedFluxShadowModel.from_base(
            model,
            num_layers=num_layers,
            num_single_layers=2 * num_layers,
            share_embeddings=config.share_embeddings,
            hidden_size=config.shadow_hidden_size,
            num_attention_heads=config.shadow_num_attention_heads,
            intermediate_size=config.shadow_intermediate_size,
        )

    @classmethod
    def load_pretrained(cls, model: nn.Module, model_id_or_path: str) -> DetachedFluxShadowModel:
        from diffusers import Flux2Transformer2DModel

        shadow_model = Flux2Transformer2DModel.from_pretrained(model_id_or_path)
        return validate_pretrained_flux_shadow(model, shadow_model)

    @classmethod
    def prepare_block_kwargs(cls, args: tuple, kwargs: dict[str, Any]) -> dict[str, Any]:
        block_kwargs = dict(kwargs)
        # Gradient checkpointing invokes Flux2 single-stream blocks positionally.
        for name, value in zip(
            ("encoder_hidden_states", "temb_mod", "image_rotary_emb", "joint_attention_kwargs"),
            args[1:],
        ):
            block_kwargs.setdefault(name, value)
        return block_kwargs

    @classmethod
    def forward_backbone(
        cls, backbone: DetachedFluxShadowModel, inputs_embeds: torch.Tensor, block_kwargs: dict[str, Any]
    ) -> Any:
        return backbone(inputs_embeds=inputs_embeds, block_kwargs=block_kwargs)

    @classmethod
    def initialize_projection(cls, projection: nn.Linear, shadow_hidden: int) -> None:
        with torch.no_grad():
            projection.weight.zero_()
            projection.weight[:shadow_hidden, :shadow_hidden].copy_(
                torch.eye(shadow_hidden, dtype=projection.weight.dtype, device=projection.weight.device)
            )

    @classmethod
    def freeze_embeddings(cls, backbone: nn.Module) -> None:
        freeze_flux_stem_embeddings(backbone)


_DIFFUSION_SHADOW_BACKENDS: tuple[type[DiffusionShadowBackend], ...] = (Flux2ShadowBackend,)


def get_diffusion_shadow_backend(model: nn.Module) -> type[DiffusionShadowBackend] | None:
    """Return the registered architecture backend for `model`, or `None` for the generic MLP fallback."""
    for backend in _DIFFUSION_SHADOW_BACKENDS:
        if backend.supports(model):
            return backend
