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

import math
import warnings
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import gather_params_ctx

from .config import CortexConfig


def is_dense_ffn(module: nn.Module) -> bool:
    """Whether the module is a dense FFN computing `down_proj(act_fn(gate_proj(x)) * up_proj(x))`.

    This is the convention followed by the MLPs of most Transformers decoder models (Llama, Mistral, Qwen, Gemma, ...).
    The projections must be `nn.Linear` layers. Without `gate_proj`, the FFN is expected to compute
    `down_proj(act_fn(up_proj(x)))`. Instead of separate gate and up projections, the FFN may have a fused
    `gate_up_proj`, whose output consists of the gate half followed by the up half. Some FFNs use variants of the
    clamped SwiGLU of GPT-OSS instead of `act_fn(gate) * up`, see `_apply_dense_gate`.

    Cortex relies on this formula when merging, so it cannot be used with FFNs that compute something else. Therefore,
    the FFN must not have submodules other than the projections and the activation, e.g. a normalization layer between
    the projections, or parameters or buffers of its own, e.g. an output scale, since Cortex cannot know how they are
    used.
    """
    if hasattr(module, "gate_up_proj"):
        projections = [module.gate_up_proj, getattr(module, "down_proj", None)]
    else:
        projections = [getattr(module, "up_proj", None), getattr(module, "down_proj", None)]
        if hasattr(module, "gate_proj"):
            projections.append(module.gate_proj)
    has_projections = all(isinstance(projection, nn.Linear) for projection in projections)
    # The offset SwiGLU of MiniMax-M3-VL does not use `act_fn`.
    has_activation = callable(getattr(module, "act_fn", None)) or hasattr(module, "swiglu_alpha")

    # Gemma3n sparsifies the activations of some layers based on statistics over all channels. Adding channels would
    # change these statistics, so the merged model would differ from the trained one.
    is_sparse = getattr(module, "activation_sparsity", 0) > 0
    # E.g. the dense FFNs of Inkling multiply their output by a learned scale. Merged channels would be scaled as well,
    # while the channels of the adapter are not.
    own_tensors = [*module.parameters(recurse=False), *module.buffers(recurse=False)]
    # E.g. the dense FFNs of BitNet normalize the activations before the down projection, over all channels.
    ffn_modules = {"gate_proj", "up_proj", "down_proj", "gate_up_proj", "act_fn"}
    other_modules = [name for name, _ in module.named_children() if name not in ffn_modules]

    return has_projections and has_activation and not is_sparse and not own_tensors and not other_modules


def is_packed_experts(module: nn.Module) -> bool:
    """Whether the module holds packed MoE experts that follow the Transformers experts interface.

    Transformers decorates these classes with `use_experts_implementation`, which sets the layout attributes
    `has_gate`, `has_bias`, `is_transposed`, and `is_concatenated` and provides the `_apply_gate` method. The classes
    themselves set `num_experts`. Their weights are 3D parameters with one slice per expert.
    """
    layout_attributes = ("num_experts", "has_gate", "has_bias", "is_transposed", "is_concatenated")
    return all(hasattr(module, attr) for attr in layout_attributes) and isinstance(
        getattr(module, "down_proj", None), nn.Parameter
    )


def _apply_dense_gate(ffn: nn.Module, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Combine the gate and up activations of channels in the same way as the given dense FFN.

    Most FFNs compute `act_fn(gate) * up`. Some FFNs use variants of the clamped SwiGLU of GPT-OSS, which are
    recognized by the attributes that store their parameters: The FFNs of DeepSeek-V4 and GLM5 Next clamp the gate and
    up activations to `limit` or `swiglu_limit`, and the FFNs of MiniMax-M3-VL additionally compute `(up + 1) * gate *
    sigmoid(swiglu_alpha * gate)` instead of `act_fn(gate) * up`.
    """
    limit = getattr(ffn, "swiglu_limit", getattr(ffn, "limit", None))
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)

    alpha = getattr(ffn, "swiglu_alpha", None)
    if alpha is not None:
        return (up + 1) * gate * torch.sigmoid(alpha * gate)
    return ffn.act_fn(gate) * up


def _sample_indices(population: int, num_samples: int) -> torch.Tensor:
    """Sample indices without replacement, or with replacement if more samples than indices are requested."""
    if num_samples <= population:
        return torch.randperm(population)[:num_samples]
    return torch.randint(population, (num_samples,))


def _get_full_shape(parameter: nn.Parameter) -> tuple[int, ...]:
    # DeepSpeed ZeRO-3 partitions the parameters and stores their full shape in `ds_shape`.
    return tuple(getattr(parameter, "ds_shape", parameter.shape))


def _replace_parameter(module: nn.Module, name: str, value: torch.Tensor) -> None:
    """Replace a parameter of the base layer by a tensor of a different shape.

    Merging changes the shape of the base weights, so they cannot be updated in place. The new parameter keeps the
    dtype, device, and `requires_grad` of the old one.
    """
    old = getattr(module, name)
    new = value.to(device=old.device, dtype=old.dtype).contiguous()
    setattr(module, name, nn.Parameter(new, requires_grad=old.requires_grad))


def _update_intermediate_size(module: nn.Module, delta: int) -> None:
    # Some FFNs store their intermediate size as an attribute. Keep it consistent with the merged weights.
    for name in ("intermediate_size", "intermediate_dim"):
        value = getattr(module, name, None)
        if isinstance(value, int):
            setattr(module, name, value + delta)


def _cat_channels(tensor: torch.Tensor, added: torch.Tensor, dim: int, concatenated: bool) -> torch.Tensor:
    """Append the rows of new channels along `dim` of a (gate_)up weight or bias.

    With `concatenated`, the tensor holds the gate rows of all channels followed by their up rows, like a fused
    `gate_up_proj`, and `added` has the same layout. The new gate and up rows are then inserted at the end of the
    respective half. Otherwise, the rows of each channel are adjacent, so the new rows are appended.
    """
    if not concatenated:
        return torch.cat([tensor, added], dim=dim)

    gate, up = tensor.chunk(2, dim=dim)
    added_gate, added_up = added.chunk(2, dim=dim)
    return torch.cat([gate, added_gate, up, added_up], dim=dim)


def _remove_last_channels(tensor: torch.Tensor, num_rows: int, dim: int, concatenated: bool) -> torch.Tensor:
    """Remove the rows of the last channels along `dim` of a (gate_)up weight or bias, undoing `_cat_channels`."""
    if not concatenated:
        return tensor.narrow(dim, 0, tensor.shape[dim] - num_rows).clone()

    gate, up = tensor.chunk(2, dim=dim)
    width = gate.shape[dim] - num_rows // 2
    return torch.cat([gate.narrow(dim, 0, width), up.narrow(dim, 0, width)], dim=dim)


def _get_dense_gate_up(ffn: nn.Module) -> dict[str, tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Return the weights and biases of the gate and up projections of a dense FFN, splitting a fused projection."""
    if hasattr(ffn, "gate_up_proj"):
        gate_weight, up_weight = ffn.gate_up_proj.weight.chunk(2)
        bias = ffn.gate_up_proj.bias
        gate_bias, up_bias = bias.chunk(2) if bias is not None else (None, None)
        return {"gate": (gate_weight, gate_bias), "up": (up_weight, up_bias)}

    weights = {"up": (ffn.up_proj.weight, ffn.up_proj.bias)}
    if hasattr(ffn, "gate_proj"):
        weights["gate"] = (ffn.gate_proj.weight, ffn.gate_proj.bias)
    return weights


def _append_rows(linear: nn.Linear, weight: torch.Tensor, bias: Optional[torch.Tensor], concatenated: bool) -> None:
    """Append output features to a linear layer, see `_cat_channels`."""
    if linear.bias is not None:
        if bias is None:
            # Channels without biases, i.e. the zero channels used for padding (see `pad_ffn`) and channels added with
            # `cortex_bias="none"`, get zero biases.
            bias = linear.bias.new_zeros(weight.shape[0])
        _replace_parameter(linear, "bias", _cat_channels(linear.bias, bias.to(linear.bias), 0, concatenated))
    _replace_parameter(linear, "weight", _cat_channels(linear.weight, weight.to(linear.weight), 0, concatenated))
    linear.out_features = linear.weight.shape[0]


def _remove_rows(linear: nn.Linear, num_rows: int, concatenated: bool) -> None:
    """Remove the last output features of a linear layer, undoing `_append_rows`."""
    if linear.bias is not None:
        _replace_parameter(linear, "bias", _remove_last_channels(linear.bias, num_rows, 0, concatenated))
    _replace_parameter(linear, "weight", _remove_last_channels(linear.weight, num_rows, 0, concatenated))
    linear.out_features = linear.weight.shape[0]


def _append_dense_channels(
    ffn: nn.Module,
    up: torch.Tensor,
    down: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
    up_bias: Optional[torch.Tensor] = None,
    gate_bias: Optional[torch.Tensor] = None,
) -> None:
    """Append channels to a dense FFN.

    The gate and up weights are given as `(num_channels, in_features)`, the down weights as `(out_features,
    num_channels)`. Missing biases are treated as zero.
    """
    # The added channels are additional output features of the gate and up projections ...
    if hasattr(ffn, "gate_up_proj"):
        bias = torch.cat([gate_bias, up_bias]) if up_bias is not None else None
        _append_rows(ffn.gate_up_proj, torch.cat([gate, up]), bias, concatenated=True)
    else:
        _append_rows(ffn.up_proj, up, up_bias, concatenated=False)
        if gate is not None:
            _append_rows(ffn.gate_proj, gate, gate_bias, concatenated=False)

    # ... and additional input features of the down projection.
    weight = torch.cat([ffn.down_proj.weight, down.to(ffn.down_proj.weight)], dim=1)
    _replace_parameter(ffn.down_proj, "weight", weight)
    ffn.down_proj.in_features = weight.shape[1]

    _update_intermediate_size(ffn, down.shape[1])


def _remove_dense_channels(ffn: nn.Module, num_channels: int) -> None:
    """Remove the last channels of a dense FFN, undoing `_append_dense_channels`."""
    if hasattr(ffn, "gate_up_proj"):
        _remove_rows(ffn.gate_up_proj, 2 * num_channels, concatenated=True)
    else:
        _remove_rows(ffn.up_proj, num_channels, concatenated=False)
        if hasattr(ffn, "gate_proj"):
            _remove_rows(ffn.gate_proj, num_channels, concatenated=False)

    width = ffn.down_proj.weight.shape[1] - num_channels
    _replace_parameter(ffn.down_proj, "weight", ffn.down_proj.weight[:, :width].clone())
    ffn.down_proj.in_features = width

    _update_intermediate_size(ffn, -num_channels)


def _get_up_name(experts: nn.Module) -> str:
    return "gate_up_proj" if experts.has_gate else "up_proj"


def _get_packed_weight(experts: nn.Module, name: str) -> torch.Tensor:
    """Return a packed weight of the experts as `(num_experts, out_features, in_features)`."""
    weight = getattr(experts, name)
    return weight.transpose(-1, -2) if experts.is_transposed else weight


def _set_packed_weight(experts: nn.Module, name: str, weight: torch.Tensor) -> None:
    """Set a packed weight of the experts, given as `(num_experts, out_features, in_features)`."""
    _replace_parameter(experts, name, weight.transpose(-1, -2) if experts.is_transposed else weight)


def _append_expert_channels(
    experts: nn.Module, up: torch.Tensor, down: torch.Tensor, up_bias: Optional[torch.Tensor] = None
) -> None:
    """Append channels to every expert of packed experts.

    The (gate_)up weights are given as `(num_experts, rows, in_features)` in the gate/up layout of the experts, the
    down weights as `(num_experts, out_features, num_channels)`. A missing bias is treated as zero.
    """
    concatenated = experts.has_gate and experts.is_concatenated
    up_name = _get_up_name(experts)
    up_weight = _get_packed_weight(experts, up_name)
    _set_packed_weight(experts, up_name, _cat_channels(up_weight, up.to(up_weight), -2, concatenated))

    base_bias = getattr(experts, up_name + "_bias", None)
    if base_bias is not None:
        if up_bias is None:
            # Channels without biases get zero biases, see `_append_rows`.
            up_bias = base_bias.new_zeros(up.shape[:-1])
        bias = _cat_channels(base_bias, up_bias.to(base_bias), -1, concatenated)
        _replace_parameter(experts, up_name + "_bias", bias)

    down_weight = _get_packed_weight(experts, "down_proj")
    _set_packed_weight(experts, "down_proj", torch.cat([down_weight, down.to(down_weight)], dim=-1))

    _update_intermediate_size(experts, down.shape[-1])


def _remove_expert_channels(experts: nn.Module, num_channels: int) -> None:
    """Remove the last channels of every expert of packed experts, undoing `_append_expert_channels`."""
    concatenated = experts.has_gate and experts.is_concatenated
    num_rows = 2 * num_channels if experts.has_gate else num_channels
    up_name = _get_up_name(experts)
    up_weight = _get_packed_weight(experts, up_name)
    _set_packed_weight(experts, up_name, _remove_last_channels(up_weight, num_rows, -2, concatenated))

    base_bias = getattr(experts, up_name + "_bias", None)
    if base_bias is not None:
        _replace_parameter(experts, up_name + "_bias", _remove_last_channels(base_bias, num_rows, -1, concatenated))

    down_weight = _get_packed_weight(experts, "down_proj")
    width = down_weight.shape[-1] - num_channels
    _set_packed_weight(experts, "down_proj", down_weight[..., :width].clone())

    _update_intermediate_size(experts, -num_channels)


def get_ffn_width(module: nn.Module) -> Optional[int]:
    """Return the number of channels of a dense FFN or packed experts that Cortex supports, or `None`."""
    if is_packed_experts(module):
        return _get_full_shape(module.down_proj)[-2 if module.is_transposed else -1]
    if is_dense_ffn(module):
        return _get_full_shape(module.down_proj.weight)[1]
    return None


@torch.no_grad()
def pad_ffn(module: nn.Module, width: int) -> None:
    """Pad a dense FFN or packed experts with zero channels to the given, larger width.

    The down weights of the zero channels are zero, so they don't change the output of the FFN, whatever its activation
    function is.
    """
    num_channels = width - get_ffn_width(module)
    if is_packed_experts(module):
        up_weight = _get_packed_weight(module, _get_up_name(module))
        down_weight = _get_packed_weight(module, "down_proj")
        num_rows = 2 * num_channels if module.has_gate else num_channels
        up = up_weight.new_zeros(up_weight.shape[0], num_rows, up_weight.shape[-1])
        down = down_weight.new_zeros(down_weight.shape[0], down_weight.shape[1], num_channels)
        _append_expert_channels(module, up, down)
    else:
        has_fused_gate = hasattr(module, "gate_up_proj")
        in_features = module.gate_up_proj.in_features if has_fused_gate else module.up_proj.in_features
        up = module.down_proj.weight.new_zeros(num_channels, in_features)
        down = module.down_proj.weight.new_zeros(module.down_proj.out_features, num_channels)
        has_gate = has_fused_gate or hasattr(module, "gate_proj")
        # The gate weights of the zero channels are zero as well.
        _append_dense_channels(module, up, down, gate=up if has_gate else None)


@torch.no_grad()
def unpad_ffn(module: nn.Module, width: int) -> None:
    """Remove the last channels of a dense FFN or packed experts down to the given width, undoing `pad_ffn`."""
    num_channels = get_ffn_width(module) - width
    if is_packed_experts(module):
        _remove_expert_channels(module, num_channels)
    else:
        _remove_dense_channels(module, num_channels)


class CortexLayer(BaseTunerLayer):
    """Base class of the Cortex layers, which wrap a whole FFN and add trainable channels to it.

    Merging appends the channels of an adapter to the intermediate dimension of the base FFN, so a merged FFN is wider
    than the original one. Unmerging removes the appended channels again. Since channels are always appended at the
    end, adapters are unmerged in the reverse order in which they were merged. The down bias of an adapter does not
    belong to its channels, so it is added to the down bias of the base FFN instead, and subtracted again when
    unmerging.
    """

    def __init__(self, base_layer: nn.Module) -> None:
        self.base_layer = base_layer
        self._disable_adapters = False
        self.merged_adapters = []

    def _get_base_layer_device_and_dtype(self, base_layer):
        # The base layer is a whole FFN without a single `weight`, so use its down projection as reference.
        down_proj = base_layer.down_proj
        weight = down_proj if isinstance(down_proj, nn.Parameter) else down_proj.weight
        return weight.device, weight.dtype

    def _get_adapter_parameters(self, adapter_name: str) -> list[nn.Parameter]:
        parameters = []
        for layer_name in self.adapter_layer_names:
            adapters = getattr(self, layer_name)
            if adapter_name in adapters:
                parameters.extend(adapters[adapter_name].parameters())
        return parameters

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the channels of the active adapters into the base FFN, making it wider.

        Args:
            safe_merge (`bool`, *optional*):
                If `True`, check that the adapter weights are finite before merging them. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If `None`, all active adapters will be merged.
                Defaults to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        for adapter_name in adapter_names:
            if adapter_name not in self._get_available_adapters():
                continue

            down_biases = self._get_down_biases(adapter_name)
            if safe_merge:
                # Merging concatenates the adapter weights to the base weights, so it suffices to check the adapter
                # weights. They are cast to the dtype of the base weights, where large values can overflow, e.g. in
                # float16. Only the down bias is added to that of the base FFN, so check the sum as well.
                _, dtype = self._get_base_layer_device_and_dtype(self.get_base_layer())
                is_finite = all(torch.isfinite(p.to(dtype)).all() for p in self._get_adapter_parameters(adapter_name))
                if down_biases is not None:
                    base_bias, bias = down_biases
                    is_finite = is_finite and torch.isfinite(base_bias + bias.to(base_bias)).all().item()
                if not is_finite:
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {adapter_name} seems to be broken"
                    )

            with torch.no_grad():
                self._append_channels(adapter_name)
                if down_biases is not None:
                    base_bias, bias = down_biases
                    base_bias.add_(bias.to(base_bias))
            self.merged_adapters.append(adapter_name)

    def unmerge(self) -> None:
        """
        Remove all merged channels from the base FFN, restoring its original weights.

        The down biases of the adapters are subtracted from the down bias of the base FFN, which restores it up to
        floating point errors.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while self.merged_adapters:
            adapter_name = self.merged_adapters.pop()
            with torch.no_grad():
                self._remove_channels(adapter_name)
                down_biases = self._get_down_biases(adapter_name)
                if down_biases is not None:
                    base_bias, bias = down_biases
                    base_bias.sub_(bias.to(base_bias))

    def _append_channels(self, adapter_name: str) -> None:
        raise NotImplementedError

    def _remove_channels(self, adapter_name: str) -> None:
        raise NotImplementedError

    def _get_down_biases(self, adapter_name: str) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Return the down bias of the base FFN and that of the adapter, or `None` if the adapter has no down bias."""
        raise NotImplementedError


class CortexMLP(nn.Module, CortexLayer):
    """Cortex layer for a dense FFN, see `is_dense_ffn`.

    The added channels are computed like the channels of the base FFN, using its activation function, and their output
    is added to the output of the base FFN. The gate and up weights of the added channels are stored separately, even
    if the base FFN has a fused `gate_up_proj`.
    """

    adapter_layer_names = ("cortex_gate", "cortex_up", "cortex_down")

    def __init__(self, base_layer: nn.Module, adapter_name: str, config: CortexConfig) -> None:
        super().__init__()
        CortexLayer.__init__(self, base_layer)
        # Like other PEFT layers, expose the input and output features of the base layer.
        up_proj = base_layer.gate_up_proj if hasattr(base_layer, "gate_up_proj") else base_layer.up_proj
        self.in_features, self.out_features = up_proj.in_features, base_layer.down_proj.out_features

        self.cortex_gate = nn.ModuleDict()
        self.cortex_up = nn.ModuleDict()
        self.cortex_down = nn.ModuleDict()
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config)

    @property
    def has_gate(self) -> bool:
        base = self.get_base_layer()
        return hasattr(base, "gate_proj") or hasattr(base, "gate_up_proj")

    def update_layer(self, adapter_name: str, config: CortexConfig) -> None:
        base = self.get_base_layer()
        if hasattr(base, "gate_up_proj"):
            gate_proj = up_proj = base.gate_up_proj
        else:
            gate_proj, up_proj = getattr(base, "gate_proj", None), base.up_proj

        # With `cortex_bias="auto"`, each projection has a bias if the corresponding projection of the base FFN has
        # one, so that the merged FFN has the same architecture and sampled channels are exact copies of native
        # channels.
        use_bias = config.cortex_bias == "auto"
        self.cortex_up[adapter_name] = nn.Linear(
            up_proj.in_features, config.k, bias=use_bias and up_proj.bias is not None
        )
        if self.has_gate:
            self.cortex_gate[adapter_name] = nn.Linear(
                gate_proj.in_features, config.k, bias=use_bias and gate_proj.bias is not None
            )
        self.cortex_down[adapter_name] = nn.Linear(
            config.k, base.down_proj.out_features, bias=use_bias and base.down_proj.bias is not None
        )

        self.reset_cortex_parameters(adapter_name, config)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=config.inference_mode)

    @torch.no_grad()
    def reset_cortex_parameters(self, adapter_name: str, config: CortexConfig) -> None:
        if any(p.is_meta for p in self._get_adapter_parameters(adapter_name)):
            # The weights will be loaded from a checkpoint (`low_cpu_mem_usage=True`).
            return

        if config.init_weights is False:
            # Keep the random initialization of all projections, so that the adapter changes the output (for testing).
            return

        # The gate and up projections keep the default initialization of nn.Linear, unless they are sampled.
        down = self.cortex_down[adapter_name]
        nn.init.zeros_(down.weight)
        if down.bias is not None:
            nn.init.zeros_(down.bias)
        if config.init_weights == "sample":
            self._sample_channels(adapter_name, config.init_noise_std)

    def _sample_channels(self, adapter_name: str, noise_std: float) -> None:
        """Copy the gate and up weights of randomly chosen channels of the base FFN."""
        base = self.get_base_layer()
        projections = [getattr(base, name) for name in ("gate_proj", "up_proj", "gate_up_proj") if hasattr(base, name)]
        base_parameters = [p for projection in projections for p in projection.parameters()]
        # With DeepSpeed ZeRO-3, the base weights are partitioned, so gather them to copy their channels.
        with gather_params_ctx(base_parameters, modifier_rank=None):
            base_weights = _get_dense_gate_up(base)
            # Use the same channels for gate and up, so that each sampled channel is a copy of a complete native
            # channel.
            channel_idx = _sample_indices(base_weights["up"][0].shape[0], self.cortex_up[adapter_name].out_features)
            adapter_projections = {"gate": self.cortex_gate, "up": self.cortex_up}
            for name, (weight, bias) in base_weights.items():
                projection = adapter_projections[name][adapter_name]
                sampled_weight = weight[channel_idx.to(weight.device)]
                projection.weight.copy_(sampled_weight + noise_std * torch.randn_like(sampled_weight))
                if projection.bias is not None:
                    projection.bias.copy_(bias[channel_idx.to(bias.device)])

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for adapter_name in self.active_adapters:
                # Skip the adapters that don't adapt this FFN and the merged ones, which are already part of it.
                if adapter_name not in self.cortex_up or adapter_name in self.merged_adapters:
                    continue
                result = result + self._compute_channel_output(x, adapter_name).to(result.dtype)

        return result

    def _compute_channel_output(self, x: torch.Tensor, adapter_name: str) -> torch.Tensor:
        base = self.get_base_layer()
        up = self.cortex_up[adapter_name]
        x = self._cast_input_dtype(x, up.weight.dtype)
        if self.has_gate:
            hidden = _apply_dense_gate(base, self.cortex_gate[adapter_name](x), up(x))
        else:
            hidden = base.act_fn(up(x))
        return self.cortex_down[adapter_name](hidden)

    def _append_channels(self, adapter_name: str) -> None:
        up = self.cortex_up[adapter_name]
        gate = self.cortex_gate[adapter_name] if self.has_gate else None
        _append_dense_channels(
            self.get_base_layer(),
            up=up.weight,
            down=self.cortex_down[adapter_name].weight,
            gate=gate.weight if gate is not None else None,
            up_bias=up.bias,
            gate_bias=gate.bias if gate is not None else None,
        )

    def _remove_channels(self, adapter_name: str) -> None:
        _remove_dense_channels(self.get_base_layer(), self.cortex_up[adapter_name].out_features)

    def _get_down_biases(self, adapter_name: str) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        bias = self.cortex_down[adapter_name].bias
        return (self.get_base_layer().down_proj.bias, bias) if bias is not None else None

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "cortex." + rep


class PackedChannels(nn.Module):
    """The weights of the channels that Cortex adds to packed MoE experts.

    `up` holds the gate and up weights in the same layout as the `gate_up_proj` of the base experts (concatenated or
    interleaved), so that the `_apply_gate` method of the experts can be applied to its output. If the experts have no
    gate, it only holds the up weights. All weights are stored untransposed, i.e. as `(out_features, in_features)`.
    With `has_bias`, `up_bias` and `down_bias` hold the biases of the up and down projections. With `num_experts`,
    every weight and bias has an additional leading expert dimension.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        width: int,
        has_gate: bool,
        has_bias: bool,
        num_experts: Optional[int] = None,
    ) -> None:
        super().__init__()
        expert_dims = () if num_experts is None else (num_experts,)
        up_features = 2 * width if has_gate else width
        self.up = nn.Parameter(torch.empty(*expert_dims, up_features, in_features))
        self.up_bias = nn.Parameter(torch.empty(*expert_dims, up_features)) if has_bias else None
        self.down = nn.Parameter(torch.empty(*expert_dims, out_features, width))
        self.down_bias = nn.Parameter(torch.empty(*expert_dims, out_features)) if has_bias else None

    @property
    def width(self) -> int:
        return self.down.shape[-1]

    def extra_repr(self) -> str:
        # These weights are plain parameters, which nn.Module does not show in its representation.
        return f"up={tuple(self.up.shape)}, down={tuple(self.down.shape)}, bias={self.up_bias is not None}"


class CortexExperts(nn.Module, CortexLayer):
    """Cortex layer for packed MoE experts that follow the Transformers experts interface.

    The added channels are computed like native expert channels, using the `_apply_gate` method of the experts (e.g.
    the clamped SwiGLU of GPT-OSS). Routed channels belong to individual experts: they are applied to the tokens routed
    to their expert and weighted by the routing weight, exactly like the native channels of that expert. Public
    channels are common to all experts: they are applied to all tokens and weighted by the sum of the routing weights
    of each token, which is the same as adding them to every expert. They take the place of native shared experts in
    MoE blocks without them, see `moe_target`.

    Like the base experts, the forward method expects the flattened hidden states, the indices of the selected experts,
    and their routing weights as positional arguments, which is how the Transformers MoE blocks call their experts.
    """

    adapter_layer_names = ("cortex_routed", "cortex_public")

    def __init__(
        self, base_layer: nn.Module, adapter_name: str, config: CortexConfig, add_public: bool, add_routed: bool
    ) -> None:
        super().__init__()
        CortexLayer.__init__(self, base_layer)
        # Like other PEFT layers, expose the input and output features of the base layer, i.e. of each expert.
        up_shape = _get_full_shape(getattr(base_layer, _get_up_name(base_layer)))
        down_shape = _get_full_shape(base_layer.down_proj)
        if base_layer.is_transposed:
            self.in_features, self.out_features = up_shape[-2], down_shape[-1]
        else:
            self.in_features, self.out_features = up_shape[-1], down_shape[-2]

        self.cortex_routed = nn.ModuleDict()
        self.cortex_public = nn.ModuleDict()
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config, add_public=add_public, add_routed=add_routed)

    def _get_branches(self, adapter_name: str) -> list[tuple[PackedChannels, bool]]:
        """The channels added by an adapter as `(channels, is_routed)` pairs, in the order in which they are merged."""
        branches = []
        if adapter_name in self.cortex_routed:
            branches.append((self.cortex_routed[adapter_name], True))
        if adapter_name in self.cortex_public:
            branches.append((self.cortex_public[adapter_name], False))
        return branches

    def update_layer(self, adapter_name: str, config: CortexConfig, add_public: bool, add_routed: bool) -> None:
        # Which branches to add depends on `moe_target` and on the MoE block of the experts, see
        # `CortexModel._create_and_replace`.
        base = self.get_base_layer()

        # As for dense FFNs, the projections have biases if those of the base experts have them, see `cortex_bias`.
        has_bias = base.has_bias and config.cortex_bias == "auto"

        def create_channels(width, num_experts=None):
            return PackedChannels(self.in_features, self.out_features, width, base.has_gate, has_bias, num_experts)

        if add_routed:
            width = config.k_routed if config.k_routed is not None else config.k
            self.cortex_routed[adapter_name] = create_channels(width, num_experts=base.num_experts)
        if add_public:
            self.cortex_public[adapter_name] = create_channels(config.k)

        self.reset_cortex_parameters(adapter_name, config)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=config.inference_mode)

    @torch.no_grad()
    def reset_cortex_parameters(self, adapter_name: str, config: CortexConfig) -> None:
        if any(p.is_meta for p in self._get_adapter_parameters(adapter_name)):
            # The weights will be loaded from a checkpoint (`low_cpu_mem_usage=True`).
            return

        for channels, _ in self._get_branches(adapter_name):
            # Same as the default initialization of nn.Linear, applied to the weights of each expert.
            bound = 1 / math.sqrt(channels.up.shape[-1])
            nn.init.uniform_(channels.up, -bound, bound)
            if channels.up_bias is not None:
                nn.init.uniform_(channels.up_bias, -bound, bound)

            if config.init_weights is False:
                # Random down weights and biases, so that the adapter changes the output (for testing).
                bound = 1 / math.sqrt(channels.width)
                nn.init.uniform_(channels.down, -bound, bound)
                if channels.down_bias is not None:
                    nn.init.uniform_(channels.down_bias, -bound, bound)
            else:
                nn.init.zeros_(channels.down)
                if channels.down_bias is not None:
                    nn.init.zeros_(channels.down_bias)

        if config.init_weights == "sample":
            self._sample_channels(adapter_name, config.init_noise_std)

    def _sample_channels(self, adapter_name: str, noise_std: float) -> None:
        """Copy the gate and up weights of randomly chosen channels of the base experts.

        Routed channels are sampled from the channels of their own expert, public channels from those of all experts.
        """
        base = self.get_base_layer()
        up_name = _get_up_name(base)
        base_bias = getattr(base, up_name + "_bias", None)
        base_parameters = [p for p in (getattr(base, up_name), base_bias) if p is not None]
        # With DeepSpeed ZeRO-3, the base weights are partitioned, so gather them to copy their channels.
        with gather_params_ctx(base_parameters, modifier_rank=None):
            base_weight = _get_packed_weight(base, up_name)
            num_experts = base_weight.shape[0]
            intermediate_size = base_weight.shape[-2] // 2 if base.has_gate else base_weight.shape[-2]

            for channels, is_routed in self._get_branches(adapter_name):
                if is_routed:
                    expert_idx = torch.arange(num_experts)[:, None].expand(-1, channels.width)
                    channel_idx = torch.stack(
                        [_sample_indices(intermediate_size, channels.width) for _ in range(num_experts)]
                    )
                else:
                    flat_idx = _sample_indices(num_experts * intermediate_size, channels.width)
                    expert_idx, channel_idx = flat_idx // intermediate_size, flat_idx % intermediate_size

                expert_idx, row_idx = self._get_gate_up_rows(expert_idx, channel_idx, intermediate_size)
                expert_idx, row_idx = expert_idx.to(base_weight.device), row_idx.to(base_weight.device)
                weight = base_weight[expert_idx, row_idx]
                channels.up.copy_(weight + noise_std * torch.randn_like(weight))
                if channels.up_bias is not None:
                    channels.up_bias.copy_(base_bias[expert_idx, row_idx])

    def _get_gate_up_rows(
        self, expert_idx: torch.Tensor, channel_idx: torch.Tensor, intermediate_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find the rows of the (gate_)up weight that belong to the given channels of the given experts.

        In the concatenated layout, the gate rows of all channels come before their up rows. In the interleaved layout,
        the gate and up rows of each channel are adjacent. The returned expert and row indices list the selected rows
        in the same layout, so the selected rows can directly be used as adapter weights.
        """
        base = self.get_base_layer()
        if not base.has_gate:
            return expert_idx, channel_idx

        if base.is_concatenated:
            row_idx = torch.cat([channel_idx, channel_idx + intermediate_size], dim=-1)
            return torch.cat([expert_idx, expert_idx], dim=-1), row_idx

        row_idx = torch.stack([2 * channel_idx, 2 * channel_idx + 1], dim=-1).flatten(-2)
        return expert_idx.repeat_interleave(2, dim=-1), row_idx

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(hidden_states, top_k_index, top_k_weights, *args, **kwargs)
        else:
            result = self.base_layer(hidden_states, top_k_index, top_k_weights, *args, **kwargs)
            for adapter_name in self.active_adapters:
                # Merged adapters are already part of the base experts.
                if adapter_name in self.merged_adapters:
                    continue

                if adapter_name in self.cortex_routed:
                    channels = self.cortex_routed[adapter_name]
                    output = self._compute_routed_output(hidden_states, top_k_index, top_k_weights, channels)
                    result = result + output.to(result.dtype)

                if adapter_name in self.cortex_public:
                    channels = self.cortex_public[adapter_name]
                    x = self._cast_input_dtype(hidden_states, channels.up.dtype)
                    output = self._compute_channel_output(
                        x, channels.up, channels.up_bias, channels.down, channels.down_bias
                    )
                    # Weighting the public channels by the sum of the routing weights is the same as adding them to
                    # every selected expert, which is how they are merged. This makes merging exact for every router.
                    output = output * top_k_weights.sum(dim=-1, keepdim=True).to(output.dtype)
                    result = result + output.to(result.dtype)

        return result

    def _compute_channel_output(
        self,
        x: torch.Tensor,
        up: torch.Tensor,
        up_bias: Optional[torch.Tensor],
        down: torch.Tensor,
        down_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Compute the added channels like the base experts compute their channels.
        base = self.get_base_layer()
        hidden = F.linear(x, up, up_bias)
        hidden = base._apply_gate(hidden) if base.has_gate else base.act_fn(hidden)
        return F.linear(hidden, down, down_bias)

    def _compute_routed_output(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        channels: PackedChannels,
    ) -> torch.Tensor:
        """Compute the output of the routed channels in the same way as the native experts compute theirs.

        Each token is processed by the channels of each of its selected experts. Their outputs are weighted by the
        routing weights and summed.
        """
        x = self._cast_input_dtype(hidden_states, channels.up.dtype)
        output = torch.zeros(x.shape[0], channels.down.shape[-2], dtype=x.dtype, device=x.device)

        # Split the packed weights once. Indexing them separately for every expert would allocate a full-size gradient
        # per expert in the backward pass.
        ups, downs = channels.up.unbind(0), channels.down.unbind(0)
        up_biases = channels.up_bias.unbind(0) if channels.up_bias is not None else [None] * len(ups)
        down_biases = channels.down_bias.unbind(0) if channels.down_bias is not None else [None] * len(downs)

        for expert in top_k_index.unique().tolist():
            if not 0 <= expert < len(ups):
                # Like the native experts, skip the out-of-range indices that are used with expert parallelism.
                continue
            token_idx, slot_idx = torch.where(top_k_index == expert)
            # Like the native down bias, the down bias of the expert is weighted by the routing weight as well.
            expert_output = self._compute_channel_output(
                x[token_idx], ups[expert], up_biases[expert], downs[expert], down_biases[expert]
            )
            routing_weights = top_k_weights[token_idx, slot_idx, None].to(expert_output.dtype)
            # Under autocast, e.g. with mixed precision training, the channels compute in a lower precision than
            # `output`, and `index_add_` requires the same dtype. So cast like the native experts do.
            output.index_add_(0, token_idx, (expert_output * routing_weights).to(output.dtype))
        return output

    def _append_channels(self, adapter_name: str) -> None:
        base = self.get_base_layer()
        for channels, is_routed in self._get_branches(adapter_name):
            up, up_bias, down = channels.up, channels.up_bias, channels.down
            if not is_routed:
                # The public channels are added to every expert, see `forward`.
                up = up.expand(base.num_experts, *up.shape)
                down = down.expand(base.num_experts, *down.shape)
                if up_bias is not None:
                    up_bias = up_bias.expand(base.num_experts, *up_bias.shape)

            _append_expert_channels(base, up, down, up_bias)

    def _remove_channels(self, adapter_name: str) -> None:
        num_channels = sum(channels.width for channels, _ in self._get_branches(adapter_name))
        _remove_expert_channels(self.get_base_layer(), num_channels)

    def _get_down_biases(self, adapter_name: str) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        biases = [
            channels.down_bias for channels, _ in self._get_branches(adapter_name) if channels.down_bias is not None
        ]
        # The down bias of the channels common to all experts has no expert dimension, so broadcasting adds it to the
        # down bias of every expert, like their other weights in `_append_channels`.
        return (self.get_base_layer().down_proj_bias, sum(biases)) if biases else None

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "cortex." + rep
