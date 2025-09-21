# Copyright 2025-present the HuggingFace Inc. team.
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
from typing import Any, Literal, Optional

import torch
import torch.nn as nn

from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


class DeloraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names: tuple[str, ...] = ("delora_A", "delora_B", "delora_lambda")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names: tuple[str, ...] = (
        "r",
        "delora_dropout",
        # Persisted initial copies and norms
        "delora_initial_A",
        "delora_initial_B",
        "delora_initial_lambda",
        "delora_w_norm",
        "delora_initial_w_norm",
    )

    def __init__(self, base_layer: nn.Module, use_residual_init: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.delora_dropout = nn.ModuleDict({})
        self.delora_A = nn.ParameterDict({})
        self.delora_B = nn.ParameterDict({})
        self.delora_lambda = nn.ParameterDict({})
        # Store flag from kwargs (defaults to False if not provided)
        self.use_residual_init = use_residual_init
        # Non-trainable initial copies so that delta is zero at init
        # Use persistent buffers so they are included in state_dict and saved.
        self.delora_initial_A = BufferDict({}, persistent=True)
        self.delora_initial_B = BufferDict({}, persistent=True)
        # store lambdas as tensor buffer (shape [1]) for easier device moves and persistence
        self.delora_initial_lambda = BufferDict({}, persistent=True)
        self.delora_initial_w_norm = BufferDict({}, persistent=True)  # initial base weight norm per adapter
        self.delora_w_norm = BufferDict(
            {}, persistent=True
        )  # fixed reference norm per-adapter used for delta computation
        # Track which adapters had their initial delta subtracted from base weights
        self._initials_applied = set()
        # Track nesting depth of disable->enable toggling to avoid double-adding/subtracting
        self._disable_depth = 0
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer_mod = self.get_base_layer()
        if isinstance(base_layer_mod, nn.Linear):
            self.in_features, self.out_features = base_layer_mod.in_features, base_layer_mod.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer_mod)}")

    @staticmethod
    def _compute_delta(
        A: torch.Tensor, B: torch.Tensor, lambda_: torch.Tensor, r: int, w_norm: torch.Tensor
    ) -> torch.Tensor:
        """Compute delta = B @ diag(lambda_/r / (||A_i||*||B^j||)) @ A, scaled by provided w_norm (per-input channel)"""
        An = torch.clamp(A.norm(dim=1), min=1e-12)
        Bn = torch.clamp(B.norm(dim=0), min=1e-12)
        diag = torch.diag_embed((lambda_ / r) / (An * Bn))
        delta = B @ diag @ A
        delta = delta * w_norm.unsqueeze(0)
        return delta

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        if adapter not in self.delora_A or adapter not in self.delora_B:
            raise ValueError(f"Adapter {adapter} not found.")

        delta = self._compute_delta(
            self.delora_A[adapter],
            self.delora_B[adapter],
            self.delora_lambda[adapter],
            self.r[adapter],
            self.delora_w_norm[adapter],
        )
        # When not using residual init, subtract a frozen copy of the initial delta to keep effective identity init
        if (not self.use_residual_init) and (adapter in self.delora_initial_A and adapter in self.delora_initial_B):
            delta0 = self._compute_delta(
                self.delora_initial_A[adapter],
                self.delora_initial_B[adapter],
                self.delora_initial_lambda[adapter],
                self.r[adapter],
                self.delora_initial_w_norm[adapter],
            )
            delta = delta - delta0
        return delta

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lambda_: float,
        module_dropout: float,
        init_weights: bool = True,
        use_residual_init: Optional[bool] = None,
        inference_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        if use_residual_init is not None:
            # allow toggling via update for already-wrapped layers
            self.use_residual_init = use_residual_init
        if module_dropout > 0.0:
            self.delora_dropout[adapter_name] = nn.Dropout(p=module_dropout)
        else:
            self.delora_dropout[adapter_name] = nn.Identity()

        self.delora_A[adapter_name] = nn.Parameter(torch.empty(r, self.in_features))
        self.delora_B[adapter_name] = nn.Parameter(torch.empty(self.out_features, r))
        self.delora_lambda[adapter_name] = nn.Parameter(torch.full((1,), float(lambda_)))

        # initialize and store initial copies
        self.reset_delora_parameters(adapter_name, init_weights, lambda_)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    @torch.no_grad()
    def reset_delora_parameters(
        self,
        adapter_name: str,
        init_weights: bool | Literal["kaiming_uniform", "gaussian", "xavier_uniform"] = True,
        lambda_: float = 1.0,
    ) -> None:
        if adapter_name not in self.delora_A.keys():
            return

        if init_weights is True:
            nn.init.kaiming_uniform_(self.delora_A[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.delora_B[adapter_name], a=math.sqrt(5))
        else:
            # fill with small noise to break symmetry; user explicitly requested non-identity start
            nn.init.normal_(self.delora_A[adapter_name], mean=0.0, std=0.02)
            nn.init.normal_(self.delora_B[adapter_name], mean=0.0, std=0.02)

        # capture a fixed norm for this adapter to use for future delta computations
        with torch.no_grad():
            w = self.get_base_layer().weight
            if w.device.type != "meta":
                w_norm = torch.norm(w.data, dim=0).detach()
            else:
                # For meta tensors, we can't compute the norm, so use a default value
                w_norm = torch.ones(w.shape[1], device=w.device)  # , dtype=torch.float32)
            self.delora_w_norm[adapter_name] = w_norm
            self.delora_initial_w_norm[adapter_name] = w_norm.clone()

        # initial copies and lambda (kept non-trainable). Stored on CPU only when use_residual_init=True
        if init_weights is not False:
            if self.delora_A[adapter_name].device.type != "meta":
                if self.use_residual_init:
                    # For real tensors, copy the actual initialized data
                    self.delora_initial_A[adapter_name] = (
                        self.delora_A[adapter_name].detach().to(copy=True, non_blocking=False).cpu()
                    )
                    self.delora_initial_B[adapter_name] = (
                        self.delora_B[adapter_name].detach().to(copy=True, non_blocking=False).cpu()
                    )
                else:
                    # For real tensors, copy to the target device
                    self.delora_initial_A[adapter_name] = (
                        self.delora_A[adapter_name].detach().to(device=w.device, copy=True)
                    )
                    self.delora_initial_B[adapter_name] = (
                        self.delora_B[adapter_name].detach().to(device=w.device, copy=True)
                    )
                self.delora_initial_lambda[adapter_name] = torch.tensor(
                    [float(self.delora_lambda[adapter_name].detach().item())], device=w.device
                )
            else:
                # For meta tensors, create meta tensors for initial copies too
                self.delora_initial_A[adapter_name] = torch.empty_like(self.delora_A[adapter_name], device="meta")
                self.delora_initial_B[adapter_name] = torch.empty_like(self.delora_B[adapter_name], device="meta")
                self.delora_initial_lambda[adapter_name] = torch.tensor([float(lambda_)], device="meta")

            if self.use_residual_init and adapter_name not in self._initials_applied:
                # store initial base weight norm
                self.delora_initial_w_norm[adapter_name] = w_norm
                # subtract initial delta using the reference norm (only if not on meta device)
                if self.delora_initial_A[adapter_name].device.type != "meta" and w.device.type != "meta":
                    with torch.no_grad():
                        # subtract exact initial delta so effective initial effect is zero
                        delta0 = self._compute_delta(
                            self.delora_initial_A[adapter_name].to(w.device),
                            self.delora_initial_B[adapter_name].to(w.device),
                            self.delora_initial_lambda[adapter_name].to(w.device),
                            self.r[adapter_name],
                            self.delora_initial_w_norm[adapter_name].to(w.device),
                        )
                        self.get_base_layer().weight.data.sub_(delta0)
                    self._initials_applied.add(adapter_name)
                else:
                    # For meta tensors, just mark as applied but don't actually subtract
                    self._initials_applied.add(adapter_name)

        else:
            if adapter_name in self.delora_initial_A:
                del self.delora_initial_A[adapter_name]
            if adapter_name in self.delora_initial_B:
                del self.delora_initial_B[adapter_name]
            if adapter_name in self.delora_initial_lambda:
                del self.delora_initial_lambda[adapter_name]

    def _move_adapter_to_device_of_base_layer(self, adapter_name: str, device: Optional[torch.device] = None) -> None:
        """
        Move the adapter of the given name to the device of the base layer.
        Override to handle meta tensors properly.
        """
        if device is None:
            base_layer = self.get_base_layer()
            if isinstance(base_layer, nn.MultiheadAttention):
                base_layer = base_layer.out_proj
            # check weight and qweight (for GPTQ)
            for weight_name in ("weight", "qweight"):
                weight = getattr(base_layer, weight_name, None)
                if weight is not None:
                    device = weight.device
                    dtype = weight.dtype
                    break
            else:
                # no break encountered: could not determine the device
                return

        meta = torch.device("meta")
        # loop through all potential adapter layers and move them to the device of the base layer; be careful to only
        # move this specific adapter to the device, as the other adapters could be on different devices
        # see #1639
        for adapter_layer_name in self.adapter_layer_names + self.other_param_names:
            adapter_layer = getattr(self, adapter_layer_name, None)
            if not isinstance(adapter_layer, (nn.ModuleDict, nn.ParameterDict, BufferDict)):
                continue
            if adapter_name not in adapter_layer:
                continue

            # Check if the adapter layer contains meta tensors
            if hasattr(adapter_layer[adapter_name], "device") and adapter_layer[adapter_name].device == meta:
                # Skip moving meta tensors - they'll be materialized later
                continue

            # For ParameterDict, check individual parameters
            if hasattr(adapter_layer[adapter_name], "parameters"):
                if any(p.device == meta for p in adapter_layer[adapter_name].parameters()):
                    continue

            # For modules, check buffers
            if hasattr(adapter_layer[adapter_name], "buffers"):
                if any(b.device == meta for b in adapter_layer[adapter_name].buffers()):
                    continue

            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device, dtype=dtype)


class DeloraLinear(nn.Module, DeloraLayer):
    # DeLoRA implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int,
        lambda_: float,
        module_dropout: float,
        init_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        DeloraLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lambda_, module_dropout, init_weights)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.delora_A.keys():
                base_layer = self.get_base_layer()
                delta_weight = (
                    self.get_delta_weight(active_adapter)
                    .detach()
                    .to(dtype=base_layer.weight.dtype, device=base_layer.weight.device)
                )
                with torch.no_grad():
                    if safe_merge:
                        orig_weights = base_layer.weight.data.clone()
                        orig_weights = orig_weights + delta_weight

                        if not torch.isfinite(orig_weights).all():
                            raise ValueError(
                                f"NaNs detected in merged weights for adapter {active_adapter}; aborting merge"
                            )

                        base_layer.weight.data = orig_weights
                    else:
                        base_layer.weight.data.add_(delta_weight)

                self.merged_adapters.append(active_adapter)

    def unmerge(self, unapply_initials=False) -> None:
        """
        Unmerge all merged adapter layers from the base weights.

        Args:
            unapply_initials (`bool`, *optional*, defaults to `False`):
                If True, and if `use_residual_init=True`, also remove the initial deltas that were subtracted at setup time from the
                base weights. This effectively reverts the base weights to their original state before wrapping with DeloraLayer.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.delora_A.keys():
                with torch.no_grad():
                    delta_weight = self.get_delta_weight(active_adapter)
                    self.get_base_layer().weight.data.sub_(delta_weight.to(self.get_base_layer().weight.dtype))

        # Reverting to original base weights, if use_residual_init=True
        if unapply_initials and self.use_residual_init and len(self._initials_applied) > 0:
            w = self.get_base_layer().weight
            for adapter in self._initials_applied:
                if (
                    (adapter in self.delora_initial_A)
                    and (adapter in self.delora_initial_B)
                    and (adapter in self.delora_initial_w_norm)
                ):
                    # add back exact initial delta so effective initial effect is zero
                    delta0 = self._compute_delta(
                        self.delora_initial_A[adapter].to(w.device),
                        self.delora_initial_B[adapter].to(w.device),
                        self.delora_initial_lambda[adapter].to(w.device),
                        self.r[adapter],
                        self.delora_initial_w_norm[adapter].to(w.device),
                    )
                    self.get_base_layer().weight.data = self.get_base_layer().weight.data + delta0.to(w.dtype)
                self._initials_applied.remove(adapter)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            if not self.active_adapters:
                return self.base_layer(x, *args, **kwargs).to(previous_dtype)

            base_out = self.base_layer(x, *args, **kwargs)
            add_out = torch.zeros_like(base_out)

            for adapter in self.active_adapters:
                if adapter not in self.delora_A:
                    continue

                x_d = self.delora_dropout[adapter](x)

                # Decomposed delta calculation
                # 1. (x * w_norm) @ A.T
                h = nn.functional.linear(x_d * self.delora_w_norm[adapter], self.delora_A[adapter])

                # 2. h @ diag
                An = torch.clamp(self.delora_A[adapter].norm(dim=1), min=1e-12)
                Bn = torch.clamp(self.delora_B[adapter].norm(dim=0), min=1e-12)
                scaling = (self.delora_lambda[adapter] / self.r[adapter]) / (An * Bn)
                h = h * scaling

                # 3. h @ B.T
                h = nn.functional.linear(h, self.delora_B[adapter])

                # When not using residual init, subtract a frozen copy of the initial delta to keep effective identity init
                if (not self.use_residual_init) and (
                    adapter in self.delora_initial_A and adapter in self.delora_initial_B
                ):
                    # 1. (initial)
                    h0 = nn.functional.linear(
                        x_d * self.delora_initial_w_norm[adapter], self.delora_initial_A[adapter]
                    )
                    # 2. (initial)
                    A0n = torch.clamp(self.delora_initial_A[adapter].norm(dim=1), min=1e-12)
                    B0n = torch.clamp(self.delora_initial_B[adapter].norm(dim=0), min=1e-12)
                    scaling0 = (self.delora_initial_lambda[adapter] / self.r[adapter]) / (A0n * B0n)
                    h0 = h0 * scaling0
                    # 3. (initial)
                    h0 = nn.functional.linear(h0, self.delora_initial_B[adapter])
                    h = h - h0

                add_out += h

            result = base_out + add_out.to(base_out.dtype)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "delora." + rep

    # Helpers to handle disable/enable with use_residual_init
    @torch.no_grad()
    def _apply_initials_to_base(self, sign: int) -> None:
        """Apply (sign=+1) or remove (sign=-1) the sum of all initial deltas to the base weight.

        Only used when use_residual_init=True and the layer is not merged. Uses the stored initial
        A/B/lambda and initial_w_norm per adapter to reconstruct the exact initial delta that was
        subtracted at setup time.
        """
        if not self._initials_applied:
            return
        w = self.get_base_layer().weight
        total = torch.zeros_like(w)
        for adapter in self._initials_applied:
            if (
                (adapter in self.delora_initial_A)
                and (adapter in self.delora_initial_B)
                and (adapter in self.delora_initial_w_norm)
                and (adapter in self.r)
                and (adapter in self.delora_initial_lambda)
            ):
                delta0 = self._compute_delta(
                    self.delora_initial_A[adapter].to(w.device),
                    self.delora_initial_B[adapter].to(w.device),
                    self.delora_initial_lambda[adapter].to(w.device),
                    self.r[adapter],
                    self.delora_initial_w_norm[adapter].to(w.device),
                )
                total = total + delta0
        if sign > 0:
            w.data = w.data + total.to(w.dtype)
        else:
            w.data = w.data - total.to(w.dtype)

    # Adjust base weights after loading to keep outputs invariant across save/load
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # If use_residual_init=True, we need to handle base weight restoration carefully.
        # When a DeloraLayer is initialized, it subtracts a fresh, random initial delta.
        # We need to undo this before applying the saved initial delta from the state_dict.
        if self.use_residual_init:
            with torch.no_grad():
                w = self.get_base_layer().weight
                if w.device.type != "meta":
                    # Add back the freshly subtracted random initial delta to restore the original base weight.
                    for adapter_name in self._initials_applied:
                        if (
                            adapter_name in self.delora_initial_A
                            and adapter_name in self.delora_initial_B
                            and adapter_name in self.delora_initial_lambda
                            and adapter_name in self.delora_initial_w_norm
                        ):
                            # Skip if any of the initial tensors are meta tensors
                            if (
                                self.delora_initial_A[adapter_name].device.type == "meta"
                                or self.delora_initial_B[adapter_name].device.type == "meta"
                                or self.delora_initial_lambda[adapter_name].device.type == "meta"
                                or self.delora_initial_w_norm[adapter_name].device.type == "meta"
                            ):
                                continue
                            fresh_delta = self._compute_delta(
                                self.delora_initial_A[adapter_name].to(w.device),
                                self.delora_initial_B[adapter_name].to(w.device),
                                self.delora_initial_lambda[adapter_name].to(w.device),
                                self.r[adapter_name],
                                self.delora_initial_w_norm[adapter_name].to(w.device),
                            )
                            w.data += fresh_delta.to(w.dtype)

        # Create a filtered state dict with only keys that belong to this module
        module_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(prefix):
                # Remove the prefix to get the relative key
                relative_key = key[len(prefix) :]
                module_state_dict[relative_key] = value

        # Manually load ParameterDict and BufferDict parameters
        for param_name in ["delora_A", "delora_B", "delora_lambda"]:
            param_dict = getattr(self, param_name)
            for adapter_name in param_dict.keys():
                key = f"{param_name}.{adapter_name}"
                if key in module_state_dict:
                    param_dict[adapter_name].data.copy_(module_state_dict[key])
                    del module_state_dict[key]  # Remove so it doesn't interfere with regular loading

        # Manually load BufferDict parameters
        for buffer_name in [
            "delora_initial_A",
            "delora_initial_B",
            "delora_initial_lambda",
            "delora_w_norm",
            "delora_initial_w_norm",
        ]:
            if hasattr(self, buffer_name):
                buffer_dict = getattr(self, buffer_name)
                for adapter_name in list(buffer_dict.keys()):  # Use list() to avoid dict changed during iteration
                    key = f"{buffer_name}.{adapter_name}"
                    if key in module_state_dict:
                        buffer_dict[adapter_name] = module_state_dict[key]
                        del module_state_dict[key]

        # Call the regular PyTorch loading for any remaining parameters
        if module_state_dict:
            # Reconstruct full state dict with prefix for remaining keys
            remaining_state_dict = {prefix + key: value for key, value in module_state_dict.items()}
            nn.Module._load_from_state_dict(
                self, remaining_state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )

        # Subtract the initial delta that was loaded from the checkpoint
        if self.use_residual_init:
            with torch.no_grad():
                w = self.get_base_layer().weight
                if w.device.type != "meta":
                    for adapter_name in self._initials_applied:
                        if (
                            adapter_name in self.delora_initial_A
                            and adapter_name in self.delora_initial_B
                            and adapter_name in self.delora_initial_lambda
                            and adapter_name in self.delora_initial_w_norm
                            and adapter_name in self.r
                        ):
                            # Skip if any of the initial tensors are meta tensors
                            if (
                                self.delora_initial_A[adapter_name].device.type == "meta"
                                or self.delora_initial_B[adapter_name].device.type == "meta"
                                or self.delora_initial_lambda[adapter_name].device.type == "meta"
                                or self.delora_initial_w_norm[adapter_name].device.type == "meta"
                            ):
                                continue
                            loaded_initial_delta = self._compute_delta(
                                self.delora_initial_A[adapter_name].to(w.device),
                                self.delora_initial_B[adapter_name].to(w.device),
                                self.delora_initial_lambda[adapter_name].to(w.device),
                                self.r[adapter_name],
                                self.delora_initial_w_norm[adapter_name].to(w.device),
                            )
                            w.data.sub_(loaded_initial_delta.to(w.dtype))

    def enable_adapters(self, enabled: bool) -> None:
        """Toggle enabling/disabling of adapters.

        When use_residual_init=True, we also need to temporarily restore the initial deltas on disable
        to ensure the base forward reproduces the original base model output. On re-enable, subtract them
        again to return to the steady state where initial deltas are factored out.
        """
        # Let the base class flip the disable flag
        super().enable_adapters(enabled)

        # If we didn't subtract initials at setup, or merged state, there's nothing to compensate
        if not self.use_residual_init or self.merged:
            return

        if not enabled:
            # entering disabled state; only apply once for nested contexts
            if self._disable_depth == 0:
                self._apply_initials_to_base(sign=+1)
            self._disable_depth += 1
        else:
            # leaving disabled state; only revert when the outermost context exits
            if self._disable_depth > 0:
                self._disable_depth -= 1
                if self._disable_depth == 0:
                    self._apply_initials_to_base(sign=-1)
